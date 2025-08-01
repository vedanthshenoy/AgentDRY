import streamlit as st
import asyncio
import nest_asyncio
import os
import sys
import logging
import time
from datetime import datetime

# Apply nest_asyncio to handle asyncio in Streamlit
nest_asyncio.apply()

# Add paths for importing modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'clients', 'gemini_clients')))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from gemini_mcp_client import GeminiLLM, process_query, load_tools, predict_tool_name
from main import create_and_update_tool
from mcp import ClientSession
from mcp.client.sse import sse_client
from google.genai import types
import httpx

# Configure Streamlit page
st.set_page_config(
    page_title="AgentDRY - AI Agent with Dynamic Tools",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        border-bottom: 2px solid #f0f2f6;
        margin-bottom: 2rem;
    }
    .status-success { color: #28a745; font-weight: bold; }
    .status-error { color: #dc3545; font-weight: bold; }
    .status-warning { color: #ffc107; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "past_chats" not in st.session_state:
    st.session_state.past_chats = []
if "tools" not in st.session_state:
    st.session_state.tools = []
if "session" not in st.session_state:
    st.session_state.session = None
if "tool_object" not in st.session_state:
    st.session_state.tool_object = None
if "server_connected" not in st.session_state:
    st.session_state.server_connected = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "llm" not in st.session_state:
    st.session_state.llm = None

# Initialize LLM
@st.cache_resource
def get_llm():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("❌ GEMINI_API_KEY environment variable not found!")
        st.stop()
    return GeminiLLM(api_key=api_key)

# Connect to server and load tools
def connect_to_server():
    try:
        async def _connect():
            async with sse_client(url="http://localhost:8000/sse") as streams:
                async with ClientSession(*streams) as session:
                    await session.initialize()
                    tools_data = await load_tools(session)
                    tool_object = types.Tool(function_declarations=tools_data)
                    return session, tools_data, tool_object, True
        
        return asyncio.run(_connect())
    except Exception as e:
        return None, [], None, False

# Refresh tools
def refresh_tools():
    session, tools_data, tool_object, connected = connect_to_server()
    st.session_state.session = session
    st.session_state.tools = tools_data
    st.session_state.tool_object = tool_object
    st.session_state.server_connected = connected
    return connected

# Process user query
def process_user_query(user_input):
    if not st.session_state.server_connected:
        st.session_state.messages.append({
            "role": "assistant", 
            "content": "❌ Server not connected. Please refresh tools."
        })
        return

    async def _process():
        llm = st.session_state.llm
        session = st.session_state.session
        tools = st.session_state.tool_object
        
        # Step 1: Try existing tools
        tool_called, updated_messages, response_text = await process_query(
            session, llm, tools, user_input, st.session_state.chat_history
        )
        
        st.session_state.chat_history = updated_messages
        
        if tool_called and response_text:
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response_text
            })
            return True
        
        # Step 2: Check if tool creation needed
        needs_tool = await llm.determine_tool_creation_necessity(user_input)
        
        if not needs_tool:
            response = response_text or "I can help with that, but please be more specific."
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response
            })
            return True
        
        # Step 3: Create tool
        qtype, general_part, _ = await llm.classify_and_split_question(user_input)
        tool_topic = general_part if qtype == "direct" else user_input
        proposed_name = predict_tool_name(tool_topic)
        
        # Check if tool exists
        existing_names = {t["name"].lower() for t in st.session_state.tools}
        if proposed_name.lower() in existing_names:
            st.session_state.messages.append({
                "role": "assistant", 
                "content": f"🔧 Tool '{proposed_name}' exists but wasn't used. Try rephrasing."
            })
            return True
        
        # Create the tool
        st.session_state.messages.append({
            "role": "assistant", 
            "content": f"🛠️ Creating tool for: **{tool_topic}**"
        })
        
        return qtype, user_input  # Return for reprocessing
    
    result = asyncio.run(_process())
    
    if result == True:
        return  # Query processed successfully
    
    # Tool creation needed
    qtype, original_query = result
    
    # Actually create the tool
    tool_topic = st.session_state.messages[-1]["content"].split("**")[1]
    create_and_update_tool(f"Create a function for {tool_topic}")
    
    st.session_state.messages.append({
        "role": "assistant", 
        "content": "✅ Tool created! Server restarting..."
    })
    
    # Wait and reprocess for direct queries
    if qtype == "direct":
        time.sleep(8)  # Wait for server restart
        
        st.session_state.messages.append({
            "role": "assistant", 
            "content": "🔄 Reconnecting and processing your request..."
        })
        
        # Refresh connection with fresh session
        if refresh_tools():
            # Try the query again with a completely fresh session
            async def _reprocess():
                try:
                    # Create a brand new session connection
                    async with sse_client(url="http://localhost:8000/sse") as streams:
                        async with ClientSession(*streams) as fresh_session:
                            await fresh_session.initialize()
                            fresh_tools_data = await load_tools(fresh_session)
                            fresh_tool_object = types.Tool(function_declarations=fresh_tools_data)
                            
                            # Update session state with fresh connection
                            st.session_state.session = fresh_session
                            st.session_state.tools = fresh_tools_data
                            st.session_state.tool_object = fresh_tool_object
                            
                            llm = st.session_state.llm
                            
                            tool_called, updated_messages, response_text = await process_query(
                                fresh_session, llm, fresh_tool_object, original_query, st.session_state.chat_history
                            )
                            
                            st.session_state.chat_history = updated_messages
                            
                            if tool_called and response_text:
                                st.session_state.messages.append({
                                    "role": "assistant", 
                                    "content": f"✅ Success!\n\n{response_text}"
                                })
                            else:
                                st.session_state.messages.append({
                                    "role": "assistant", 
                                    "content": "❌ Tool created but couldn't be used. Try again."
                                })
                except Exception as e:
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": f"❌ Error during reprocessing: {str(e)}"
                    })
            
            asyncio.run(_reprocess())
        else:
            st.session_state.messages.append({
                "role": "assistant", 
                "content": "❌ Couldn't reconnect. Please refresh tools manually."
            })

# Initialize LLM
if st.session_state.llm is None:
    st.session_state.llm = get_llm()

# Header
st.markdown('<div class="main-header">', unsafe_allow_html=True)
st.title("🤖 AgentDRY")
st.markdown("*AI Assistant with Dynamic Tool Creation*")
st.markdown('</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("🧰 Tools Manager")
    
    # Server status
    if st.session_state.server_connected:
        st.markdown('<p class="status-success">🟢 Server Connected</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p class="status-error">🔴 Server Disconnected</p>', unsafe_allow_html=True)
    
    # Refresh button
    if st.button("🔄 Refresh Tools"):
        with st.spinner("Refreshing..."):
            if refresh_tools():
                st.success(f"✅ Loaded {len(st.session_state.tools)} tools!")
            else:
                st.error("❌ Could not connect to server")
        st.rerun()
    
    st.markdown("---")
    
    # Tools list
    st.subheader(f"Available Tools ({len(st.session_state.tools)})")
    
    if st.session_state.tools:
        for tool in st.session_state.tools:
            with st.expander(f"🔧 {tool['name']}", expanded=False):
                st.markdown(f"**Description:** {tool['description']}")
                if tool.get('parameters') and tool['parameters'].get('properties'):
                    st.markdown("**Parameters:**")
                    for param_name, param_info in tool['parameters']['properties'].items():
                        param_type = param_info.get('type', 'unknown')
                        param_desc = param_info.get('description', 'No description')
                        st.markdown(f"- `{param_name}` ({param_type}): {param_desc}")
    else:
        if st.session_state.server_connected:
            st.info("No tools yet. Ask me to create one!")
        else:
            st.error("Connect to server first")
    
    st.markdown("---")
    
    # Past chats
    st.subheader("📚 Chat History")
    
    if st.session_state.past_chats:
        if st.button("🗑️ Clear History"):
            st.session_state.past_chats = []
            st.rerun()
        
        for i, chat in enumerate(reversed(st.session_state.past_chats)):
            with st.expander(f"💬 Chat {len(st.session_state.past_chats) - i}"):
                for role, message in chat:
                    icon = "👤" if role == "user" else "🤖"
                    display_msg = message[:100] + "..." if len(message) > 100 else message
                    st.markdown(f"{icon} **{role.title()}:** {display_msg}")
    else:
        st.info("No chat history yet")

# Main chat area
col1, col2 = st.columns([4, 1])

with col1:
    # Connect on first load
    if not st.session_state.server_connected:
        with st.spinner("Connecting to server..."):
            refresh_tools()
    
    # Display messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if user_input := st.chat_input("Ask me anything or request a new tool..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Handle quit
        if user_input.lower().strip() == "quit":
            if st.session_state.messages:
                st.session_state.past_chats.append(st.session_state.messages.copy())
            
            st.session_state.messages = [{"role": "assistant", "content": "👋 Goodbye! Chat saved to history."}]
            st.session_state.chat_history = []
            st.rerun()
        
        # Process query
        with st.spinner("Processing..."):
            process_user_query(user_input)
        
        st.rerun()

with col2:
    # Status panel
    st.markdown("### 📊 Status")
    
    if st.session_state.server_connected:
        st.markdown('<p class="status-success">✅ Ready</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p class="status-error">❌ Offline</p>', unsafe_allow_html=True)
    
    st.metric("Tools Available", len(st.session_state.tools))
    st.metric("Messages", len(st.session_state.messages))
    st.metric("Past Chats", len(st.session_state.past_chats))