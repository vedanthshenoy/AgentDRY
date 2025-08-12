import streamlit as st
import asyncio
import logging
import os
from datetime import datetime
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = current_dir  # C:\prass\agentdry
clients_dir = os.path.join(project_root, 'clients', 'gemini_clients')

# Add paths to system path - KEEP EXACTLY SAME
import sys
sys.path.insert(0, project_root)
sys.path.insert(0, clients_dir)
from gemini_mcp_client_improved import AgentDRY
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure streamlit logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - STREAMLIT - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("logs/streamlit.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="MCP Agent Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better visibility and modern design
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .stMarkdown {
        color: white !important;
    }
    
    .stTextInput > div > div > input {
        background-color: rgba(255, 255, 255, 0.1);
        color: white;
        border: 1px solid rgba(255, 255, 255, 0.3);
        border-radius: 10px;
    }
    
    .stTextInput > div > div > input::placeholder {
        color: rgba(255, 255, 255, 0.7);
    }
    
    .stButton > button {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
    
    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 10px;
        backdrop-filter: blur(10px);
    }
    
    .user-message {
        background: rgba(255, 255, 255, 0.1);
        border-left: 4px solid #667eea;
        margin-left: 2rem;
    }
    
    .assistant-message {
        background: rgba(255, 255, 255, 0.15);
        border-left: 4px solid #764ba2;
        margin-right: 2rem;
    }
    
    .tool-card {
        background: rgba(255, 255, 255, 0.1);
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(10px);
    }
    
    .stExpander {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.1);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        backdrop-filter: blur(10px);
    }
    
    h1, h2, h3 {
        color: white !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .stSelectbox > div > div {
        background-color: rgba(255, 255, 255, 0.1);
        color: white;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_agent():
    """Initialize and cache the agent."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("GEMINI_API_KEY not found in environment variables")
        st.stop()
    
    agent = AgentDRY(api_key)
    return agent

def initialize_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "agent_initialized" not in st.session_state:
        st.session_state.agent_initialized = False
    if "tools" not in st.session_state:
        st.session_state.tools = []
    if "server_status" not in st.session_state:
        st.session_state.server_status = "Unknown"

async def initialize_agent_async(agent):
    """Initialize agent asynchronously."""
    try:
        success = await agent.initialize()
        st.session_state.agent_initialized = success
        st.session_state.server_status = "Connected" if agent.server_available else "Disconnected"
        if agent.server_available:
            st.session_state.tools = await agent.tool_manager.get_tools()
        return success
    except Exception as e:
        logger.error(f"Agent initialization failed: {e}")
        st.session_state.server_status = "Error"
        return False

def display_header():
    """Display the main header."""
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<h1 style='text-align: center; margin-bottom: 2rem;'>🤖 MCP Agent Dashboard</h1>", unsafe_allow_html=True)

def display_status_bar(agent):
    """Display status information."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Server Status</h4>
            <p style="color: {'#4CAF50' if st.session_state.server_status == 'Connected' else '#f44336'};">
                {st.session_state.server_status}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Available Tools</h4>
            <p>{len(st.session_state.tools)}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Messages</h4>
            <p>{len(st.session_state.messages)}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        if st.button("🔄 Refresh", key="refresh_status"):
            st.rerun()

def display_chat_interface(agent):
    """Display the main chat interface."""
    st.markdown("<h2>💬 Chat</h2>", unsafe_allow_html=True)
    
    # Chat container
    chat_container = st.container()
    
    # Display chat history
    with chat_container:
        for i, message in enumerate(st.session_state.messages):
            if message["role"] == "user":
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>You:</strong> {message["content"]}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message assistant-message">
                    <strong>Assistant:</strong> {message["content"]}
                </div>
                """, unsafe_allow_html=True)
    
    # Chat input
    with st.form(key="chat_form", clear_on_submit=True):
        col1, col2 = st.columns([4, 1])
        
        with col1:
            user_input = st.text_input(
                "Ask me anything...",
                placeholder="Type your message here...",
                label_visibility="collapsed"
            )
        
        with col2:
            submitted = st.form_submit_button("Send 📤", use_container_width=True)
    
    # Process user input
    if submitted and user_input:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Show thinking spinner
        with st.spinner("🤔 Thinking..."):
            try:
                # Process query using asyncio.run
                result = asyncio.run(agent.process_query(user_input))
                
                # Add assistant response
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": result.response
                })
                
                # Refresh tools if a tool was used
                if result.tool_used and agent.server_available:
                    st.session_state.tools = asyncio.run(agent.tool_manager.get_tools())
                
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": error_msg
                })
                logger.error(f"Query processing error: {e}")
        
        st.rerun()

def display_sidebar(agent):
    """Display the sidebar with tools and controls."""
    with st.sidebar:
        st.markdown("<h2>🛠️ Tools & Controls</h2>", unsafe_allow_html=True)
        
        # Tools section
        with st.expander("📋 Available Tools", expanded=False):
            if st.session_state.tools:
                for tool in st.session_state.tools:
                    st.markdown(f"""
                    <div class="tool-card">
                        <h4>{tool.name}</h4>
                        <p style="color: rgba(255,255,255,0.8); font-size: 0.9em;">{tool.description}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Delete tool button
                    if st.button(f"🗑️ Delete", key=f"delete_{tool.name}"):
                        with st.spinner(f"Deleting {tool.name}..."):
                            success = asyncio.run(agent.delete_tool(tool.name))
                            if success:
                                st.success(f"Tool '{tool.name}' deleted successfully!")
                                st.session_state.tools = asyncio.run(agent.tool_manager.get_tools())
                                st.rerun()
                            else:
                                st.error(f"Failed to delete tool '{tool.name}'")
            else:
                st.info("No tools available")
        
        # Chat history section
        with st.expander("📚 Chat History", expanded=False):
            if st.session_state.messages:
                for i, msg in enumerate(reversed(st.session_state.messages[-10:])):  # Last 10 messages
                    role_icon = "👤" if msg["role"] == "user" else "🤖"
                    st.markdown(f"""
                    <div style="padding: 0.5rem; margin: 0.2rem 0; background: rgba(255,255,255,0.1); border-radius: 5px;">
                        <strong>{role_icon} {msg["role"].title()}:</strong><br>
                        <span style="font-size: 0.9em;">{msg["content"][:100]}{'...' if len(msg["content"]) > 100 else ''}</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No chat history yet")
        
        # Controls section
        st.markdown("---")
        st.markdown("<h3>🎛️ Controls</h3>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.messages = []
                agent.clear_conversation()
                st.rerun()
        
        with col2:
            if st.button("🔄 Refresh Tools", use_container_width=True):
                if agent.server_available:
                    with st.spinner("Refreshing tools..."):
                        st.session_state.tools = asyncio.run(agent.tool_manager.get_tools())
                    st.rerun()
                else:
                    st.warning("Server not available")
        
        # Server info
        st.markdown("---")
        st.markdown("<h3>📊 Server Info</h3>", unsafe_allow_html=True)
        st.markdown(f"**Status:** {st.session_state.server_status}")
        st.markdown(f"**Tools:** {len(st.session_state.tools)}")
        st.markdown(f"**Messages:** {len(st.session_state.messages)}")

def main_app():
    """Main Streamlit application - now synchronous."""
    initialize_session_state()
    
    # Get agent instance
    agent = get_agent()
    
    # Initialize agent if not already done
    if not st.session_state.agent_initialized:
        with st.spinner("🚀 Initializing agent..."):
            asyncio.run(initialize_agent_async(agent))
    
    # Display header
    display_header()
    
    # Create main layout
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Status bar
        display_status_bar(agent)
        st.markdown("---")
        
        # Main chat interface
        display_chat_interface(agent)
    
    with col2:
        # Sidebar content in main area for better visibility
        display_sidebar(agent)

def run_async_streamlit():
    """Run the Streamlit app."""
    try:
        main_app()  # Now synchronous
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        logger.error(f"Streamlit app error: {e}")

if __name__ == "__main__":
    # Ensure logs directory exists
    os.makedirs("logs", exist_ok=True)
    
    # Run the app
    run_async_streamlit()