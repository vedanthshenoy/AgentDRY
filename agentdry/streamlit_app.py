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
from gemini_mcp_client_improved import GeminiMCPClient
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
    page_title="Gemini MCP Client",
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
        background: linear-gradient(135deg, #232526 0%, #414345 100%); 
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
def get_client():
    """Initialize and cache the client."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("GEMINI_API_KEY not found in environment variables")
        st.stop()
    
    client = GeminiMCPClient(api_key)
    return client

def initialize_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "client_initialized" not in st.session_state:
        st.session_state.client_initialized = False
    if "server_status" not in st.session_state:
        st.session_state.server_status = {}

async def initialize_client_async(client):
    """Initialize client asynchronously."""
    try:
        success = await client.initialize()
        st.session_state.client_initialized = success
        if success:
            st.session_state.server_status = await client.get_server_status()
        return success
    except Exception as e:
        logger.error(f"Client initialization failed: {e}")
        st.session_state.server_status = {"server_available": False, "error": str(e)}
        return False

def display_header():
    """Display the main header."""
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<h1 style='text-align: center; margin-bottom: 2rem;'>🤖 Gemini MCP Client</h1>", unsafe_allow_html=True)

def display_status_bar(client):
    """Display status information."""
    col1, col2, col3, col4 = st.columns(4)
    
    status = st.session_state.server_status
    server_available = status.get('server_available', False)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Server Status</h4>
            <p style="color: {'#4CAF50' if server_available else '#f44336'};">
                {'Connected' if server_available else 'Disconnected'}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Available Tools</h4>
            <p>{status.get('available_tools', 0)}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Server Context</h4>
            <p>{status.get('server_context', 'unknown')}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        if st.button("🔄 Refresh", key="refresh_status"):
            if st.session_state.client_initialized:
                st.session_state.server_status = asyncio.run(client.get_server_status())
            st.rerun()

def display_chat_interface(client):
    """Display the main chat interface."""
    st.markdown("<h2>💬 Chat</h2>", unsafe_allow_html=True)
    
    # Chat container
    chat_container = st.container()
    
    # Display chat history
    with chat_container:
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>You:</strong> {message["content"]}
                </div>
                """, unsafe_allow_html=True)
            else:
                tools_info = ""
                if message.get("tools_used"):
                    tools_info = f'<br><small>🔧 Tools used: {", ".join(message["tools_used"])}'
                    if message.get("tool_args"):
                        tools_info += f' with args {message["tool_args"]}'
                    tools_info += "</small>"

                if message.get("tool_created"):
                    tools_info += '<br><small>🎉 New tool created!</small>'
                
                st.markdown(f"""
                <div class="chat-message assistant-message">
                    <strong>Assistant:</strong> {message["content"]}
                    {tools_info}
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
                # Process query - get the detailed response
                response_data = asyncio.run(client.process_query(user_input))
                
                if response_data:
                    # response_data is a dict with conversation details
                    assistant_response = response_data.get("assistant", "No response received")
                    tools_used = response_data.get("tools_used", [])
                    tool_created = response_data.get("tool_created", False)
                    
                    # Add assistant response
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": assistant_response,
                        "tools_used": tools_used,
                        "tool_created": tool_created
                    })
                    
                    # Refresh status if tools were used or created
                    if tools_used or tool_created:
                        st.session_state.server_status = asyncio.run(client.get_server_status())
                else:
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": "Sorry, I couldn't process your request."
                    })
                
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": error_msg
                })
                logger.error(f"Query processing error: {e}")
        
        st.rerun()

def display_sidebar(client):
    """Display the sidebar with tools and controls."""
    with st.sidebar:
        st.markdown("<h2>🛠️ Tools & Controls</h2>", unsafe_allow_html=True)
        
        # Server info section
        status = st.session_state.server_status
        server_available = status.get('server_available', False)
        
        with st.expander("📊 Server Status", expanded=True):
            st.markdown(f"**Status:** {'🟢 Connected' if server_available else '🔴 Disconnected'}")
            st.markdown(f"**Context:** {status.get('server_context', 'unknown')}")
            st.markdown(f"**Tools:** {status.get('available_tools', 0)}")
            
            if status.get('tool_names'):
                st.markdown("**Available Tools:**")
                for tool_name in status['tool_names']:
                    st.markdown(f"  • {tool_name}")
        
        # Chat history section
        with st.expander("📚 Chat History", expanded=False):
            if st.session_state.messages:
                # Show last 10 messages - fixed the slice issue
                recent_messages = st.session_state.messages[-10:] if len(st.session_state.messages) > 10 else st.session_state.messages
                for i, msg in enumerate(reversed(recent_messages)):
                    role_icon = "👤" if msg["role"] == "user" else "🤖"
                    content_preview = msg["content"][:100] + ('...' if len(msg["content"]) > 100 else '')
                    
                    st.markdown(f"""
                    <div style="padding: 0.5rem; margin: 0.2rem 0; background: rgba(255,255,255,0.1); border-radius: 5px;">
                        <strong>{role_icon} {msg["role"].title()}:</strong><br>
                        <span style="font-size: 0.9em;">{content_preview}</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No chat history yet")
        
        # Statistics section
        if st.session_state.messages:
            with st.expander("📈 Statistics", expanded=False):
                total_messages = len(st.session_state.messages)
                user_messages = len([m for m in st.session_state.messages if m["role"] == "user"])
                tool_uses = len([m for m in st.session_state.messages if m.get("tools_used")])
                tools_created = len([m for m in st.session_state.messages if m.get("tool_created")])
                
                st.markdown(f"**Total Messages:** {total_messages}")
                st.markdown(f"**User Messages:** {user_messages}")
                st.markdown(f"**Tool Uses:** {tool_uses}")
                st.markdown(f"**Tools Created:** {tools_created}")
        
        # Controls section
        st.markdown("---")
        st.markdown("<h3>🎛️ Controls</h3>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.messages = []
                client.clear_conversation()
                st.success("Chat cleared!")
                st.rerun()
        
        with col2:
            if st.button("🔄 Refresh Status", use_container_width=True):
                if st.session_state.client_initialized:
                    with st.spinner("Refreshing status..."):
                        try:
                            st.session_state.server_status = asyncio.run(client.get_server_status())
                            st.success("Status refreshed!")
                        except Exception as e:
                            st.error(f"Refresh failed: {e}")
                    st.rerun()
                else:
                    st.warning("Client not initialized")

def main_app():
    """Main Streamlit application."""
    initialize_session_state()
    
    # Get client instance
    client = get_client()
    
    # Initialize client if not already done
    if not st.session_state.client_initialized:
        with st.spinner("🚀 Initializing client..."):
            success = asyncio.run(initialize_client_async(client))
            if not success:
                st.error("Failed to initialize client. Please check your configuration.")
                st.error(f"Error details: {st.session_state.server_status.get('error', 'Unknown error')}")
                st.stop()
    
    # Display header
    display_header()
    
    # Create main layout
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Status bar
        display_status_bar(client)
        st.markdown("---")
        
        # Main chat interface
        display_chat_interface(client)
    
    with col2:
        # Sidebar content
        display_sidebar(client)

def run_streamlit_app():
    """Run the Streamlit app."""
    try:
        main_app()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        logger.error(f"Streamlit app error: {e}")
        # Show error details in development
        if os.getenv("STREAMLIT_DEBUG", "false").lower() == "true":
            st.exception(e)

if __name__ == "__main__":
    # Ensure logs directory exists
    os.makedirs("logs", exist_ok=True)
    
    # Run the app
    run_streamlit_app()