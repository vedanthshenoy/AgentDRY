import streamlit as st
import asyncio
import logging
import os
import json
from datetime import datetime
from typing import Dict, Any
import sys

# Add the current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from dotenv import load_dotenv

# Import the MCP client (assuming it's saved as mcp_client.py)
from mcp_client import MCPClient, QueryResult

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - MCP_STREAMLIT - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/mcp_streamlit.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="MCP Chat Interface",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for modern design
st.markdown("""
<style>
    .main > div {
        padding-top: 1rem;
    }
    
    .stApp {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
    }
    
    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 15px;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    .user-message {
        background: rgba(255, 255, 255, 0.15);
        border-left: 4px solid #4CAF50;
        margin-left: 3rem;
    }
    
    .assistant-message {
        background: rgba(255, 255, 255, 0.2);
        border-left: 4px solid #2196F3;
        margin-right: 3rem;
    }
    
    .tool-call {
        background: rgba(255, 193, 7, 0.2);
        border: 1px solid rgba(255, 193, 7, 0.5);
        border-radius: 10px;
        padding: 0.8rem;
        margin: 0.5rem 0;
    }
    
    .server-status {
        background: rgba(255, 255, 255, 0.1);
        padding: 0.8rem;
        border-radius: 10px;
        margin: 0.3rem 0;
        border-left: 4px solid;
    }
    
    .status-connected {
        border-left-color: #4CAF50;
    }
    
    .status-error {
        border-left-color: #f44336;
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.15);
        padding: 1.2rem;
        border-radius: 15px;
        text-align: center;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .tool-card {
        background: rgba(255, 255, 255, 0.1);
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    h1, h2, h3, h4 {
        color: white !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .stTextInput > div > div > input {
        background-color: rgba(255, 255, 255, 0.15);
        color: white;
        border: 2px solid rgba(255, 255, 255, 0.3);
        border-radius: 15px;
        padding: 12px;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #4CAF50;
        box-shadow: 0 0 10px rgba(76, 175, 80, 0.3);
    }
    
    .stButton > button {
        background: linear-gradient(45deg, #4CAF50, #45a049);
        color: white;
        border: none;
        border-radius: 15px;
        font-weight: bold;
        padding: 0.7rem 2rem;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.2);
    }
    
    .sampling-info {
        background: rgba(156, 39, 176, 0.2);
        border: 1px solid rgba(156, 39, 176, 0.5);
        border-radius: 10px;
        padding: 0.8rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_mcp_client():
    """Initialize and cache the MCP client."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("GEMINI_API_KEY not found in environment variables")
        st.stop()
    
    return MCPClient(api_key)

def initialize_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "client_initialized" not in st.session_state:
        st.session_state.client_initialized = False
    if "tools" not in st.session_state:
        st.session_state.tools = []
    if "server_status" not in st.session_state:
        st.session_state.server_status = {}
    if "sampling_enabled" not in st.session_state:
        st.session_state.sampling_enabled = True

async def initialize_client_async(client):
    """Initialize MCP client asynchronously."""
    try:
        success = await client.initialize()
        st.session_state.client_initialized = success
        if success:
            st.session_state.tools = await client.get_available_tools()
            st.session_state.server_status = await client.get_server_status()
        return success
    except Exception as e:
        logger.error(f"Client initialization failed: {e}")
        st.session_state.client_initialized = False
        return False

def display_header():
    """Display the main header."""
    st.markdown("""
    <div style='text-align: center; padding: 2rem 0;'>
        <h1 style='font-size: 3rem; margin-bottom: 0.5rem;'>🔗 MCP Chat Interface</h1>
        <p style='color: rgba(255,255,255,0.8); font-size: 1.2rem;'>
            Multi-server Model Context Protocol Chat with Sampling
        </p>
    </div>
    """, unsafe_allow_html=True)

def display_metrics(client):
    """Display key metrics."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        connected_servers = sum(1 for status in st.session_state.server_status.values() 
                              if status == "Connected")
        st.markdown(f"""
        <div class="metric-card">
            <h4>Connected Servers</h4>
            <h2 style="color: #4CAF50;">{connected_servers}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Available Tools</h4>
            <h2 style="color: #2196F3;">{len(st.session_state.tools)}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Chat Messages</h4>
            <h2 style="color: #FF9800;">{len(st.session_state.messages)}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        sampling_status = "Enabled" if st.session_state.sampling_enabled else "Disabled"
        color = "#9C27B0" if st.session_state.sampling_enabled else "#757575"
        st.markdown(f"""
        <div class="metric-card">
            <h4>Sampling</h4>
            <h2 style="color: {color};">{sampling_status}</h2>
        </div>
        """, unsafe_allow_html=True)

def display_chat_interface(client):
    """Display the main chat interface."""
    st.markdown("## 💬 Chat")
    
    # Chat container with fixed height for scrolling
    chat_container = st.container()
    
    # Display chat history
    with chat_container:
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>👤 You:</strong><br>
                    {message["content"]}
                </div>
                """, unsafe_allow_html=True)
            
            elif message["role"] == "assistant":
                st.markdown(f"""
                <div class="chat-message assistant-message">
                    <strong>🤖 Assistant:</strong><br>
                    {message["content"]}
                </div>
                """, unsafe_allow_html=True)
                
                # Show tool usage if available
                if message.get("tool_used"):
                    st.markdown(f"""
                    <div class="tool-call">
                        <strong>🔧 Tool Used:</strong> {message.get("tool_name", "Unknown")}<br>
                        <strong>Result:</strong> {str(message.get("tool_result", "No result"))[:200]}...
                    </div>
                    """, unsafe_allow_html=True)
                
                # Show sampling info if used
                if message.get("sampling_used"):
                    st.markdown(f"""
                    <div class="sampling-info">
                        <strong>🎯 Sampling:</strong> AI reasoning and tool selection enabled
                    </div>
                    """, unsafe_allow_html=True)
    
    # Chat input form
    st.markdown("---")
    with st.form(key="chat_form", clear_on_submit=True):
        col1, col2 = st.columns([5, 1])
        
        with col1:
            user_input = st.text_input(
                "Type your message...",
                placeholder="Ask me anything! I can use math tools, create dynamic tools, or discover servers.",
                label_visibility="collapsed"
            )
        
        with col2:
            submitted = st.form_submit_button("Send 🚀", use_container_width=True)
    
    # Process user input
    if submitted and user_input.strip():
        # Add user message
        st.session_state.messages.append({
            "role": "user", 
            "content": user_input.strip()
        })
        
        # Process with loading indicator
        with st.spinner("🤔 Processing your request..."):
            try:
                result = asyncio.run(client.process_query(user_input.strip()))
                
                # Add assistant response with metadata
                assistant_message = {
                    "role": "assistant",
                    "content": result.response,
                    "tool_used": result.tool_used,
                    "tool_name": result.tool_name,
                    "tool_result": result.tool_result,
                    "sampling_used": result.sampling_used
                }
                
                st.session_state.messages.append(assistant_message)
                
                # Refresh tools and server status
                st.session_state.tools = asyncio.run(client.get_available_tools())
                st.session_state.server_status = asyncio.run(client.get_server_status())
                
            except Exception as e:
                error_msg = f"❌ Error processing your request: {str(e)}"
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })
                logger.error(f"Query processing error: {e}")
        
        st.rerun()

def display_sidebar(client):
    """Display sidebar with controls and information."""
    with st.sidebar:
        st.markdown("## 🛠 Control Panel")
        
        # Sampling toggle
        st.session_state.sampling_enabled = st.toggle(
            "🎯 Enable Sampling", 
            value=st.session_state.sampling_enabled,
            help="Use AI reasoning for tool selection and response generation"
        )
        
        st.markdown("---")
        
        # Server status section
        with st.expander("🌐 Server Status", expanded=True):
            if st.session_state.server_status:
                for server, status in st.session_state.server_status.items():
                    css_class = "status-connected" if status == "Connected" else "status-error"
                    status_icon = "✅" if status == "Connected" else "❌"
                    st.markdown(f"""
                    <div class="server-status {css_class}">
                        <strong>{status_icon} {server.upper()}</strong><br>
                        <small>{status}</small>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No server status available")
        
        # Tools section
        with st.expander("🔧 Available Tools", expanded=False):
            if st.session_state.tools:
                for tool in st.session_state.tools:
                    server_color = {
                        "agentdry": "#4CAF50",
                        "math": "#2196F3", 
                        "seeker": "#FF9800"
                    }.get(tool["server"], "#757575")
                    
                    st.markdown(f"""
                    <div class="tool-card">
                        <strong style="color: {server_color};">{tool["name"]}</strong><br>
                        <small>{tool["description"][:80]}...</small><br>
                        <em>Server: {tool["server"]}</em>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No tools available")
        
        # Conversation controls
        st.markdown("---")
        st.markdown("## 🎛 Controls")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🗑 Clear Chat", use_container_width=True):
                st.session_state.messages = []
                client.clear_conversation()
                st.rerun()
        
        with col2:
            if st.button("🔄 Refresh", use_container_width=True):
                if st.session_state.client_initialized:
                    with st.spinner("Refreshing..."):
                        st.session_state.tools = asyncio.run(client.get_available_tools())
                        st.session_state.server_status = asyncio.run(client.get_server_status())
                    st.rerun()
        
        # Sample queries section
        st.markdown("---")
        st.markdown("## 💡 Sample Queries")
        
        sample_queries = [
            "Calculate 25 * 47 + 123",
            "Create a tool to generate random passwords",
            "Show me all connected servers",
            "What tools are available?",
            "Execute some Python code",
            "Refresh server discovery"
        ]
        
        for query in sample_queries:
            if st.button(f"💬 {query}", key=f"sample_{hash(query)}", use_container_width=True):
                # Add query to chat
                st.session_state.messages.append({
                    "role": "user",
                    "content": query
                })
                
                # Process query
                with st.spinner("Processing..."):
                    try:
                        result = asyncio.run(client.process_query(query))
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": result.response,
                            "tool_used": result.tool_used,
                            "tool_name": result.tool_name,
                            "tool_result": result.tool_result,
                            "sampling_used": result.sampling_used
                        })
                        
                        # Refresh data
                        st.session_state.tools = asyncio.run(client.get_available_tools())
                        st.session_state.server_status = asyncio.run(client.get_server_status())
                        
                    except Exception as e:
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"Error: {str(e)}"
                        })
                
                st.rerun()
        
        # Debug info
        st.markdown("---")
        st.markdown("## 🐛 Debug Info")
        st.text(f"Client initialized: {st.session_state.client_initialized}")
        st.text(f"Total messages: {len(st.session_state.messages)}")
        st.text(f"Timestamp: {datetime.now().strftime('%H:%M:%S')}")

def main():
    """Main Streamlit application."""
    initialize_session_state()
    
    # Get MCP client
    client = get_mcp_client()
    
    # Initialize client if needed
    if not st.session_state.client_initialized:
        with st.spinner("🚀 Initializing MCP connections..."):
            success = asyncio.run(initialize_client_async(client))
            if not success:
                st.error("Failed to initialize MCP client. Please check your server configurations.")
                st.stop()
            else:
                st.success("MCP client initialized successfully!")
                st.balloons()
    
    # Display header
    display_header()
    
    # Main layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Display metrics
        display_metrics(client)
        st.markdown("---")
        
        # Main chat interface
        display_chat_interface(client)
    
    with col2:
        # Sidebar controls
        display_sidebar(client)

if __name__ == "__main__":
    # Ensure logs directory exists
    os.makedirs("logs", exist_ok=True)
    
    # Run the app
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        logger.error(f"Streamlit app error: {e}")