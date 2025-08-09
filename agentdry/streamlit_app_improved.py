import streamlit as st
import asyncio
import os
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
import nest_asyncio

# Setup correct paths for your project structure
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = current_dir  # C:\prass\agentdry
clients_dir = os.path.join(project_root, 'clients', 'gemini_clients')

# Add paths to system path
import sys
sys.path.insert(0, project_root)
sys.path.insert(0, clients_dir)

# Apply nest_asyncio for Streamlit compatibility
nest_asyncio.apply()

# Setup logging
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "streamlit.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configure Streamlit
st.set_page_config(
    page_title="AgentDRY - AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .status-card {
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .status-success {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .status-error {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
    .status-warning {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
    }
    .status-info {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Import handling with proper fallbacks
try:
    # First try to import the improved client
    improved_client_path = os.path.join(clients_dir, "gemini_mcp_client_improved.py")
    
    if os.path.exists(improved_client_path):
        import importlib.util
        spec = importlib.util.spec_from_file_location("gemini_mcp_client_improved", improved_client_path)
        client_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(client_module)
        
        AgentDRY = client_module.AgentDRY
        QueryResult = client_module.QueryResult
        MCPTool = client_module.MCPTool
        logger.info("Successfully imported improved client components")
        CLIENT_TYPE = "improved"
    else:
        # Fallback to original client
        from gemini_mcp_client import GeminiClient, process_query, get_tools
        from mcp.client.sse import sse_client
        from mcp import ClientSession
        
        st.warning("⚠️ Using original client. Consider upgrading to improved version.")
        CLIENT_TYPE = "original"
        
        # Define compatibility classes
        from dataclasses import dataclass
        
        @dataclass
        class QueryResult:
            success: bool
            response: str
            tool_used: bool = False
            error: Optional[str] = None
        
        @dataclass 
        class MCPTool:
            name: str
            description: str
            parameters: Dict[str, Any]
        
        class AgentDRY:
            """Compatibility wrapper for original client"""
            def __init__(self, api_key):
                self.api_key = api_key
                self.initialized = False
                self.llm = GeminiClient(api_key)
                self.tools = []
                self.server_url = "http://localhost:8000/sse"
                self.server_available = False
            
            async def initialize(self):
                try:
                    async with sse_client(url=self.server_url) as streams:
                        async with ClientSession(*streams) as session:
                            await asyncio.wait_for(session.initialize(), timeout=10.0)
                            tools_data = await get_tools(session)
                            self.tools = [
                                MCPTool(
                                    name=tool["name"], 
                                    description=tool["description"], 
                                    parameters=tool.get("parameters", {})
                                ) 
                                for tool in tools_data
                            ]
                            self.initialized = True
                            self.server_available = True
                            return True
                except Exception as e:
                    logger.warning(f"Server connection failed, running in direct mode: {e}")
                    self.initialized = True
                    self.server_available = False
                    return True
            
            async def process_query(self, query: str) -> QueryResult:
                try:
                    # Simple math check for original client
                    import re
                    query_lower = query.lower().strip()
                    if re.search(r'what\s+is\s+\d+\s*[\+\-\*\/]\s*\d+', query_lower):
                        # Try to evaluate simple math
                        try:
                            # Extract the math expression
                            math_match = re.search(r'(\d+\s*[\+\-\*\/]\s*\d+)', query_lower)
                            if math_match:
                                expr = math_match.group(1).replace(' ', '')
                                # Simple evaluation (safe for basic operations)
                                if '+' in expr:
                                    parts = expr.split('+')
                                    result = sum(int(p) for p in parts)
                                elif '-' in expr:
                                    parts = expr.split('-')
                                    result = int(parts[0]) - sum(int(p) for p in parts[1:])
                                elif '*' in expr:
                                    parts = expr.split('*')
                                    result = 1
                                    for p in parts:
                                        result *= int(p)
                                elif '/' in expr:
                                    parts = expr.split('/')
                                    result = int(parts[0])
                                    for p in parts[1:]:
                                        result /= int(p)
                                
                                return QueryResult(
                                    success=True,
                                    response=f"The answer is {result}",
                                    tool_used=False
                                )
                        except:
                            pass
                    
                    # If server is not available, use direct LLM
                    if not self.server_available:
                        # Generate direct response using Gemini
                        from google import generativeai as genai
                        model = genai.GenerativeModel("gemini-1.5-flash")
                        response = model.generate_content(query)
                        return QueryResult(
                            success=True,
                            response=response.text,
                            tool_used=False
                        )
                    
                    # Original tool-based processing
                    async with sse_client(url=self.server_url) as streams:
                        async with ClientSession(*streams) as session:
                            await session.initialize()
                            tools_data = await get_tools(session)
                            tool_called, history, response = await process_query(
                                session, self.llm, tools_data, query, []
                            )
                            return QueryResult(
                                success=True, 
                                response=response, 
                                tool_used=tool_called
                            )
                except Exception as e:
                    error_msg = f"Query processing failed: {str(e)}"
                    logger.error(error_msg)
                    
                    # Fallback to direct response even on error
                    try:
                        from google import generativeai as genai
                        model = genai.GenerativeModel("gemini-1.5-flash")
                        response = model.generate_content(f"Please help with: {query}")
                        return QueryResult(
                            success=True, 
                            response=response.text, 
                            tool_used=False
                        )
                    except Exception as fallback_error:
                        return QueryResult(
                            success=False, 
                            response=f"I'm having trouble processing that request: {error_msg}", 
                            error=str(e)
                        )
            
            async def refresh_tools(self):
                try:
                    if not self.server_available:
                        return []
                        
                    async with sse_client(url=self.server_url) as streams:
                        async with ClientSession(*streams) as session:
                            await session.initialize()
                            tools_data = await get_tools(session)
                            self.tools = [
                                MCPTool(
                                    name=tool["name"], 
                                    description=tool["description"], 
                                    parameters=tool.get("parameters", {})
                                ) 
                                for tool in tools_data
                            ]
                            return self.tools
                except Exception as e:
                    logger.error(f"Tool refresh failed: {e}")
                    self.server_available = False
                    return []

    # Import tool creation function (optional)
    try:
        from main import create_and_update_tool
    except ImportError:
        logger.info("Tool creation module not available")
        def create_and_update_tool(query):
            st.warning("⚠️ Tool creation module not available")
            
except Exception as e:
    st.error(f"❌ Critical import error: {e}")
    st.info("Please ensure all required packages are installed: `pip install google-generativeai mcp httpx streamlit nest-asyncio`")
    st.stop()

# Session state management
def init_session_state():
    """Initialize session state variables."""
    defaults = {
        "agent": None,
        "initialized": False,
        "messages": [],
        "past_chats": [],
        "tools": [],
        "server_connected": False,
        "server_available": True,  # Assume available until proven otherwise
        "last_error": None,
        "processing": False,
        "direct_mode": False
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

def add_message(role: str, content: str):
    """Add a message to the current conversation."""
    st.session_state.messages.append({
        "role": role,
        "content": content,
        "timestamp": datetime.now().strftime("%H:%M:%S")
    })

def render_status_card(status: str, message: str, card_type: str = "info"):
    """Render a status card."""
    css_class = f"status-{card_type}"
    st.markdown(f"""
    <div class="status-card {css_class}">
        <strong>{status}</strong><br>
        {message}
    </div>
    """, unsafe_allow_html=True)

def render_metrics():
    """Render metrics cards."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{len(st.session_state.tools)}</h3>
            <p>Tools Available</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{len(st.session_state.messages)}</h3>
            <p>Messages</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{len(st.session_state.past_chats)}</h3>
            <p>Past Chats</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        if st.session_state.direct_mode:
            status = "🟡 Direct Mode"
            color = "#ffc107"
        elif st.session_state.server_connected:
            status = "🟢 Connected"
            color = "#28a745"
        else:
            status = "🔴 Disconnected"
            color = "#dc3545"
            
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: {color};">{status.split()[1]}</h3>
            <p>Status</p>
        </div>
        """, unsafe_allow_html=True)

async def initialize_agent():
    """Initialize the agent."""
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            st.error("❌ GEMINI_API_KEY environment variable not found!")
            st.info("Please create a .env file with your GEMINI_API_KEY")
            return False
        
        if st.session_state.agent is None:
            st.session_state.agent = AgentDRY(api_key)
        
        success = await st.session_state.agent.initialize()
        
        if success:
            st.session_state.initialized = True
            st.session_state.last_error = None
            
            # Check server availability
            if CLIENT_TYPE == "improved":
                st.session_state.server_connected = st.session_state.agent.server_available
                st.session_state.direct_mode = not st.session_state.agent.server_available
                st.session_state.tools = await st.session_state.agent.tool_manager.get_tools()
            else:
                st.session_state.server_connected = st.session_state.agent.server_available
                st.session_state.direct_mode = not st.session_state.agent.server_available
                st.session_state.tools = st.session_state.agent.tools
            
            return True
        else:
            st.session_state.server_connected = False
            st.session_state.direct_mode = True
            st.session_state.last_error = "Initialization failed"
            return False
            
    except Exception as e:
        error_msg = f"Initialization error: {str(e)}"
        logger.error(error_msg)
        st.session_state.last_error = error_msg
        st.session_state.server_connected = False
        st.session_state.direct_mode = True
        return False

async def refresh_connection():
    """Refresh the connection and tools."""
    try:
        if st.session_state.agent is None:
            return await initialize_agent()
        
        if CLIENT_TYPE == "improved":
            tools = await st.session_state.agent.tool_manager.refresh_tools()
            st.session_state.server_connected = st.session_state.agent.server_available
            st.session_state.direct_mode = not st.session_state.agent.server_available
        else:
            tools = await st.session_state.agent.refresh_tools()
            st.session_state.server_connected = st.session_state.agent.server_available
            st.session_state.direct_mode = not st.session_state.agent.server_available
        
        st.session_state.tools = tools
        st.session_state.last_error = None
        return True
        
    except Exception as e:
        error_msg = f"Refresh failed: {str(e)}"
        logger.error(error_msg)
        st.session_state.last_error = error_msg
        st.session_state.server_connected = False
        st.session_state.direct_mode = True
        return False

async def process_user_input(user_input: str):
    """Process user input."""
    if st.session_state.agent is None:
        add_message("assistant", "❌ Agent not initialized. Please refresh the connection.")
        return
    
    try:
        st.session_state.processing = True
        add_message("user", user_input)
        
        result = await st.session_state.agent.process_query(user_input)
        
        if result.success:
            add_message("assistant", result.response)
            if result.tool_used:
                st.success("✅ Tool executed successfully!")
            elif st.session_state.direct_mode:
                st.info("ℹ️ Answered using direct AI response (no tools needed)")
        else:
            add_message("assistant", f"❌ {result.response}")
            if result.error:
                st.session_state.last_error = result.error
                
    except Exception as e:
        error_msg = f"Error processing input: {str(e)}"
        logger.error(error_msg)
        add_message("assistant", f"❌ {error_msg}")
    finally:
        st.session_state.processing = False

def main():
    """Main Streamlit application."""
    # Initialize session state
    init_session_state()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 AgentDRY</h1>
        <p>Dynamic AI Assistant with Tool Creation</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize agent if not done
    if not st.session_state.initialized:
        st.info("🚀 Click the button below to initialize the agent. It will work with or without an MCP server.")
        
        if st.button("Initialize Agent", type="primary", use_container_width=True):
            with st.spinner("Initializing agent..."):
                success = asyncio.run(initialize_agent())
            if success:
                if st.session_state.direct_mode:
                    st.warning("⚠️ Initialized in Direct Mode (no MCP server). Basic queries will work fine!")
                else:
                    st.success("✅ Agent initialized successfully with MCP server!")
                st.rerun()
            else:
                st.error("❌ Failed to initialize agent.")
        return
    
    # Main layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Status display
        if st.session_state.direct_mode:
            render_status_card("🟡 Direct Mode", f"Working without MCP server | Client: {CLIENT_TYPE}", "warning")
        elif st.session_state.server_connected:
            render_status_card("🟢 Connected", f"Agent ready with tools | Client: {CLIENT_TYPE}", "success")
        else:
            render_status_card("🔴 Disconnected", "Connection to server lost", "error")
        
        if st.session_state.last_error:
            render_status_card("⚠️ Warning", st.session_state.last_error, "warning")
        
        # Chat messages
        st.subheader("💬 Conversation")
        
        # Display messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                st.caption(f"Time: {message['timestamp']}")
        
        # Chat input
        if user_input := st.chat_input(
            "Ask anything! Try: 'What is 2+2?' or 'Hello'", 
            disabled=st.session_state.processing
        ):
            if user_input.lower().strip() == "clear":
                st.session_state.messages = []
                st.rerun()
            elif user_input.lower().strip() == "save":
                if st.session_state.messages:
                    st.session_state.past_chats.append({
                        "messages": st.session_state.messages.copy(),
                        "timestamp": datetime.now().isoformat()
                    })
                    st.session_state.messages = []
                    st.success("💾 Chat saved!")
                    st.rerun()
            else:
                with st.spinner("Processing your request..."):
                    asyncio.run(process_user_input(user_input))
                st.rerun()
        
        # Action buttons
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            if st.button("🔄 Refresh", disabled=st.session_state.processing):
                with st.spinner("Refreshing connection..."):
                    success = asyncio.run(refresh_connection())
                if success:
                    tool_count = len(st.session_state.tools)
                    if st.session_state.direct_mode:
                        st.info("ℹ️ Refreshed! Working in direct mode.")
                    else:
                        st.success(f"✅ Refreshed! {tool_count} tools loaded")
                else:
                    st.error("❌ Refresh failed, using direct mode")
                st.rerun()
        
        with col_b:
            if st.button("💾 Save Chat", disabled=not st.session_state.messages):
                if st.session_state.messages:
                    st.session_state.past_chats.append({
                        "messages": st.session_state.messages.copy(),
                        "timestamp": datetime.now().isoformat()
                    })
                    st.session_state.messages = []
                    st.success("💾 Chat saved!")
                    st.rerun()
        
        with col_c:
            if st.button("🗑️ Clear", disabled=not st.session_state.messages):
                st.session_state.messages = []
                st.rerun()
    
    with col2:
        # Sidebar content
        st.subheader("📊 Dashboard")
        render_metrics()
        
        # Mode explanation
        if st.session_state.direct_mode:
            st.info("""
            **🟡 Direct Mode**
            
            Running without MCP server. Can handle:
            - Basic conversations
            - Simple math (2+2, etc.)
            - General questions
            - Creative writing
            
            For advanced tools, start MCP server.
            """)
        
        # Tools
        st.subheader("🧰 Available Tools")
        if st.session_state.tools:
            for tool in st.session_state.tools:
                with st.expander(f"🔧 {tool.name}"):
                    st.write(f"**Description:** {tool.description}")
                    
                    # Handle parameters display
                    params = tool.parameters
                    if isinstance(params, dict) and "properties" in params:
                        st.write("**Parameters:**")
                        for param_name, param_info in params["properties"].items():
                            if isinstance(param_info, dict):
                                param_type = param_info.get("type", "unknown")
                                param_desc = param_info.get("description", "No description")
                                st.write(f"- `{param_name}` ({param_type}): {param_desc}")
                    else:
                        st.write("**Parameters:** None or unavailable")
        elif st.session_state.direct_mode:
            st.info("No tools available in Direct Mode. Start MCP server to access tools!")
        else:
            st.info("No tools available. Ask to create one!")
        
        # Chat history
        st.subheader("📚 Chat History")
        if st.session_state.past_chats:
            if st.button("🗑️ Clear History"):
                st.session_state.past_chats = []
                st.rerun()
            
            for i, chat in enumerate(reversed(st.session_state.past_chats[-5:])):
                with st.expander(f"💬 Chat {len(st.session_state.past_chats) - i}"):
                    timestamp = chat.get('timestamp', 'Unknown')
                    st.caption(f"Saved: {timestamp}")
                    
                    for msg in chat.get('messages', [])[:3]:
                        role_icon = "👤" if msg["role"] == "user" else "🤖"
                        content = msg["content"]
                        if len(content) > 100:
                            content = content[:100] + "..."
                        st.write(f"{role_icon} **{msg['role'].title()}:** {content}")
        else:
            st.info("No chat history yet.")
        
        # Quick examples
        st.subheader("💡 Try These Examples")
        examples = [
            "What is 2+2?",
            "Hello, how are you?",
            "Tell me a joke",
            "Explain quantum computing",
            "What's the capital of France?"
        ]
        
        for example in examples:
            if st.button(f"💬 {example}", key=f"example_{example[:10]}"):
                if not st.session_state.processing:
                    with st.spinner("Processing..."):
                        asyncio.run(process_user_input(example))
                    st.rerun()

if __name__ == "__main__":
    main()