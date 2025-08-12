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

# Add paths to system path - KEEP EXACTLY SAME
import sys
sys.path.insert(0, project_root)
sys.path.insert(0, clients_dir)

# Apply nest_asyncio for Streamlit compatibility
nest_asyncio.apply()

# Setup logging with enhanced tool logging
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Setup main logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "streamlit.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Setup dedicated tool operations logger
tool_logger = logging.getLogger("streamlit_tool_ops")
tool_handler = logging.FileHandler(os.path.join(LOG_DIR, "streamlit_tool_operations.log"))
tool_handler.setFormatter(logging.Formatter(
    '%(asctime)s - STREAMLIT_TOOL - %(levelname)s - %(message)s'
))
tool_logger.addHandler(tool_handler)
tool_logger.setLevel(logging.INFO)

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
    .tool-card {
        background: #f8f9fa;
        padding: 0.8rem;
        border-left: 4px solid #007bff;
        margin: 0.5rem 0;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# Import handling with proper fallbacks
try:
    # Try to import the improved client
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
        st.error("❌ Improved client not found. Please ensure gemini_mcp_client_improved.py exists.")
        st.stop()
        
except Exception as e:
    st.error(f"❌ Critical import error: {e}")
    st.info("Please ensure all required packages are installed: `pip install google-generativeai mcp httpx streamlit nest-asyncio`")
    st.stop()

# Import tool creation function
try:
    from main import create_and_update_tool
except ImportError:
    logger.warning("Tool creation module not available")
    def create_and_update_tool(query):
        st.warning("⚠️ Tool creation module not available")

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
        "server_available": True,
        "last_error": None,
        "processing": False,
        "direct_mode": False,
        "conversation_context": {}  # For storing context like "same radius"
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
    """Initialize the agent with enhanced logging."""
    try:
        logger.info("🚀 STREAMLIT: Starting agent initialization...")
        tool_logger.info("🚀 STREAMLIT AGENT INITIALIZATION STARTED")
        
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.error("❌ GEMINI_API_KEY environment variable not found!")
            tool_logger.error("❌ GEMINI_API_KEY not found")
            st.error("❌ GEMINI_API_KEY environment variable not found!")
            st.info("Please create a .env file with your GEMINI_API_KEY")
            return False
        
        if st.session_state.agent is None:
            logger.info("Creating new AgentDRY instance...")
            tool_logger.info("🔧 Creating new AgentDRY instance...")
            st.session_state.agent = AgentDRY(api_key)
        
        logger.info("Initializing agent...")
        tool_logger.info("⚙️ Initializing agent...")
        success = await st.session_state.agent.initialize()
        
        if success:
            st.session_state.initialized = True
            st.session_state.last_error = None
            
            # Check server availability
            st.session_state.server_connected = st.session_state.agent.server_available
            st.session_state.direct_mode = not st.session_state.agent.server_available
            st.session_state.tools = await st.session_state.agent.tool_manager.get_tools()
            
            logger.info(f"✅ Agent initialized successfully!")
            logger.info(f"   → Server connected: {st.session_state.server_connected}")
            logger.info(f"   → Direct mode: {st.session_state.direct_mode}")
            logger.info(f"   → Tools available: {len(st.session_state.tools)}")
            
            tool_logger.info("✅ STREAMLIT AGENT INITIALIZATION SUCCESSFUL")
            tool_logger.info(f"   → Server connected: {st.session_state.server_connected}")
            tool_logger.info(f"   → Tools loaded: {[t.name for t in st.session_state.tools]}")
            
            return True
        else:
            st.session_state.server_connected = False
            st.session_state.direct_mode = True
            st.session_state.last_error = "Initialization failed"
            logger.error("❌ Agent initialization failed")
            tool_logger.error("❌ STREAMLIT AGENT INITIALIZATION FAILED")
            return False
            
    except Exception as e:
        error_msg = f"Initialization error: {str(e)}"
        logger.error(error_msg)
        tool_logger.error(f"❌ STREAMLIT INITIALIZATION ERROR: {str(e)}", exc_info=True)
        st.session_state.last_error = error_msg
        st.session_state.server_connected = False
        st.session_state.direct_mode = True
        return False

async def refresh_connection():
    """Refresh the connection and tools with enhanced logging."""
    try:
        logger.info("🔄 STREAMLIT: Refreshing connection...")
        tool_logger.info("🔄 STREAMLIT CONNECTION REFRESH STARTED")
        
        if st.session_state.agent is None:
            logger.info("Agent is None, reinitializing...")
            tool_logger.info("⚙️ Agent is None, reinitializing...")
            return await initialize_agent()
        
        logger.info("Refreshing tools...")
        tool_logger.info("🧰 Refreshing tools...")
        tools = await st.session_state.agent.tool_manager.refresh_tools()
        st.session_state.server_connected = st.session_state.agent.server_available
        st.session_state.direct_mode = not st.session_state.agent.server_available
        st.session_state.tools = tools
        st.session_state.last_error = None
        
        logger.info(f"✅ Connection refreshed successfully!")
        logger.info(f"   → Server connected: {st.session_state.server_connected}")
        logger.info(f"   → Tools available: {len(tools)}")
        
        tool_logger.info("✅ STREAMLIT CONNECTION REFRESH SUCCESSFUL")
        tool_logger.info(f"   → Server connected: {st.session_state.server_connected}")
        tool_logger.info(f"   → Tools refreshed: {[t.name for t in tools]}")
        
        return True
        
    except Exception as e:
        error_msg = f"Refresh failed: {str(e)}"
        logger.error(error_msg)
        tool_logger.error(f"❌ STREAMLIT CONNECTION REFRESH FAILED: {str(e)}", exc_info=True)
        st.session_state.last_error = error_msg
        st.session_state.server_connected = False
        st.session_state.direct_mode = True
        return False

async def process_user_input(user_input: str):
    """Process user input with enhanced logging."""
    if st.session_state.agent is None:
        logger.error("❌ Agent not initialized")
        tool_logger.error("❌ STREAMLIT: Agent not initialized for query processing")
        add_message("assistant", "❌ Agent not initialized. Please refresh the connection.")
        return
    
    try:
        logger.info(f"🎯 STREAMLIT: Processing user input: {user_input}")
        tool_logger.info("🎯 STREAMLIT USER INPUT PROCESSING STARTED")
        tool_logger.info(f"   → User input: {user_input}")
        tool_logger.info(f"   → Timestamp: {datetime.now().isoformat()}")
        tool_logger.info(f"   → Current mode: {'Direct' if st.session_state.direct_mode else 'Connected'}")
        tool_logger.info(f"   → Available tools: {[t.name for t in st.session_state.tools]}")
        
        st.session_state.processing = True
        add_message("user", user_input)
        
        # Show processing message for tool creation queries
        query_lower = user_input.lower()
        might_create_tool = any(pattern in query_lower for pattern in [
            'factorial', 'area', 'circumference', 'perimeter', 'volume', 'fibonacci', 'prime'
        ]) and not await st.session_state.agent.llm.can_answer_directly(user_input)
        
        if might_create_tool:
            logger.info("🛠️ Query might require tool creation")
            tool_logger.info("🛠️ POTENTIAL TOOL CREATION QUERY DETECTED")
            tool_logger.info(f"   → Query patterns matched: {[p for p in ['factorial', 'area', 'circumference', 'perimeter', 'volume', 'fibonacci', 'prime'] if p in query_lower]}")
            with st.empty():
                st.info("🛠️ Analyzing query... May need to create a specialized tool.")
        
        logger.info("📤 Sending query to agent...")
        tool_logger.info("📤 SENDING QUERY TO AGENT")
        result = await st.session_state.agent.process_query(user_input)
        
        # Enhanced result logging
        logger.info(f"📥 Received result from agent:")
        logger.info(f"   → Success: {result.success}")
        logger.info(f"   → Tool used: {result.tool_used}")
        logger.info(f"   → Response length: {len(result.response)} chars")
        
        tool_logger.info("📥 AGENT RESPONSE RECEIVED")
        tool_logger.info(f"   → Success: {result.success}")
        tool_logger.info(f"   → Tool used: {result.tool_used}")
        tool_logger.info(f"   → Has error: {result.error is not None}")
        tool_logger.info(f"   → Response: {result.response[:200]}{'...' if len(result.response) > 200 else ''}")
        
        if result.success:
            add_message("assistant", result.response)
            
            # Show appropriate success message with enhanced logging
            if result.tool_used:
                logger.info("✅ Tool executed successfully!")
                tool_logger.info("✅ STREAMLIT: Tool execution completed successfully")
                st.success("✅ Tool executed successfully!")
            elif "Created tool" in result.response:
                logger.info("🛠️ New tool created and used!")
                tool_logger.info("🛠️ STREAMLIT: New tool created and used")
                st.success("🛠️ New tool created and used!")
            elif st.session_state.direct_mode:
                logger.info("ℹ️ Answered using direct AI response")
                tool_logger.info("ℹ️ STREAMLIT: Direct AI response provided")
                st.info("ℹ️ Answered using direct AI response")
        else:
            add_message("assistant", f"❌ {result.response}")
            if result.error:
                st.session_state.last_error = result.error
                logger.error(f"❌ Query processing failed: {result.error}")
                tool_logger.error(f"❌ STREAMLIT QUERY PROCESSING FAILED: {result.error}")
                
    except Exception as e:
        error_msg = f"Error processing input: {str(e)}"
        logger.error(error_msg)
        tool_logger.error(f"❌ STREAMLIT INPUT PROCESSING ERROR: {str(e)}", exc_info=True)
        add_message("assistant", f"❌ {error_msg}")
    finally:
        st.session_state.processing = False
        tool_logger.info("🏁 STREAMLIT USER INPUT PROCESSING COMPLETED")

def render_tool_card(tool: MCPTool):
    """Render a tool card with better formatting."""
    st.markdown(f"""
    <div class="tool-card">
        <h4>🔧 {tool.name}</h4>
        <p><strong>Description:</strong> {tool.description}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Show parameters if available
    if isinstance(tool.parameters, dict) and "properties" in tool.parameters:
        with st.expander("View Parameters", expanded=False):
            for param_name, param_info in tool.parameters["properties"].items():
                if isinstance(param_info, dict):
                    param_type = param_info.get("type", "unknown")
                    param_desc = param_info.get("description", "No description")
                    st.write(f"- **{param_name}** ({param_type}): {param_desc}")

def main():
    """Main Streamlit application."""
    # Initialize session state
    init_session_state()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 AgentDRY</h1>
        <p>Dynamic AI Assistant with Smart Tool Creation</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize agent if not done
    if not st.session_state.initialized:
        st.info("🚀 Click the button below to initialize the agent. It will work with or without an MCP server.")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Initialize Agent", type="primary", use_container_width=True):
                logger.info("🚀 STREAMLIT: User clicked initialize button")
                tool_logger.info("🚀 STREAMLIT: User initiated agent initialization")
                with st.spinner("Initializing agent..."):
                    success = asyncio.run(initialize_agent())
                if success:
                    if st.session_state.direct_mode:
                        logger.info("⚠️ Initialized in Direct Mode")
                        tool_logger.info("⚠️ STREAMLIT: Initialized in Direct Mode")
                        st.warning("⚠️ Initialized in Direct Mode (no MCP server). Basic queries work, tools will be created as needed!")
                    else:
                        logger.info("✅ Agent initialized successfully with MCP server!")
                        tool_logger.info("✅ STREAMLIT: Agent initialized with MCP server")
                        st.success("✅ Agent initialized successfully with MCP server!")
                    st.rerun()
                else:
                    logger.error("❌ Failed to initialize agent")
                    tool_logger.error("❌ STREAMLIT: Agent initialization failed")
                    st.error("❌ Failed to initialize agent.")
        return
    
    # Main layout
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Status display
        if st.session_state.direct_mode:
            render_status_card("🟡 Direct Mode", f"Working without MCP server | Will create tools as needed", "warning")
        elif st.session_state.server_connected:
            render_status_card("🟢 Connected", f"Agent ready with {len(st.session_state.tools)} tools available", "success")
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
        
        # Chat input with smart placeholder
        placeholder_text = "Try: 'What is 2+2?', 'Find factorial of 5', 'Area of circle with radius 10', 'Hi there!'"
        
        if user_input := st.chat_input(
            placeholder_text,
            disabled=st.session_state.processing
        ):
            # Enhanced logging for user input
            logger.info(f"📝 STREAMLIT: User submitted input: {user_input}")
            tool_logger.info(f"📝 STREAMLIT USER INPUT SUBMITTED: {user_input}")
            
            # Handle special commands
            if user_input.lower().strip() == "clear":
                logger.info("🗑️ STREAMLIT: Clearing chat")
                tool_logger.info("🗑️ STREAMLIT: Chat cleared by user")
                st.session_state.messages = []
                if st.session_state.agent:
                    st.session_state.agent.clear_conversation()
                st.rerun()
            elif user_input.lower().strip() == "save":
                if st.session_state.messages:
                    logger.info("💾 STREAMLIT: Saving chat")
                    tool_logger.info(f"💾 STREAMLIT: Chat saved with {len(st.session_state.messages)} messages")
                    st.session_state.past_chats.append({
                        "messages": st.session_state.messages.copy(),
                        "timestamp": datetime.now().isoformat()
                    })
                    st.session_state.messages = []
                    if st.session_state.agent:
                        st.session_state.agent.clear_conversation()
                    st.success("💾 Chat saved!")
                    st.rerun()
            else:
                # Process the query
                logger.info(f"⚙️ STREAMLIT: Processing query: {user_input}")
                tool_logger.info(f"⚙️ STREAMLIT: Starting query processing")
                with st.spinner("Processing your request..."):
                    asyncio.run(process_user_input(user_input))
                st.rerun()
        
        # Action buttons
        st.markdown("---")
        col_a, col_b, col_c, col_d = st.columns(4)
        
        with col_a:
            if st.button("🔄 Refresh", disabled=st.session_state.processing):
                logger.info("🔄 STREAMLIT: User clicked refresh button")
                tool_logger.info("🔄 STREAMLIT: User initiated refresh")
                with st.spinner("Refreshing connection..."):
                    success = asyncio.run(refresh_connection())
                if success:
                    tool_count = len(st.session_state.tools)
                    if st.session_state.direct_mode:
                        logger.info("ℹ️ Refreshed! Working in direct mode.")
                        tool_logger.info("ℹ️ STREAMLIT: Refreshed in direct mode")
                        st.info("ℹ️ Refreshed! Working in direct mode.")
                    else:
                        logger.info(f"✅ Refreshed! {tool_count} tools loaded")
                        tool_logger.info(f"✅ STREAMLIT: Refreshed with {tool_count} tools")
                        st.success(f"✅ Refreshed! {tool_count} tools loaded")
                else:
                    logger.error("❌ Refresh failed, using direct mode")
                    tool_logger.error("❌ STREAMLIT: Refresh failed")
                    st.error("❌ Refresh failed, using direct mode")
                st.rerun()
        
        with col_b:
            if st.button("💾 Save Chat", disabled=not st.session_state.messages):
                if st.session_state.messages:
                    logger.info("💾 STREAMLIT: Saving chat via button")
                    tool_logger.info(f"💾 STREAMLIT: Manual chat save with {len(st.session_state.messages)} messages")
                    st.session_state.past_chats.append({
                        "messages": st.session_state.messages.copy(),
                        "timestamp": datetime.now().isoformat()
                    })
                    st.session_state.messages = []
                    if st.session_state.agent:
                        st.session_state.agent.clear_conversation()
                    st.success("💾 Chat saved!")
                    st.rerun()
        
        with col_c:
            if st.button("🗑️ Clear", disabled=not st.session_state.messages):
                logger.info("🗑️ STREAMLIT: Clearing chat via button")
                tool_logger.info("🗑️ STREAMLIT: Manual chat clear")
                st.session_state.messages = []
                if st.session_state.agent:
                    st.session_state.agent.clear_conversation()
                st.rerun()
        
        with col_d:
            if st.button("🔧 Create Tool", disabled=st.session_state.processing):
                logger.info("🔧 STREAMLIT: User clicked create tool button")
                tool_logger.info("🔧 STREAMLIT: Manual tool creation initiated")
                st.session_state.show_tool_creator = not st.session_state.get("show_tool_creator", False)
                st.rerun()
        
        # Tool creation interface
        if st.session_state.get("show_tool_creator", False):
            st.markdown("---")
            with st.expander("🛠️ Manual Tool Creator", expanded=True):
                tool_description = st.text_input(
                    "Tool Description",
                    placeholder="e.g., Calculate the square root of a number"
                )
                
                if st.button("Create Tool") and tool_description:
                    logger.info(f"🔨 STREAMLIT: Creating tool: {tool_description}")
                    tool_logger.info(f"🔨 STREAMLIT MANUAL TOOL CREATION")
                    tool_logger.info(f"   → Tool description: {tool_description}")
                    with st.spinner("Creating tool..."):
                        try:
                            create_and_update_tool(f"Create a function for {tool_description}")
                            logger.info(f"✅ Tool creation initiated for: {tool_description}")
                            tool_logger.info(f"✅ STREAMLIT: Tool creation completed: {tool_description}")
                            st.success(f"✅ Tool creation initiated for: {tool_description}")
                            st.info("🔄 Server will restart. Click 'Refresh' in a few seconds.")
                        except Exception as e:
                            logger.error(f"❌ Tool creation failed: {e}")
                            tool_logger.error(f"❌ STREAMLIT TOOL CREATION FAILED: {str(e)}", exc_info=True)
                            st.error(f"❌ Tool creation failed: {e}")
    
    with col2:
        # Sidebar content
        st.subheader("📊 Dashboard")
        render_metrics()
        
        # Quick status info
        if st.session_state.direct_mode:
            st.info("""
            **🟡 Direct Mode**
            
            ✅ Basic conversations  
            ✅ Simple math (2+2, etc.)  
            ✅ General questions  
            ✅ Tool creation on demand  
            
            Advanced tools work after creation!
            """)
        elif st.session_state.server_connected:
            st.success("""
            **🟢 Connected Mode**
            
            ✅ All existing tools available  
            ✅ Smart tool creation  
            ✅ Full MCP server features  
            """)
        
        # Tools section
        st.subheader("🧰 Available Tools")
        if st.session_state.tools:
            # Group tools by category for better display
            computational_tools = []
            utility_tools = []
            
            for tool in st.session_state.tools:
                if any(keyword in tool.name.lower() for keyword in ['calculate', 'factorial', 'area', 'add', 'divide']):
                    computational_tools.append(tool)
                else:
                    utility_tools.append(tool)
            
            if computational_tools:
                st.markdown("**🧮 Computational Tools**")
                for tool in computational_tools:
                    render_tool_card(tool)
            
            if utility_tools:
                st.markdown("**⚙️ Utility Tools**")
                for tool in utility_tools:
                    render_tool_card(tool)
                    
        elif st.session_state.direct_mode:
            st.info("🔧 No tools yet. Ask me to calculate something and I'll create the right tool!")
        else:
            st.info("🔧 No tools available. Ask to create one!")
        
        # Context memory display
        if st.session_state.agent and hasattr(st.session_state.agent.conversation, 'context_memory'):
            context = st.session_state.agent.conversation.context_memory
            if context:
                st.subheader("🧠 Context Memory")
                for key, value in context.items():
                    st.write(f"• **{key.title()}:** {value}")
        
        # Chat history
        st.subheader("📚 Chat History")
        if st.session_state.past_chats:
            if st.button("🗑️ Clear History"):
                logger.info("🗑️ STREAMLIT: Clearing chat history")
                tool_logger.info("🗑️ STREAMLIT: Chat history cleared")
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
        
        # Enhanced quick examples with logging
        st.subheader("💡 Try These Examples")
        
        with st.expander("🧮 Computational Queries", expanded=False):
            computational_examples = [
                "Find factorial of 5",
                "What is the area of circle with radius 10",
                "Calculate circumference of circle with radius 7",
                "What is 15 + 25"
            ]
            
            for example in computational_examples:
                if st.button(f"📊 {example}", key=f"comp_{hash(example)}"):
                    if not st.session_state.processing:
                        logger.info(f"📊 STREAMLIT: User clicked example: {example}")
                        tool_logger.info(f"📊 STREAMLIT EXAMPLE CLICKED: {example}")
                        with st.spinner("Processing..."):
                            asyncio.run(process_user_input(example))
                        st.rerun()
        
        with st.expander("💬 General Queries", expanded=False):
            general_examples = [
                "Hi, how are you?",
                "What is the capital of Rajasthan?",
                "How does photosynthesis work?",
                "Tell me a joke"
            ]
            
            for example in general_examples:
                if st.button(f"💭 {example}", key=f"gen_{hash(example)}"):
                    if not st.session_state.processing:
                        logger.info(f"💭 STREAMLIT: User clicked example: {example}")
                        tool_logger.info(f"💭 STREAMLIT EXAMPLE CLICKED: {example}")
                        with st.spinner("Processing..."):
                            asyncio.run(process_user_input(example))
                        st.rerun()
        
        with st.expander("🔄 Context Queries", expanded=False):
            st.info("Try these after asking about a circle:")
            context_examples = [
                "What is the area of circle with same radius?",
                "Calculate circumference with same radius"
            ]
            
            for example in context_examples:
                if st.button(f"🔗 {example}", key=f"ctx_{hash(example)}"):
                    if not st.session_state.processing:
                        logger.info(f"🔗 STREAMLIT: User clicked context example: {example}")
                        tool_logger.info(f"🔗 STREAMLIT CONTEXT EXAMPLE CLICKED: {example}")
                        with st.spinner("Processing..."):
                            asyncio.run(process_user_input(example))
                        st.rerun()

    # Footer with tips and logging info
    st.markdown("---")
    
    # Add logging information section
    with st.expander("📋 Logging Information", expanded=False):
        st.markdown("""
        **📁 Log Files Created:**
        - `logs/streamlit.log` - Main Streamlit application logs
        - `logs/client.log` - Client connection and general operation logs  
        - `logs/tool_operations.log` - Detailed tool execution logs
        - `logs/streamlit_tool_operations.log` - Streamlit-specific tool logs
        
        **🔍 What Gets Logged:**
        - Complete tool invocations with names and arguments
        - Parameter extraction and validation
        - Tool execution results and timing
        - Error details with full tracebacks
        - Server connection status changes
        - User interactions and button clicks
        
        **📊 Tool Execution Logs Include:**
        - Tool name and description
        - All arguments passed to the tool
        - Execution timestamps
        - Success/failure status
        - Response content and length
        - Error details when failures occur
        """)
    
    with st.expander("💡 Usage Tips", expanded=False):
        st.markdown("""
        **Smart Tool Creation:**
        - Ask computational questions like "Find factorial of 5" 
        - The agent will create tools automatically when needed
        - Tools are reused for similar future queries
        
        **Context Awareness:**
        - Say "area of circle with same radius" after discussing a specific radius
        - The agent remembers recent context like dimensions
        
        **Direct Answers:**
        - Simple math: "What is 2+2?"
        - General knowledge: "Capital of France?"
        - Greetings and casual conversation
        
        **Commands:**
        - Type "clear" to clear current chat
        - Type "save" to save chat to history
        
        **Monitoring:**
        - Check log files in the `logs/` directory for detailed execution traces
        - Tool operations are logged separately for easy debugging
        """)

if __name__ == "__main__":
    main()