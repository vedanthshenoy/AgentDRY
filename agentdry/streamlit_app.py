import streamlit as st
import asyncio
import logging
import os
import sys
from datetime import datetime
from dotenv import load_dotenv

# Set project paths
current_dir = os.path.dirname(os.path.abspath(__file__))  # C:\prass\agentdry
project_root = current_dir
clients_dir = os.path.join(project_root, "clients", "gemini_clients")

# Add paths
sys.path.insert(0, project_root)
sys.path.insert(0, clients_dir)

# Import Gemini bridge
from bridge import GeminiBridge

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - STREAMLIT - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("logs/streamlit.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Streamlit settings
st.set_page_config(page_title="AgentDRY", page_icon="🤖", layout="wide")


@st.cache_resource
def get_agent():
    """Initialize and cache GeminiBridge agent."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("❌ GEMINI_API_KEY not found in .env")
        st.stop()

    return GeminiBridge(api_key)


def init_session():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "tools" not in st.session_state:
        st.session_state.tools = []
    if "server_status" not in st.session_state:
        st.session_state.server_status = "Unknown"
    if "agent_initialized" not in st.session_state:
        st.session_state.agent_initialized = False


async def init_agent(agent):
    try:
        ok = await agent.initialize()
        st.session_state.agent_initialized = ok
        st.session_state.server_status = "Connected" if agent.server_available else "Disconnected"
        if agent.server_available:
            st.session_state.tools = await agent.tool_manager.get_tools()
    except Exception as e:
        logger.error(f"Agent init failed: {e}")
        st.session_state.server_status = "Error"


def chat_ui(agent):
    st.subheader("💬 Chat")

    # Show history
    for msg in st.session_state.messages:
        role = "👤 You" if msg["role"] == "user" else "🤖 Assistant"
        st.markdown(f"**{role}:** {msg['content']}")

    # Input form
    with st.form("chat_form", clear_on_submit=True):
        q = st.text_input("Ask something...", placeholder="Type here...")
        send = st.form_submit_button("Send")

    if send and q:
        st.session_state.messages.append({"role": "user", "content": q})
        with st.spinner("Thinking..."):
            try:
                result = asyncio.run(agent.process_query(q))
                st.session_state.messages.append({"role": "assistant", "content": result.response})
                if result.tool_used:
                    st.session_state.tools = asyncio.run(agent.tool_manager.get_tools())
            except Exception as e:
                logger.error(e)
                st.session_state.messages.append({"role": "assistant", "content": f"Error: {e}"})
        st.rerun()


def sidebar_ui(agent):
    st.sidebar.title("🛠 Tools & Controls")

    with st.sidebar.expander("📋 Tools"):
        if st.session_state.tools:
            for tool in st.session_state.tools:
                st.markdown(f"**{tool.name}** – {tool.description}")
                if st.button(f"🗑 Delete {tool.name}"):
                    with st.spinner("Deleting..."):
                        ok = asyncio.run(agent.delete_tool(tool.name))
                        if ok:
                            st.session_state.tools = asyncio.run(agent.tool_manager.get_tools())
                            st.success("Deleted")
                            st.rerun()
        else:
            st.info("No tools available")

    st.sidebar.markdown("---")
    if st.button("🧹 Clear Chat"):
        st.session_state.messages = []
        agent.clear_conversation()
        st.rerun()
    if st.button("🔄 Refresh Tools"):
        st.session_state.tools = asyncio.run(agent.tool_manager.get_tools())
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.write(f"**Server:** {st.session_state.server_status}")
    st.sidebar.write(f"**Tools:** {len(st.session_state.tools)}")
    st.sidebar.write(f"**Messages:** {len(st.session_state.messages)}")


def main():
    os.makedirs("logs", exist_ok=True)
    init_session()
    agent = get_agent()

    if not st.session_state.agent_initialized:
        with st.spinner("🚀 Initializing..."):
            asyncio.run(init_agent(agent))

    chat_ui(agent)
    sidebar_ui(agent)


if __name__ == "__main__":
    main()
