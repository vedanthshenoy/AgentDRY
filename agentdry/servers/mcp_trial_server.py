import logging
import os
import sys
import threading
import time
import subprocess
from mcp.server.fastmcp import FastMCP

# --- Setup Logging ---
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "trial_server.log")),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("server")

mcp = FastMCP("Math")

# Keep a registry of tool functions
ALL_TOOLS = {}

def register_tool(func):
    """Register tool with MCP and store in ALL_TOOLS for refresh later."""
    mcp.tool()(func)
    ALL_TOOLS[func.__name__] = func
    return func

# ----------------- TOOLS -----------------

@register_tool
def add(a: int, b: int) -> int:
    """Add two integers and return the sum."""
    return a + b

@register_tool
def subtract(a: int, b: int) -> int:
    """Subtract second integer from the first and return the result."""
    return a - b

@register_tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers and return the product."""
    return a * b

@register_tool
def divide_numbers(numerator: int, denominator: int):
    """Divide one integer by another. Returns result or error message if denominator is zero."""
    try:
        return numerator / denominator
    except ZeroDivisionError:
        return "Cannot divide by zero."

@register_tool
def create_tool(request: str) -> str:
    """Placeholder: Create a new tool dynamically from a description (not implemented)."""
    return f"Requested creation of tool: {request}"

@register_tool
def refresh_tools() -> str:
    """Refresh all registered tools so Claude can see them after reconnect."""
    for func in ALL_TOOLS.values():
        mcp.tool()(func)
    return f"Refreshed {len(ALL_TOOLS)} tools"

# ----------------- HELPERS -----------------

def auto_refresh_on_startup():
    """Ensure tools are registered and log the list on startup."""
    refreshed = refresh_tools()
    tool_list = ", ".join(ALL_TOOLS.keys())
    logger.info(f"Auto refresh on startup: {refreshed}")
    logger.info(f"Available tools: {tool_list}")

def file_watcher():
    """Watch this file for changes and restart the process when modified."""
    current_file = os.path.abspath(__file__)
    last_modified = os.path.getmtime(current_file)
    while True:
        time.sleep(1)
        try:
            current_modified = os.path.getmtime(current_file)
            if current_modified > last_modified:
                logger.info("File changed, restarting server gracefully...")
                last_modified = current_modified
                time.sleep(2)
                python = sys.executable
                os.execl(python, python, *sys.argv)
        except Exception as e:
            logger.error(f"Watcher error: {e}", exc_info=True)
            break

# ----------------- MAIN -----------------

if __name__ == "__main__":
    auto_refresh_on_startup()
    watcher_thread = threading.Thread(target=file_watcher, daemon=True)
    watcher_thread.start()
    logger.info("MCP Trial Server running on SSE :8000")
    try:
        mcp.run(transport="sse")
    except Exception as e:
        logger.error(f"Server failed: {e}", exc_info=True)
        raise
