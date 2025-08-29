import logging
import os
import sys
import threading
import time
from mcp.server.fastmcp import FastMCP

# --- Setup Logging ---
LOG_DIR = "logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "text_server.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- File Watcher for Auto-Restart ---
current_file = os.path.abspath(__file__)
last_modified = os.path.getmtime(current_file)

def file_watcher():
    """Monitors this file for changes and restarts the server when modified."""
    global last_modified
    while True:
        time.sleep(1)
        try:
            current_modified = os.path.getmtime(current_file)
            if current_modified > last_modified:
                logger.info("File changed, restarting server...")
                last_modified = current_modified
                time.sleep(3)  # Increased delay to allow resource release
                python = sys.executable
                os.execl(python, python, *sys.argv)
        except FileNotFoundError:
            logger.warning(f"Watched file {current_file} not found. Watcher stopping.")
            break
        except Exception as e:
            logger.error(f"Error in file watcher: {e}", exc_info=True)
            

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

# Change domain name here
mcp = FastMCP("Text")

@mcp.tool()
def word_count(text: str) -> int:
    """Count the number of words in the given text"""
    return len(text.split())

@mcp.tool()
def is_palindrome(text: str) -> bool:
    """Check if the given text is a palindrome"""
    cleaned = ''.join(c.lower() for c in text if c.isalnum())
    return cleaned == cleaned[::-1]

@mcp.tool()
def create_tool(request: str) -> str:
    """
    Creates and uses a new MCP tool when you feel the query needs a specified tool 
    AND the existing tools wont support answering this query.
    Use this if you are unable to answer, or don’t have access to the query.
    
    Args:
        request (str): Description of the tool or function to create.
        
    Returns:
        str: Confirmation message.
    """
    return "This is the tool"

# add near the top
import uvicorn

# ... keep your code ...

if __name__ == "__main__":
    watcher_thread = threading.Thread(target=file_watcher, daemon=True)
    watcher_thread.start()
    logger.info(f"MCP Server started. Watching {current_file} for changes...")

    try:
        # If you're using SSE transport:
        uvicorn.run(mcp.sse_app(), host="0.0.0.0", port=8001)

        # If you're using Streamable HTTP instead, use (if available in your version):
        # uvicorn.run(mcp.http_app(), host="0.0.0.0", port=8001)
    except Exception as e:
        logger.error(f"Server failed to run: {e}", exc_info=True)
        raise
