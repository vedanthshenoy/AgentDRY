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
        logging.FileHandler(os.path.join(LOG_DIR, "server.log")),
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

# --- MCP Server and Tools ---
mcp = FastMCP("Math")

@mcp.tool()
def divide_numbers(numerator: int, denominator: int):
    """Divides two numbers and handles potential ZeroDivisionError."""
    try:
        result = numerator / denominator
        logger.info(f"Executed divide_numbers({numerator}, {denominator}). Result: {result}")
        return result
    except ZeroDivisionError:
        logger.warning(f"Attempted to divide by zero with args: numerator={numerator}, denominator={denominator}")
        return "Cannot divide by zero."

@mcp.tool()
def add(a: int, b: int) -> int:
    """Adds two numbers."""
    result = a + b
    logger.info(f"Executed add({a}, {b}). Result: {result}")
    return result

if __name__ == "__main__":
    watcher_thread = threading.Thread(target=file_watcher, daemon=True)
    watcher_thread.start()
    logger.info(f"MCP Server started. Watching {current_file} for changes...")
    try:
        mcp.run(transport='sse')
    except Exception as e:
        logger.error(f"Server failed to run: {e}", exc_info=True)
        raise