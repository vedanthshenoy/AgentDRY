import logging
import os
import sys
import threading
import time
import signal
import atexit
from pathlib import Path
from mcp.server.fastmcp import FastMCP

# --- Setup Logging ---
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "auto_restart_server.log")),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("AutoRestartServer")

# Global flag for graceful shutdown
shutdown_requested = False
restart_requested = False

def create_mcp_server():
    """Create and configure the MCP server with all tools."""
    mcp = FastMCP("AutoMath")
    
    @mcp.tool()
    def add(a: int, b: int) -> int:
        """Add two integers and return the sum."""
        logger.info(f"Adding {a} + {b} = {a + b}")
        return a + b

    @mcp.tool()
    def subtract(a: int, b: int) -> int:
        """Subtract second integer from the first and return the result."""
        logger.info(f"Subtracting {a} - {b} = {a - b}")
        return a - b

    @mcp.tool()
    def multiply(a: int, b: int) -> int:
        """Multiply two integers and return the product."""
        logger.info(f"Multiplying {a} * {b} = {a * b}")
        return a * b

    # Add new tools here and save to trigger restart
    
    
    

    return mcp

def file_watcher():
    """Watch this file for changes and trigger graceful restart."""
    global shutdown_requested, restart_requested
    current_file = Path(__file__)
    
    try:
        last_modified = current_file.stat().st_mtime
        logger.info(f"File watcher started, monitoring: {current_file}")
        
        while not shutdown_requested:
            time.sleep(1)  # Check every second
            try:
                current_modified = current_file.stat().st_mtime
                if current_modified > last_modified:
                    logger.info("📝 File changed! Triggering restart in 2 seconds...")
                    time.sleep(2)  # Brief delay to ensure file write is complete
                    
                    # Trigger restart
                    logger.info("🔄 Requesting restart...")
                    restart_requested = True
                    shutdown_requested = True
                    break
                    
            except Exception as e:
                logger.error(f"Error checking file: {e}")
                time.sleep(5)  # Wait longer on errors
                
    except Exception as e:
        logger.error(f"File watcher failed: {e}")

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    global shutdown_requested
    logger.info(f"Received signal {signum}, shutting down gracefully...")
    shutdown_requested = True

def cleanup():
    """Cleanup function called on exit."""
    logger.info("🧹 Cleanup completed")

def run_server_instance():
    """Run a single instance of the MCP server."""
    try:
        import uvicorn
        
        # Create fresh MCP server instance
        mcp = create_mcp_server()
        logger.info(f"📋 Server ready with tools. Listening on SSE port 8001...")
        
        # Run the server with uvicorn
        uvicorn.run(mcp.sse_app(), host="0.0.0.0", port=8000)
        
    except Exception as e:
        logger.error(f"❌ Server instance error: {e}", exc_info=True)
        raise

def main():
    global shutdown_requested, restart_requested
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    atexit.register(cleanup)
    
    # Start file watcher in daemon thread
    watcher_thread = threading.Thread(target=file_watcher, daemon=True)
    watcher_thread.start()
    
    # Main server loop with auto-restart
    restart_count = 0
    
    while True:  # Infinite loop for restarts
        restart_count += 1
        logger.info(f"🚀 Starting MCP Server (attempt #{restart_count})")
        
        # Reset flags for this iteration
        shutdown_requested = False
        
        try:
            # Run server instance
            run_server_instance()
            
        except KeyboardInterrupt:
            logger.info("👋 Keyboard interrupt received")
            break
        except Exception as e:
            logger.error(f"❌ Server error: {e}", exc_info=True)
            if not restart_requested:
                logger.info("⏱️  Restarting due to error in 3 seconds...")
                time.sleep(3)
        
        # Check if we should restart or exit
        if restart_requested:
            logger.info("🔄 Restart requested, restarting process...")
            restart_requested = False
            
            # Restart the entire process
            python = sys.executable
            logger.info("🔄 Executing restart...")
            os.execl(python, python, *sys.argv)
            
        elif shutdown_requested:
            logger.info("🛑 Shutdown requested, exiting...")
            break
        else:
            # Unexpected exit, restart after delay
            logger.info("🔄 Server stopped unexpectedly, restarting in 2 seconds...")
            time.sleep(2)
    
    logger.info("🏁 Server shutdown complete")

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("🎯 Auto-Restarting MCP Math Server (SSE)")
    logger.info("   • Edit this file to add new tools")
    logger.info("   • Server will auto-restart when file changes")
    logger.info("   • Claude Desktop will reconnect automatically")
    logger.info("   • Running on http://0.0.0.0:8001/sse")
    logger.info("=" * 60)
    
    try:
        main()
    except Exception as e:
        logger.error(f"💥 Fatal error: {e}", exc_info=True)
        sys.exit(1)