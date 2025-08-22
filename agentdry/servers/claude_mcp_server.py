import asyncio
import json
import logging
import os
import sys
import threading
import time
import importlib.util
import tempfile
from typing import Any, Dict, List, Optional
from datetime import datetime

from mcp.server.fastmcp import FastMCP
import mcp.server.stdio

# Add the parent directory (agentdry) to the path so we can import utils
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

# Import your existing utilities
try:
    from utils.append_to_server import create_tool_from_user_input, append_code_to_server_file
    from main import create_and_update_tool, delete_tool_from_server
    UTILS_AVAILABLE = False  #Make sure this is True , else you are not truly creating the AgentDRY tool.
    print(f"Successfully imported utils from: {project_root}")
except ImportError as e:
    print(f"Warning: Could not import utility functions: {e}")
    print(f"Attempted to import from: {project_root}")
    UTILS_AVAILABLE = False
    
    def create_tool_from_user_input(query):
        """Fallback implementation"""
        return f"""
@mcp.tool()
def generated_tool(input_data: str):
    \"\"\"Tool generated for: {query}\"\"\"
    return f"Processed: {{input_data}} for query: {query}"
"""

# Setup Logging with ASCII-only characters
LOG_DIR = "logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "agentdry_server.log"), encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# AgentDRY Dynamic Tool Manager
class AgentDRYToolManager:
    def __init__(self, server_instance):
        self.server = server_instance
        self.dynamic_tools = {}
        self.tool_counter = 0
        self.tools_dir = "agentdry_tools"
        
        # Create tools directory if it doesn't exist
        if not os.path.exists(self.tools_dir):
            os.makedirs(self.tools_dir)
    
    def create_and_register_tool(self, query: str) -> str:
        """Create a new tool using your existing logic and register it with the server."""
        try:
            self.tool_counter += 1
            tool_name = f"agentdry_tool_{self.tool_counter}"
            
            # Use your existing tool creation logic
            if UTILS_AVAILABLE:
                logger.info(f"Using AgentDRY tool creation logic for: {query}")
                # Use your main.py function which handles the complete workflow
                server_file_path = os.path.join(project_root, "servers", "claude_mcp_server.py")
                create_and_update_tool(query, server_file_path)
                
                # Also create a local copy for tracking
                tool_code = create_tool_from_user_input(query)
            else:
                # Fallback implementation
                tool_code = self._create_simple_tool(query, tool_name)
            
            # Save the tool code to a file for tracking
            tool_file = os.path.join(self.tools_dir, f"{tool_name}.py")
            with open(tool_file, 'w', encoding='utf-8') as f:
                f.write(tool_code)
                
            try:
                import importlib.util
                spec = importlib.util.spec_from_file_location(tool_name, tool_file)
                module = importlib.util.module_from_spec(spec)
                sys.modules[tool_name] = module
                spec.loader.exec_module(module)

                # assume function inside module has the same name as tool_name
                func = getattr(module, tool_name, None)
                if func:
                    self.server.tool()(func)  # <-- registers it live with FastMCP
                    logger.info(f"AgentDRY registered tool {tool_name} with MCP server")
                else:
                    logger.warning(f"No callable {tool_name} found in {tool_file}")
            except Exception as e:
                logger.error(f"Failed to load/register tool {tool_name}: {e}")
            
            # Store tool info
            self.dynamic_tools[tool_name] = {
                'code': tool_code,
                'file': tool_file,
                'query': query,
                'created_at': datetime.now().isoformat()
            }
            
            logger.info(f"AgentDRY created and registered tool: {tool_name}")
            return tool_name
            
        except Exception as e:
            logger.error(f"AgentDRY failed to create tool for query '{query}': {e}")
            raise
    
    def _create_simple_tool(self, query: str, tool_name: str) -> str:
        """Create a simple tool when utils are not available."""
        return f'''
from mcp.server.fastmcp import FastMCP

def {tool_name}(input_data: str) -> str:
    """AgentDRY tool created for: {query}"""
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"Executing AgentDRY {tool_name} with input: {{input_data}}")
    
    # Simple processing based on query type
    if "calculate" in "{query}".lower():
        try:
            result = eval(input_data) if input_data.replace(".", "").replace("+", "").replace("-", "").replace("*", "").replace("/", "").replace("(", "").replace(")", "").replace(" ", "").isdigit() else "Invalid expression"
            return f"AgentDRY calculation result: {{result}}"
        except:
            return "AgentDRY: Error in calculation"
    elif "text" in "{query}".lower():
        return f"AgentDRY text processed: {{input_data.upper()}}"
    else:
        return f"AgentDRY processed '{{input_data}}' for query: {query}"
'''
    
    def delete_tool(self, tool_name: str) -> bool:
        """Delete a dynamic tool."""
        try:
            if tool_name in self.dynamic_tools:
                # Use your existing delete function if available
                if UTILS_AVAILABLE:
                    success = delete_tool_from_server(tool_name)
                    if not success:
                        logger.warning(f"AgentDRY delete_tool_from_server returned False for {tool_name}")
                
                # Remove local tracking file
                tool_file = self.dynamic_tools[tool_name]['file']
                if os.path.exists(tool_file):
                    os.remove(tool_file)
                
                # Remove from memory
                del self.dynamic_tools[tool_name]
                
                logger.info(f"AgentDRY deleted tool: {tool_name}")
                return True
            return False
        except Exception as e:
            logger.error(f"AgentDRY failed to delete tool {tool_name}: {e}")
            return False
    
    def list_tools(self) -> Dict[str, Any]:
        """List all dynamic tools."""
        return {
            'count': len(self.dynamic_tools),
            'tools': {name: {'query': info['query'], 'created_at': info['created_at']} 
                     for name, info in self.dynamic_tools.items()}
        }

# File Watcher for Auto-Restart
current_file = os.path.abspath(__file__)
last_modified = os.path.getmtime(current_file) if os.path.exists(current_file) else 0

def file_watcher():
    """Monitor file changes and restart AgentDRY server."""
    global last_modified
    while True:
        time.sleep(2)
        try:
            if os.path.exists(current_file):
                current_modified = os.path.getmtime(current_file)
                if current_modified > last_modified:
                    logger.info("AgentDRY file changed, restarting server...")
                    last_modified = current_modified
                    time.sleep(3)
                    python = sys.executable
                    os.execl(python, python, *sys.argv)
        except Exception as e:
            logger.error(f"AgentDRY file watcher error: {e}")

# AgentDRY MCP Server Setup
mcp = FastMCP("AgentDRY")
tool_manager = AgentDRYToolManager(mcp)

# Core Tools
@mcp.tool()
def create_dynamic_tool(query_description: str) -> str:
    """
    Create a new tool dynamically based on a description of what you need.
    
    Args:
        query_description: Description of what the tool should do
    
    Returns:
        Confirmation message with the new tool name
    """
    try:
        tool_name = tool_manager.create_and_register_tool(query_description)
        return f"AgentDRY successfully created tool '{tool_name}' for: {query_description}\nYou can now use this tool in future conversations!"
    except Exception as e:
        error_msg = f"AgentDRY failed to create tool: {str(e)}"
        logger.error(error_msg)
        return error_msg

@mcp.tool()
def list_dynamic_tools() -> str:
    """List all dynamically created tools."""
    try:
        tools_info = tool_manager.list_tools()
        
        if tools_info['count'] == 0:
            return "AgentDRY: No dynamic tools have been created yet."
        
        result = f"AgentDRY Dynamic Tools ({tools_info['count']} total):\n\n"
        
        for tool_name, info in tools_info['tools'].items():
            result += f"* {tool_name}\n"
            result += f"   Created for: {info['query']}\n"
            result += f"   Created at: {info['created_at']}\n\n"
        
        return result
    except Exception as e:
        return f"AgentDRY error listing tools: {str(e)}"

@mcp.tool()
def delete_dynamic_tool(tool_name: str) -> str:
    """
    Delete a dynamically created tool.
    
    Args:
        tool_name: Name of the tool to delete
    
    Returns:
        Confirmation message
    """
    try:
        if tool_manager.delete_tool(tool_name):
            return f"AgentDRY successfully deleted tool: {tool_name}"
        else:
            return f"AgentDRY: Tool '{tool_name}' not found or could not be deleted"
    except Exception as e:
        return f"AgentDRY error deleting tool: {str(e)}"

# Utility Tools
@mcp.tool()
def execute_python_code(code: str) -> str:
    """
    Safely execute Python code in a restricted environment.
    
    Args:
        code: Python code to execute
    
    Returns:
        Result of the code execution
    """
    try:
        # Create a restricted namespace
        namespace = {
            '__builtins__': {
                'abs': abs, 'all': all, 'any': any, 'bin': bin, 'bool': bool,
                'chr': chr, 'dict': dict, 'dir': dir, 'enumerate': enumerate,
                'float': float, 'hex': hex, 'int': int, 'len': len, 'list': list,
                'max': max, 'min': min, 'oct': oct, 'ord': ord, 'pow': pow,
                'range': range, 'repr': repr, 'round': round, 'set': set,
                'sorted': sorted, 'str': str, 'sum': sum, 'tuple': tuple,
                'zip': zip, 'print': print
            },
            'math': __import__('math'),
            'datetime': __import__('datetime'),
            'json': __import__('json')
        }
        
        # Execute the code
        exec(code, namespace)
        tool_name = f"exec_tool_{int(time.time())}.py"
        tool_file = os.path.join(tool_manager.tools_dir, tool_name)
        with open(tool_file, 'w', encoding='utf-8') as f:
            f.write(code)

        tool_name = f"exec_tool_{int(time.time())}.py"

        if "result" in namespace:
            return f"AgentDRY result: {namespace['result']}"
        return "AgentDRY: Code executed successfully (no result returned)"
        
    except Exception as e:
        return f"AgentDRY error executing code: {str(e)}"

@mcp.tool()
def server_status() -> str:
    """Get the current status of the AgentDRY server."""
    try:
        tools_info = tool_manager.list_tools()
        uptime = time.time() - start_time if 'start_time' in globals() else 0
        
        status = f"""╔════════════════════════════════════════════════════╗
║                    AgentDRY                        ║
║            Made in Mangaluru by                    ║
║             Vedanth and Praas team                 ║
╚════════════════════════════════════════════════════╝

Server Status:
   - Uptime: {uptime:.1f} seconds
   - Dynamic tools created: {tools_info['count']}
   - Utils available: {'Yes' if UTILS_AVAILABLE else 'No'}

Available Commands:
   - create_dynamic_tool(description) - Create new tools
   - list_dynamic_tools() - List all dynamic tools  
   - delete_dynamic_tool(name) - Delete a tool
   - execute_python_code(code) - Run Python code safely
   - server_status() - This status report

💡 Tip: Describe what you need and AgentDRY will create a custom tool for you!"""
        
        return status.strip()
    except Exception as e:
        return f"AgentDRY error getting server status: {str(e)}"

if __name__ == "__main__":
    start_time = time.time()
    
    # Start file watcher
    watcher_thread = threading.Thread(target=file_watcher, daemon=True)
    watcher_thread.start()
    
    logger.info("╔════════════════════════════════════════════════════╗")
    logger.info("║                    AgentDRY                        ║")
    logger.info("║            Made in Mangaluru by                    ║")
    logger.info("║             Vedanth and Praas team                 ║")
    logger.info("╚════════════════════════════════════════════════════╝")
    logger.info("AgentDRY Server starting...")
    logger.info(f"AgentDRY tools directory: {tool_manager.tools_dir}")
    logger.info(f"Utils available: {UTILS_AVAILABLE}")
    
    # Check if running in test mode
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        logger.info("AgentDRY running in test mode")
        print("AgentDRY Server initialized successfully!")
        print("Server is ready for MCP communication.")
        time.sleep(2)
        sys.exit(0)
    
    logger.info("AgentDRY ready for Claude Desktop integration")
    
    try:
        # Use stdio transport for Claude Desktop
        mcp.run()
    except KeyboardInterrupt:
        logger.info("AgentDRY Server stopped by user")
    except Exception as e:
        logger.error(f"AgentDRY Server failed to start: {e}", exc_info=True)
        raise