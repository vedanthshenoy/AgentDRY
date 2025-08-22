from utils.append_to_server import create_tool_from_user_input, append_code_to_server_file
import datetime

def create_and_update_tool(query: str, server_file_path : str = r"C:\prass\agentdry\servers\log_mcp_server_autorestart.py"):
    code_string = create_tool_from_user_input(query)
    append_code_to_server_file(code_string, server_file_path)
    print("Server Updated !!!")
    print("Please Wait 🙏") # will route query back in future versions rather than user asking again
    
def delete_tool_from_server(tool_name: str) -> bool:
    """
    Delete a tool from the MCP server by removing its function from the tools directory.
    Add this function to your main.py file.
    """
    import os
    import glob
    import logging
    
    logger = logging.getLogger(__name__)
    
    try:
        # Define the tools directory path
        tools_dir = "tools"
        
        if not os.path.exists(tools_dir):
            logger.error(f"Tools directory '{tools_dir}' does not exist")
            return False
        
        # Find the tool file
        tool_file_pattern = f"{tools_dir}/{tool_name}.py"
        tool_files = glob.glob(tool_file_pattern)
        
        if not tool_files:
            # Try with different naming conventions
            possible_patterns = [
                f"{tools_dir}/*{tool_name}*.py",
                f"{tools_dir}/{tool_name}_*.py",
                f"{tools_dir}/*_{tool_name}.py"
            ]
            
            for pattern in possible_patterns:
                tool_files = glob.glob(pattern)
                if tool_files:
                    break
        
        if not tool_files:
            logger.error(f"Tool file for '{tool_name}' not found")
            return False
        
        # Delete the tool file(s)
        deleted_files = []
        for tool_file in tool_files:
            try:
                os.remove(tool_file)
                deleted_files.append(tool_file)
                logger.info(f"Deleted tool file: {tool_file}")
            except Exception as e:
                logger.error(f"Failed to delete {tool_file}: {e}")
                return False
        
        if deleted_files:
            logger.info(f"Successfully deleted {len(deleted_files)} tool file(s) for '{tool_name}'")
            
            # Trigger server restart if using file watching
            restart_file = "restart_trigger.txt"
            try:
                with open(restart_file, "w") as f:
                    f.write(f"Tool deleted: {tool_name} at {datetime.now()}")
                logger.info("Triggered server restart")
            except Exception as e:
                logger.warning(f"Could not trigger restart: {e}")
            
            return True
        
        return False
        
    except Exception as e:
        logger.error(f"Error deleting tool '{tool_name}': {e}")
        return False
    
if __name__ == "__main__":
    create_and_update_tool("How to fix langchain sandbox issue?")