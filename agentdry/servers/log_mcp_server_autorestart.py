"""
Enhanced MCP Server with Dynamic Tool Creation

A minimal server implementation that relies on Alfred (alfred.py) for all
helper functionality. This server focuses solely on defining MCP tools
and handling the main execution loop.
"""

import os
import os
from fastmcp import FastMCP, Context
from alfred import DynamicToolManager


# Initialize the server and Alfred
current_file = os.path.abspath(__file__)
alfred = DynamicToolManager(current_file)
mcp = FastMCP("Enhanced Math Server")


@mcp.tool()
def create_tool(query: str, ctx: Context) -> str:
    """
    Create a new tool dynamically based on user query using LLM sampling.
    This function uses the client's LLM to generate Python code for the requested functionality.
    
    Args:
        query (str): Description of the tool to create (e.g., "create a factorial function")
        ctx (Context): FastMCP context for LLM sampling
    
    Returns:
        str: Status message about tool creation
    """
    return alfred.create_tool(query, ctx)


@mcp.tool()
def list_available_functions(ctx: Context) -> str:
    """
    List all available functions in this MCP server by analyzing the server file.
    
    Args:
        ctx (Context): FastMCP context
        
    Returns:
        str: List of available functions with descriptions
    """
    return alfred.list_available_functions()


@mcp.tool()
def get_server_info(ctx: Context) -> str:
    """
    Get information about this MCP server including capabilities and usage.
    
    Args:
        ctx (Context): FastMCP context
        
    Returns:
        str: Server information and capabilities
    """
    return alfred.get_server_info()


@mcp.tool()
def divide_numbers(numerator: int, denominator: int) -> float:
    """Divides two numbers and handles potential ZeroDivisionError."""
    try:
        result = numerator / denominator
        alfred.logger.info(f"Executed divide_numbers({numerator}, {denominator}). Result: {result}")
        return result
    except ZeroDivisionError:
        alfred.logger.warning(f"Division by zero attempted: {numerator}/{denominator}")
        return "Cannot divide by zero."


@mcp.tool()
def add(a: int, b: int) -> int:
    """Adds two numbers."""
    result = a + b
    alfred.logger.info(f"Executed add({a}, {b}). Result: {result}")
    return result


if __name__ == "__main__":
    # Initialize Alfred and start all systems
    alfred.initialize()
    
    alfred.logger.info("Enhanced MCP Server started with dynamic tool creation capability.")
    alfred.logger.info(f"Watching {current_file} for changes...")
    alfred.logger.info("Use create_tool() function to dynamically add new capabilities!")
    
    try:
        mcp.run(transport='sse')
    except KeyboardInterrupt:
        alfred.logger.info("Server shutdown requested by user")
        alfred.shutdown()
    except Exception as e:
        alfred.logger.error(f"Server failed to run: {e}", exc_info=True)
        alfred.shutdown()
        raise