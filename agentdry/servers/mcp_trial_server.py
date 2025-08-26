from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

mcp = FastMCP("Math")


@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b

@mcp.tool()
def create_tool(request: str) -> str:
    """
    Creates and uses a new MCP tool when you feel the query needs a specified tool AND the existing tools wont support answering this query.
    Use this if you are unable to answer, or dont have access to the query 
    
    Args:
        request (str): Description of the tool or function to create.
        
    Returns:
        str: Confirmation message.
    """
    return "This is the tool"


if __name__ == "__main__":
    # Initialize and run the server
    mcp.run()
