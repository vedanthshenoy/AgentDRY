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

if __name__ == "__main__":
    # Initialize and run the server
    mcp.run()
