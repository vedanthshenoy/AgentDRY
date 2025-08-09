import asyncio
import logging
import os
import re
import sys
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import httpx
from google import generativeai as genai
from dotenv import load_dotenv
from mcp import ClientSession
from mcp.client.sse import sse_client

# Configure logging
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "client.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()

@dataclass
class MCPTool:
    """Represents an MCP tool with its metadata."""
    name: str
    description: str
    parameters: Dict[str, Any]

@dataclass
class QueryResult:
    """Represents the result of processing a query."""
    success: bool
    response: str
    tool_used: bool = False
    error: Optional[str] = None

class ConnectionManager:
    """Manages MCP server connections with retry logic and health checks."""
    
    def __init__(self, server_url: str = "http://localhost:8000/sse", max_retries: int = 3):
        self.server_url = server_url
        self.max_retries = max_retries
        self._health_check_timeout = 5.0
        
    async def check_server_health(self) -> bool:
        """Check if the MCP server is responding."""
        try:
            async with httpx.AsyncClient(timeout=self._health_check_timeout) as client:
                # Use a HEAD request to check status without downloading the stream
                response = await client.head(self.server_url)
                return response.status_code == 200
        except Exception as e:
            logger.debug(f"Health check failed: {e}")
            return False
    
    @asynccontextmanager
    async def get_session(self):
        """Get an MCP session with automatic retry and cleanup."""
        session = None
        streams = None
        
        for attempt in range(self.max_retries):
            try:
                if not await self.check_server_health():
                    if attempt == self.max_retries - 1:
                        raise ConnectionError("Server not available after all retries")
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    continue
                
                streams = sse_client(url=self.server_url)
                async with streams as stream_pair:
                    session = ClientSession(*stream_pair)
                    async with session:
                        await asyncio.wait_for(session.initialize(), timeout=10.0)
                        yield session
                        return
                        
            except Exception as e:
                logger.warning(f"Connection attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)

class GeminiClient:
    """Enhanced Gemini API client with better error handling and caching."""
    
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-1.5-flash")
        self._classification_cache = {}
        
    async def generate_response(self, prompt: str, temperature: float = 0.0) -> str:
        """Generate a response with proper error handling."""
        try:
            response = await asyncio.to_thread(
                self.model.generate_content,
                prompt,
                generation_config={"temperature": temperature}
            )
            return response.text.strip()
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return "I'm sorry, I couldn't generate a response. Please try again."
    
    async def classify_query(self, query: str) -> Tuple[str, str, str]:
        """Classify query with caching to avoid redundant API calls."""
        if query in self._classification_cache:
            return self._classification_cache[query]
        
        prompt = f"""Classify this query as 'general' or 'direct' and extract parts.

Examples:
- "Calculate factorial of a number" → general | Calculate factorial of a number | 
- "Calculate the factorial of 5" → direct | Calculate factorial of a number | Calculate the factorial of 5
- "What is 2+2" → direct | Add two numbers | What is 2+2
- "hi" → general | Greet user | 

Query: {query}
Format: type | general_part | specific_part"""
        
        try:
            response = await self.generate_response(prompt)
            match = re.match(r"(general|direct)\s*\|\s*(.*?)\s*\|\s*(.*)", response, re.IGNORECASE)
            
            if match:
                result = (match.group(1).lower(), match.group(2).strip(), match.group(3).strip())
                self._classification_cache[query] = result
                return result
        except Exception as e:
            logger.error(f"Query classification failed: {e}")
        
        # Fallback
        result = ("general", query, "")
        self._classification_cache[query] = result
        return result
    
    async def can_answer_directly(self, query: str) -> bool:
        """
        [MODIFIED FOR TESTING] This function is bypassed to always return False,
        forcing the agent to attempt tool use for all queries.
        """
        return False
    
    # async def can_answer_directly(self, query: str) -> bool:
        """Determine if a query can be answered directly without specialized tools."""
        query_lower = query.lower().strip()
        
        # Simple math expressions that don't need tools
        simple_math_patterns = [
            r'what\s+is\s+\d+\s*[\+\-\*\/]\s*\d+',
            r'\d+\s*[\+\-\*\/]\s*\d+',
            r'calculate\s+\d+\s*[\+\-\*\/]\s*\d+',
        ]
        
        for pattern in simple_math_patterns:
            if re.search(pattern, query_lower):
                return True
        
        # Conversational queries
        conversational_patterns = [
            r'^(hi|hello|hey|good\s+(morning|afternoon|evening))',
            r'^(how\s+are\s+you|what\'?s\s+up)',
            r'^(thank\s+you|thanks)',
            r'^(bye|goodbye|see\s+you)',
            r'tell\s+me\s+(about|a)\s+joke',
            r'what\s+is\s+the\s+(capital|population)\s+of',
            r'who\s+is\s+',
            r'when\s+(was|did)',
            r'where\s+is',
            r'explain\s+',
            r'define\s+',
        ]
        
        for pattern in conversational_patterns:
            if re.search(pattern, query_lower):
                return True
                
        return False
    
    async def needs_tool(self, query: str) -> bool:
        """Determine if a query requires a specialized tool."""
        # First check if we can answer directly
        if await self.can_answer_directly(query):
            return False
            
        prompt = f"""Does this query need a specialized computational tool beyond basic math? 
Respond only with 'YES' or 'NO'.

Examples:
- "Calculate Fibonacci series" → YES
- "What is the capital of France?" → NO
- "What is 2+2" → NO
- "Generate a complex report" → YES
- "Hello" → NO
- "Create a Python script" → YES

Query: {query}"""
        
        try:
            response = await self.generate_response(prompt)
            return response.strip().upper() == "YES"
        except Exception as e:
            logger.error(f"Tool necessity check failed: {e}")
            return False

class MCPToolManager:
    """Manages MCP tools with caching and intelligent matching."""
    
    def __init__(self, connection_manager: ConnectionManager):
        self.connection_manager = connection_manager
        self._tools_cache: List[MCPTool] = []
        self._cache_valid = False
    
    def _transform_schema(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Transform MCP schema to Gemini format."""
        if not isinstance(schema, dict):
            return {"type": "object", "properties": {}}
        
        properties = schema.get("properties", {})
        if not isinstance(properties, dict):
            return {"type": "object", "properties": {}}
        
        transformed = {"type": "object", "properties": {}}
        for key, value in properties.items():
            if isinstance(value, dict):
                transformed["properties"][key] = {
                    "type": value.get("type", "string"),
                    "description": value.get("description", "")
                }
        
        return transformed
    
    async def refresh_tools(self) -> List[MCPTool]:
        """Refresh the tool cache from the server."""
        try:
            async with self.connection_manager.get_session() as session:
                tools_response = await asyncio.wait_for(session.list_tools(), timeout=10.0)
                
                tools = []
                for tool in tools_response.tools:
                    try:
                        mcp_tool = MCPTool(
                            name=getattr(tool, "name", "unknown_tool"),
                            description=getattr(tool, "description", "No description"),
                            parameters=self._transform_schema(getattr(tool, "inputSchema", {}))
                        )
                        if mcp_tool.name != "unknown_tool":
                            tools.append(mcp_tool)
                    except Exception as e:
                        logger.error(f"Error processing tool: {e}")
                
                self._tools_cache = tools
                self._cache_valid = True
                logger.info(f"Refreshed {len(tools)} tools")
                return tools
                
        except Exception as e:
            logger.error(f"Failed to refresh tools: {e}")
            self._cache_valid = False
            return []
    
    async def get_tools(self) -> List[MCPTool]:
        """Get tools, refreshing cache if needed."""
        if not self._cache_valid:
            return await self.refresh_tools()
        return self._tools_cache
    
    def find_matching_tool(self, query: str) -> Optional[MCPTool]:
        """Find a tool that matches the query using intelligent matching."""
        query_lower = query.lower()
        query_words = set(re.findall(r'\w+', query_lower))
        
        best_match = None
        best_score = 0
        
        for tool in self._tools_cache:
            tool_words = set(re.findall(r'\w+', tool.name.lower()))
            description_words = set(re.findall(r'\w+', tool.description.lower()))
            
            # Calculate similarity score
            name_overlap = len(query_words & tool_words) / len(query_words) if query_words else 0
            desc_overlap = len(query_words & description_words) / len(query_words) if query_words else 0
            
            score = name_overlap * 0.7 + desc_overlap * 0.3
            
            if score > best_score and score > 0.3:  # Minimum threshold
                best_score = score
                best_match = tool
        
        return best_match
    
    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> QueryResult:
        """Execute a tool with proper error handling."""
        print("Going here....", tool_name)
        try:
            async with self.connection_manager.get_session() as session:
                result = await asyncio.wait_for(
                    session.call_tool(tool_name, arguments=arguments),
                    timeout=30.0
                )
                
                if result and result.content:
                    response = result.content[0].text if result.content[0].text else "Tool executed successfully"
                    return QueryResult(success=True, response=response, tool_used=True)
                else:
                    return QueryResult(success=False, response="Tool returned no result", error="Empty result")
                    
        except asyncio.TimeoutError:
            error_msg = f"Tool '{tool_name}' timed out"
            logger.error(error_msg)
            return QueryResult(success=False, response=error_msg, error="timeout")
        except Exception as e:
            error_msg = f"Tool '{tool_name}' execution failed: {str(e)}"
            logger.error(error_msg)
            return QueryResult(success=False, response=error_msg, error=str(e))

class ConversationManager:
    """Manages conversation history and context."""
    
    def __init__(self, max_history: int = 10):
        self.max_history = max_history
        self.history: List[Dict[str, str]] = []
    
    def add_message(self, role: str, content: str):
        """Add a message to history with automatic truncation."""
        self.history.append({"role": role, "content": content})
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
    
    def get_gemini_history(self) -> List[Dict[str, Any]]:
        """Convert history to Gemini API format."""
        return [
            {
                "role": "model" if msg["role"] == "assistant" else msg["role"],
                "parts": [{"text": msg["content"]}]
            }
            for msg in self.history
        ]
    
    def clear(self):
        """Clear conversation history."""
        self.history.clear()

class AgentDRY:
    """Main agent class that orchestrates all components."""
    
    def __init__(self, api_key: str, server_url: str = "http://localhost:8000/sse"):
        self.llm = GeminiClient(api_key)
        self.connection_manager = ConnectionManager(server_url)
        self.tool_manager = MCPToolManager(self.connection_manager)
        self.conversation = ConversationManager()
        self.server_available = False
    
    async def initialize(self) -> bool:
        """Initialize the agent and refresh tools."""
        try:
            # Check if server is available
            self.server_available = await self.connection_manager.check_server_health()
            
            if self.server_available:
                await self.tool_manager.refresh_tools()
                logger.info("Agent initialized with MCP server connection")
            else:
                logger.warning("Agent initialized without MCP server (direct mode)")
            
            return True
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            self.server_available = False
            return True  # Still allow operation without server
    
    async def process_query(self, query: str) -> QueryResult:
        """Process a user query end-to-end."""
        try:
            # Add user message to history
            self.conversation.add_message("user", query)
            
            # Check if we can answer directly without tools
            if await self.llm.can_answer_directly(query):
                response = await self.llm.generate_response(f"""
Please answer this query directly and naturally: {query}

If it's a simple math problem, calculate it. If it's a greeting or general question, respond conversationally.
""")
                self.conversation.add_message("assistant", response)
                return QueryResult(success=True, response=response)
            
            # Get available tools if server is available
            tools = []
            if self.server_available:
                try:
                    tools = await self.tool_manager.get_tools()
                except Exception as e:
                    logger.warning(f"Failed to get tools: {e}")
                    self.server_available = False
            
            # If no server or no tools, but the query needs tools, try direct LLM
            if not tools:
                if await self.llm.needs_tool(query):
                    # Try to answer anyway with a helpful message
                    response = await self.llm.generate_response(f"""
The user asked: {query}

This might require specialized tools, but I'll do my best to help with the information I have.
Please provide a helpful response, and if you can't fully answer, explain what kind of tool might be needed.
""")
                    self.conversation.add_message("assistant", response)
                    return QueryResult(success=True, response=response)
                else:
                    # Direct response
                    response = await self.llm.generate_response(query)
                    self.conversation.add_message("assistant", response)
                    return QueryResult(success=True, response=response)
            
            # Use tools if available
            try:
                # Prepare tools for LLM
                tool_declarations = [{
                    "function_declarations": [{
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters
                    }]
                } for tool in tools]
                
                # Get conversation history for LLM
                history = self.conversation.get_gemini_history()
                history.append({"role": "user", "parts": [{"text": query}]})
                
                # Generate response with tools
                response = await asyncio.to_thread(
                    self.llm.model.generate_content,
                    history,
                    tools=tool_declarations,
                    generation_config={"temperature": 0.0}
                )
                
                # Process response
                if (hasattr(response, "candidates") and response.candidates and 
                    hasattr(response.candidates[0].content, "parts") and response.candidates[0].content.parts):
                    
                    part = response.candidates[0].content.parts[0]
                    
                    if hasattr(part, "function_call") and part.function_call:
                        # LLM wants to use a tool
                        tool_name = part.function_call.name
                        arguments = dict(part.function_call.args)
                        
                        result = await self.tool_manager.execute_tool(tool_name, arguments)
                        self.conversation.add_message("assistant", result.response)
                        return result
                    
                    elif hasattr(part, "text"):
                        # Direct text response
                        response_text = part.text
                        self.conversation.add_message("assistant", response_text)
                        return QueryResult(success=True, response=response_text)
                
            except Exception as e:
                logger.warning(f"Tool-based processing failed, falling back to direct response: {e}")
                
            # Final fallback - direct LLM response
            response = await self.llm.generate_response(query)
            self.conversation.add_message("assistant", response)
            return QueryResult(success=True, response=response)
            
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            logger.error(error_msg)
            # Even in error case, try to give some response
            fallback_response = await self.llm.generate_response(f"I encountered an error, but let me try to help with: {query}")
            return QueryResult(success=False, response=fallback_response, error=str(e))
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get the current conversation history."""
        return self.conversation.history.copy()
    
    def clear_conversation(self):
        """Clear the conversation history."""
        self.conversation.clear()

# Utility functions for backward compatibility
def generate_tool_name(query: str) -> str:
    """Generate a snake_case tool name from a query."""
    if not query:
        return "default_tool"
    
    # Remove common prefixes
    cleaned = re.sub(r'^(create|generate|write)\s+(a\s+)?(tool|function)\s+(for|to|that)\s+', 
                    '', query, flags=re.IGNORECASE)
    
    # Clean and convert to snake_case
    cleaned = re.sub(r'[^a-zA-Z0-9\s]', '', cleaned).lower()
    snake_case = re.sub(r'\s+', '_', cleaned).strip('_')
    snake_case = re.sub(r'_+', '_', snake_case)
    
    # Truncate if too long
    if len(snake_case) > 50:
        snake_case = snake_case[:50].rsplit('_', 1)[0]
    
    return snake_case or "default_action"

async def main():
    """Example usage of the improved client."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY not found")
        return
    
    agent = AgentDRY(api_key)
    
    if not await agent.initialize():
        logger.error("Failed to initialize agent")
        return
    
    logger.info("Agent initialized successfully. Type 'quit' to exit.")
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            if user_input.lower() == 'quit':
                break
            
            if not user_input:
                continue
            
            result = await agent.process_query(user_input)
            print(f"Assistant: {result.response}")
            
            if not result.success and result.error:
                logger.warning(f"Query had issues: {result.error}")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
    
    logger.info("Agent shutting down.")

if __name__ == "__main__":
    asyncio.run(main())