import asyncio
import logging
import os
import re
import sys
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import httpx
from google import generativeai as genai
from dotenv import load_dotenv
from mcp import ClientSession
from mcp.client.sse import sse_client

# Add parent directory to system path - KEEP EXACTLY SAME
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

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

def setup_tool_logging():
    """Setup a separate logger specifically for tool operations."""
    tool_logger = logging.getLogger("tool_operations")
    tool_handler = logging.FileHandler(os.path.join(LOG_DIR, "tool_operations.log"))
    tool_handler.setFormatter(logging.Formatter(
        '%(asctime)s - TOOL - %(levelname)s - %(message)s'
    ))
    tool_logger.addHandler(tool_handler)
    tool_logger.setLevel(logging.INFO)
    return tool_logger

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
                    await asyncio.sleep(2 ** attempt)
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
        """Determine if a query can be answered directly without specialized tools."""
        query_lower = query.lower().strip()
        
        # Simple math expressions that don't need tools
        simple_math_patterns = [
            r'what\s+is\s+\d+\s*[\+\-\*\/]\s*\d+(?:\?)?$',
            r'^\d+\s*[\+\-\*\/]\s*\d+(?:\?)?$',
            r'calculate\s+\d+\s*[\+\-\*\/]\s*\d+(?:\?)?$',
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
            r'explain\s+(?!.*\b(?:calculate|compute|find|determine)\b)',
            r'define\s+',
            r'how\s+does\s+\w+\s+work',
            r'what\s+is\s+the\s+formula\s+of',
            r'how\s+to\s+(?!.*\b(?:calculate|compute|find|determine)\b)',
        ]
        
        for pattern in conversational_patterns:
            if re.search(pattern, query_lower):
                return True
                
        return False
    
    async def needs_tool_creation(self, query: str, available_tools: List[MCPTool]) -> Tuple[bool, str]:
        """Determine if a query requires creating a new tool."""
        # Check if we can answer directly first
        if await self.can_answer_directly(query):
            return False, ""
        
        # Check if existing tools can handle this - IMPROVED MATCHING
        matching_tool = self._find_matching_tool_improved(query, available_tools)
        if matching_tool:
            logger.info(f"Found matching tool: {matching_tool.name} for query: {query}")
            return False, ""
        
        # Check if this needs computational tools
        computational_patterns = [
            r'factorial\s+of',
            r'calculate\s+factorial',
            r'find\s+factorial',
            r'area\s+of\s+(circle|rectangle|triangle|square)',
            r'circumference\s+of',
            r'perimeter\s+of',
            r'volume\s+of',
            r'surface\s+area\s+of',
            r'fibonacci\s+(series|sequence|number)',
            r'prime\s+(numbers?|factors?)',
            r'square\s+root\s+of',
            r'power\s+of',
            r'calculate\s+\w+',
            r'compute\s+\w+',
            r'find\s+the\s+\w+\s+of',
            r'determine\s+\w+',
        ]
        
        query_lower = query.lower()
        for pattern in computational_patterns:
            if re.search(pattern, query_lower):
                # Extract the general concept for tool creation
                tool_topic = self._extract_tool_topic(query)
                return True, tool_topic
        
        return False, ""
    
    def _find_matching_tool_improved(self, query: str, tools: List[MCPTool]) -> Optional[MCPTool]:
        """Improved tool matching with better pattern recognition."""
        if not tools:
            return None
            
        query_lower = query.lower()
        
        # Direct keyword matching for common operations
        operation_mappings = {
            'factorial': ['factorial'],
            'area': ['area'],
            'circumference': ['circumference'],
            'perimeter': ['perimeter'],
            'volume': ['volume'],
            'add': ['add', 'sum', 'plus'],
            'subtract': ['subtract', 'minus'],
            'multiply': ['multiply', 'times'],
            'divide': ['divide'],
            'square_root': ['square_root', 'sqrt'],
            'fibonacci': ['fibonacci', 'fib'],
            'prime': ['prime']
        }
        
        # Check for direct operation matches
        for tool in tools:
            tool_name_lower = tool.name.lower()
            
            # Check if the query contains keywords that match this tool
            for operation, keywords in operation_mappings.items():
                if any(keyword in tool_name_lower for keyword in [operation]):
                    if any(keyword in query_lower for keyword in keywords):
                        logger.info(f"Direct keyword match: {tool.name} for query containing '{keywords}'")
                        return tool
            
            # Check for factorial specifically
            if 'factorial' in tool_name_lower and 'factorial' in query_lower:
                return tool
            
            # Check for area calculations
            if 'area' in tool_name_lower and 'area' in query_lower:
                return tool
                
            # Check for circumference calculations  
            if 'circumference' in tool_name_lower and 'circumference' in query_lower:
                return tool
        
        # Fallback to similarity matching
        query_words = set(re.findall(r'\w+', query_lower))
        best_match = None
        best_score = 0
        
        for tool in tools:
            tool_words = set(re.findall(r'\w+', tool.name.lower()))
            description_words = set(re.findall(r'\w+', tool.description.lower()))
            
            # Calculate similarity score
            name_overlap = len(query_words & tool_words) / len(query_words) if query_words else 0
            desc_overlap = len(query_words & description_words) / len(query_words) if query_words else 0
            
            score = name_overlap * 0.7 + desc_overlap * 0.3
            
            if score > best_score and score > 0.2:  # Lower threshold for better matching
                best_score = score
                best_match = tool
        
        if best_match:
            logger.info(f"Similarity match: {best_match.name} (score: {best_score:.2f})")
        
        return best_match
    
    def _find_matching_tool(self, query: str, tools: List[MCPTool]) -> Optional[MCPTool]:
        """Find a tool that matches the query using intelligent matching."""
        return self._find_matching_tool_improved(query, tools)
    
    def _extract_tool_topic(self, query: str) -> str:
        """Extract the general topic for tool creation from a specific query."""
        # Handle specific patterns with CONSISTENT naming
        patterns_and_replacements = [
            (r'factorial\s+of\s+\d+', 'calculate factorial of a number'),
            (r'what\s+is\s+the\s+factorial\s+of\s+\d+', 'calculate factorial of a number'),
            (r'find\s+factorial\s+of\s+\d+', 'calculate factorial of a number'),
            (r'area\s+of\s+(circle|rectangle|triangle|square)\s+with', r'calculate area of \1'),
            (r'circumference\s+of\s+(circle)\s+with', r'calculate circumference of \1'),
            (r'perimeter\s+of\s+(\w+)\s+with', r'calculate perimeter of \1'),
            (r'volume\s+of\s+(\w+)\s+with', r'calculate volume of \1'),
            (r'find\s+the\s+(\w+)\s+of\s+(\w+)\s+with', r'calculate \1 of \2'),
        ]
        
        query_lower = query.lower()
        for pattern, replacement in patterns_and_replacements:
            if re.search(pattern, query_lower):
                result = re.sub(pattern, replacement, query_lower)
                logger.info(f"Topic extraction: '{query}' → '{result}'")
                return result
        
        # Default extraction - remove specific numbers and values
        topic = re.sub(r'\b\d+(\.\d+)?\s*(cm|m|inch|ft|kg|g|lb)?\b', '', query_lower)
        topic = re.sub(r'\bwith\s+radius\s+\d+.*', '', topic)
        topic = re.sub(r'\bof\s+\d+.*', ' of a number', topic)
        topic = topic.strip()
        
        result = topic if topic else query
        logger.info(f"Default topic extraction: '{query}' → '{result}'")
        return result

class MCPToolManager:
    """Manages MCP tools with caching and intelligent matching."""
    
    def __init__(self, connection_manager: ConnectionManager, tool_logger: logging.Logger = None):
        self.connection_manager = connection_manager
        self._tools_cache: List[MCPTool] = []
        self._cache_valid = False
        self.tool_logger = tool_logger or logger
    
    def _transform_schema(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Transform MCP schema to a standardized format."""
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
                logger.info(f"Refreshed {len(tools)} tools: {[t.name for t in tools]}")
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
        """Execute a tool with comprehensive logging."""
        try:
            # ENHANCED LOGGING - Log the complete tool invocation
            self.tool_logger.info("="*60)
            self.tool_logger.info(f"🔧 TOOL EXECUTION STARTED")
            self.tool_logger.info(f"Tool Name: {tool_name}")
            self.tool_logger.info(f"Arguments: {arguments}")
            self.tool_logger.info(f"Arguments Type: {type(arguments)}")
            self.tool_logger.info(f"Arguments Keys: {list(arguments.keys()) if arguments else 'None'}")
            self.tool_logger.info(f"Arguments Values: {list(arguments.values()) if arguments else 'None'}")
            self.tool_logger.info(f"Timestamp: {datetime.now().isoformat()}")
            self.tool_logger.info("="*60)
            
            async with self.connection_manager.get_session() as session:
                self.tool_logger.info(f"📡 Sending tool call to MCP server...")
                self.tool_logger.info(f"   → Tool: {tool_name}")
                self.tool_logger.info(f"   → Args: {arguments}")
                
                result = await asyncio.wait_for(
                    session.call_tool(tool_name, arguments=arguments),
                    timeout=30.0
                )
                
                # Log the raw result
                self.tool_logger.info(f"📥 Raw result received from server:")
                self.tool_logger.info(f"   → Result type: {type(result)}")
                self.tool_logger.info(f"   → Has content: {hasattr(result, 'content') and result.content}")
                
                if result and result.content:
                    response_text = result.content[0].text if result.content[0].text else "Tool executed successfully"
                    
                    # DETAILED SUCCESS LOGGING
                    self.tool_logger.info("✅ TOOL EXECUTION SUCCESSFUL")
                    self.tool_logger.info(f"   → Response: {response_text}")
                    self.tool_logger.info(f"   → Response length: {len(response_text)} characters")
                    self.tool_logger.info(f"   → Response type: {type(response_text)}")
                    self.tool_logger.info("="*60)
                    
                    return QueryResult(success=True, response=response_text, tool_used=True)
                else:
                    self.tool_logger.warning("⚠️ TOOL EXECUTION - NO CONTENT")
                    self.tool_logger.warning(f"   → Result object: {result}")
                    self.tool_logger.warning(f"   → Result attributes: {dir(result) if result else 'None'}")
                    self.tool_logger.warning("="*60)
                    return QueryResult(success=False, response="Tool returned no result", error="Empty result")
                    
        except asyncio.TimeoutError:
            error_msg = f"Tool '{tool_name}' timed out after 30 seconds"
            self.tool_logger.error("⏰ TOOL EXECUTION TIMEOUT")
            self.tool_logger.error(f"   → Tool: {tool_name}")
            self.tool_logger.error(f"   → Arguments: {arguments}")
            self.tool_logger.error("="*60)
            return QueryResult(success=False, response=error_msg, error="timeout")
        except Exception as e:
            error_msg = f"Tool '{tool_name}' execution failed: {str(e)}"
            self.tool_logger.error("❌ TOOL EXECUTION FAILED")
            self.tool_logger.error(f"   → Tool: {tool_name}")
            self.tool_logger.error(f"   → Arguments: {arguments}")
            self.tool_logger.error(f"   → Error: {str(e)}")
            self.tool_logger.error(f"   → Error type: {type(e)}")
            self.tool_logger.error(f"   → Full traceback: ", exc_info=True)
            self.tool_logger.error("="*60)
            return QueryResult(success=False, response=error_msg, error=str(e))

class ConversationManager:
    """Manages conversation history and context."""
    
    def __init__(self, max_history: int = 10):
        self.max_history = max_history
        self.history: List[Dict[str, str]] = []
        self.context_memory: Dict[str, Any] = {}  # For remembering context like "same radius"
    
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
    
    def extract_and_store_context(self, query: str, response: str):
        """Extract and store context information for future reference."""
        # Extract numerical values and their context
        number_patterns = [
            (r'radius\s+(\d+(?:\.\d+)?)', 'radius'),
            (r'length\s+(\d+(?:\.\d+)?)', 'length'),
            (r'width\s+(\d+(?:\.\d+)?)', 'width'),
            (r'height\s+(\d+(?:\.\d+)?)', 'height'),
            (r'side\s+(\d+(?:\.\d+)?)', 'side'),
        ]
        
        for pattern, context_key in number_patterns:
            match = re.search(pattern, query.lower())
            if match:
                self.context_memory[context_key] = float(match.group(1))
                logger.info(f"Stored context: {context_key} = {self.context_memory[context_key]}")
    
    def resolve_context_references(self, query: str) -> str:
        """Resolve references like 'same radius' to actual values."""
        # Handle "same radius", "same length", etc.
        context_references = [
            (r'same\s+radius', 'radius'),
            (r'same\s+length', 'length'),
            (r'same\s+width', 'width'),
            (r'same\s+height', 'height'),
            (r'same\s+side', 'side'),
        ]
        
        resolved_query = query
        for pattern, context_key in context_references:
            if re.search(pattern, query.lower()) and context_key in self.context_memory:
                value = self.context_memory[context_key]
                resolved_query = re.sub(pattern, f"{context_key} {value}", resolved_query, flags=re.IGNORECASE)
                logger.info(f"Resolved '{pattern}' to '{context_key} {value}'")
        
        return resolved_query
    
    def clear(self):
        """Clear conversation history and context."""
        self.history.clear()
        self.context_memory.clear()

class AgentDRY:
    """Main agent class that orchestrates all components."""
    
    def __init__(self, api_key: str, server_url: str = "http://localhost:8000/sse"):
        self.llm = GeminiClient(api_key)
        self.connection_manager = ConnectionManager(server_url)
        self.tool_logger = setup_tool_logging()  # Setup dedicated tool logger
        self.tool_manager = MCPToolManager(self.connection_manager, self.tool_logger)
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
    
    def _convert_to_gemini_schema(self, mcp_schema: Dict[str, Any]) -> Dict[str, Any]:
        """Convert MCP schema to Gemini function calling schema format."""
        if not isinstance(mcp_schema, dict):
            return {
                "type": "object",
                "properties": {},
                "required": []
            }
        
        # Handle MCP schema format
        properties = mcp_schema.get("properties", {})
        required = mcp_schema.get("required", [])
        
        if not isinstance(properties, dict):
            return {
                "type": "object", 
                "properties": {},
                "required": []
            }
        
        # Convert properties to Gemini format
        gemini_properties = {}
        for key, value in properties.items():
            if isinstance(value, dict):
                prop_type = value.get("type", "string")
                description = value.get("description", "")
                
                # Map common types to Gemini-supported types
                type_mapping = {
                    "string": "string",
                    "integer": "integer", 
                    "number": "number",
                    "boolean": "boolean",
                    "array": "array",
                    "object": "object"
                }
                
                gemini_properties[key] = {
                    "type": type_mapping.get(prop_type, "string"),
                    "description": description
                }
        
        return {
            "type": "object",
            "properties": gemini_properties,
            "required": required if isinstance(required, list) else []
        }
    
    def _extract_parameters_from_query(self, query: str, tool_topic: str) -> Dict[str, Any]:
        """Extract parameters from a specific query based on the tool topic with logging."""
        self.tool_logger.info(f"🔍 EXTRACTING PARAMETERS")
        self.tool_logger.info(f"   → Query: {query}")
        self.tool_logger.info(f"   → Tool topic: {tool_topic}")
        
        parameters = {}
        
        # Extract numbers with units or context
        if 'factorial' in tool_topic.lower():
            match = re.search(r'factorial\s+of\s+(\d+)', query.lower())
            if match:
                parameters['number'] = int(match.group(1))
                self.tool_logger.info(f"   → Extracted factorial number: {parameters['number']}")
        
        elif 'area' in tool_topic.lower() and 'circle' in tool_topic.lower():
            match = re.search(r'radius\s+(\d+(?:\.\d+)?)', query.lower())
            if match:
                parameters['radius'] = float(match.group(1))
                self.tool_logger.info(f"   → Extracted circle radius: {parameters['radius']}")
        
        elif 'circumference' in tool_topic.lower():
            match = re.search(r'radius\s+(\d+(?:\.\d+)?)', query.lower())
            if match:
                parameters['radius'] = float(match.group(1))
                self.tool_logger.info(f"   → Extracted circumference radius: {parameters['radius']}")
        
        elif 'add' in tool_topic.lower() or 'sum' in tool_topic.lower():
            numbers = re.findall(r'\d+(?:\.\d+)?', query)
            if len(numbers) >= 2:
                parameters['a'] = float(numbers[0])
                parameters['b'] = float(numbers[1])
                self.tool_logger.info(f"   → Extracted addition numbers: a={parameters['a']}, b={parameters['b']}")
        
        # Add more parameter extraction patterns as needed
        
        self.tool_logger.info(f"   → Final extracted parameters: {parameters}")
        return parameters
    
    def _generate_tool_name(self, tool_topic: str) -> str:
        """Generate a CONSISTENT snake_case tool name from a topic."""
        if not tool_topic:
            return "default_tool"
        
        # CONSISTENT tool name mapping - this ensures same names are generated
        topic_lower = tool_topic.lower()
        
        # Handle specific patterns with exact mappings
        if 'factorial' in topic_lower:
            return 'factorial'
        elif 'area' in topic_lower and 'circle' in topic_lower:
            return 'area_of_circle'
        elif 'area' in topic_lower and 'rectangle' in topic_lower:
            return 'area_of_rectangle'
        elif 'area' in topic_lower and 'triangle' in topic_lower:
            return 'area_of_triangle'
        elif 'circumference' in topic_lower:
            return 'circumference_of_circle'
        elif 'perimeter' in topic_lower:
            return 'perimeter'
        elif 'volume' in topic_lower:
            return 'volume'
        elif 'add' in topic_lower or 'sum' in topic_lower:
            return 'add_numbers'
        elif 'fibonacci' in topic_lower:
            return 'fibonacci'
        elif 'prime' in topic_lower:
            return 'prime'
        
        # Generic conversion as fallback
        cleaned = re.sub(r'^(create|calculate|compute|find|determine)\s+', '', tool_topic, flags=re.IGNORECASE)
        cleaned = re.sub(r'[^a-zA-Z0-9\s]', '', cleaned).lower()
        snake_case = re.sub(r'\s+', '_', cleaned).strip('_')
        snake_case = re.sub(r'_+', '_', snake_case)
        
        # Truncate if too long
        if len(snake_case) > 50:
            snake_case = snake_case[:50].rsplit('_', 1)[0]
        
        return snake_case or "default_action"
    
    async def _try_tool_execution(self, query: str, tools: List[MCPTool]) -> Optional[QueryResult]:
        """Try to execute existing tools for the query with detailed logging."""
        if not tools or not self.server_available:
            self.tool_logger.info("❌ No tools available or server not available for tool execution")
            self.tool_logger.info(f"   → Tools count: {len(tools) if tools else 0}")
            self.tool_logger.info(f"   → Server available: {self.server_available}")
            return None
        
        try:
            self.tool_logger.info("🔍 ATTEMPTING TOOL EXECUTION")
            self.tool_logger.info(f"   → Query: {query}")
            self.tool_logger.info(f"   → Available tools: {[t.name for t in tools]}")
            
            # First, try to find a matching tool using improved matching
            matching_tool = self.llm._find_matching_tool_improved(query, tools)
            if matching_tool:
                self.tool_logger.info("🎯 DIRECT TOOL MATCH FOUND")
                self.tool_logger.info(f"   → Query: {query}")
                self.tool_logger.info(f"   → Matched Tool: {matching_tool.name}")
                self.tool_logger.info(f"   → Tool Description: {matching_tool.description}")
                
                # Extract parameters for direct execution
                tool_topic = matching_tool.description or matching_tool.name
                parameters = self._extract_parameters_from_query(query, tool_topic)
                
                if parameters:
                    self.tool_logger.info(f"   → Extracted Parameters: {parameters}")
                    self.tool_logger.info("   → Proceeding with direct tool execution...")
                    
                    result = await self.tool_manager.execute_tool(matching_tool.name, parameters)
                    return result
                else:
                    self.tool_logger.warning(f"   → No parameters extracted for tool: {matching_tool.name}")
            
            # Fallback to LLM function calling
            self.tool_logger.info("🤖 ATTEMPTING LLM-BASED TOOL SELECTION")
            self.tool_logger.info(f"   → Available tools: {[t.name for t in tools]}")
            
            # Prepare tools for LLM with proper Gemini format
            tool_declarations = []
            for tool in tools:
                # Convert MCP tool parameters to Gemini function declaration format
                gemini_tool = {
                    "function_declarations": [{
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": self._convert_to_gemini_schema(tool.parameters)
                    }]
                }
                tool_declarations.append(gemini_tool)
            
            # Get conversation history for LLM
            history = self.conversation.get_gemini_history()
            history.append({"role": "user", "parts": [{"text": query}]})
            
            # Combine all function declarations into a single tools object
            if tool_declarations:
                all_functions = []
                for tool_decl in tool_declarations:
                    all_functions.extend(tool_decl["function_declarations"])
                
                combined_tools = {"function_declarations": all_functions}
                
                self.tool_logger.info(f"   → Sending {len(all_functions)} tools to LLM for selection")
                self.tool_logger.info(f"   → Function names: {[f['name'] for f in all_functions]}")
                
                response = await asyncio.to_thread(
                    self.llm.model.generate_content,
                    history,
                    tools=combined_tools,
                    generation_config={"temperature": 0.0}
                )
                
                # Process response with detailed logging
                if (hasattr(response, "candidates") and response.candidates and 
                    hasattr(response.candidates[0].content, "parts") and response.candidates[0].content.parts):
                    
                    part = response.candidates[0].content.parts[0]
                    
                    if hasattr(part, "function_call") and part.function_call:
                        # LLM wants to use a tool
                        tool_name = part.function_call.name
                        arguments = dict(part.function_call.args)
                        
                        self.tool_logger.info("🎯 LLM SELECTED TOOL")
                        self.tool_logger.info(f"   → LLM chose tool: {tool_name}")
                        self.tool_logger.info(f"   → LLM provided args: {arguments}")
                        self.tool_logger.info(f"   → Original query: {query}")
                        self.tool_logger.info(f"   → Function call object: {part.function_call}")
                        
                        result = await self.tool_manager.execute_tool(tool_name, arguments)
                        return result
                    else:
                        self.tool_logger.info("   → LLM generated text response instead of tool call")
                        if hasattr(part, "text") and part.text:
                            self.tool_logger.info(f"   → LLM text response: {part.text[:100]}...")
            
            self.tool_logger.info("❌ No suitable tool execution method found")
            return None
            
        except Exception as e:
            self.tool_logger.error("❌ TOOL EXECUTION ATTEMPT FAILED")
            self.tool_logger.error(f"   → Query: {query}")
            self.tool_logger.error(f"   → Error: {str(e)}")
            self.tool_logger.error(f"   → Available tools: {[t.name for t in tools]}")
            self.tool_logger.error(f"   → Full traceback: ", exc_info=True)
            return None
    
    async def process_query(self, query: str) -> QueryResult:
        """Process a user query end-to-end with comprehensive logging."""
        logger.info("🚀 PROCESSING NEW QUERY")
        logger.info(f"   → User Query: {query}")
        logger.info(f"   → Server Available: {self.server_available}")
        logger.info(f"   → Timestamp: {datetime.now().isoformat()}")
        
        try:
            # Resolve context references first
            resolved_query = self.conversation.resolve_context_references(query)
            if resolved_query != query:
                logger.info(f"🔄 Context resolution: '{query}' → '{resolved_query}'")
                self.tool_logger.info(f"🔄 CONTEXT RESOLVED: '{query}' → '{resolved_query}'")
                query = resolved_query
            
            # Add user message to history
            self.conversation.add_message("user", query)
            
            # Check if we can answer directly without tools
            can_answer_directly = await self.llm.can_answer_directly(query)
            logger.info(f"🤔 Can answer directly: {can_answer_directly}")
            
            if can_answer_directly:
                logger.info("📝 Generating direct response...")
                response = await self.llm.generate_response(f"""
Please answer this query directly and naturally: {query}

If it's a simple math problem, calculate it. If it's a greeting or general question, respond conversationally.
""")
                self.conversation.add_message("assistant", response)
                self.conversation.extract_and_store_context(query, response)
                logger.info(f"✅ Direct response generated: {response[:100]}...")
                return QueryResult(success=True, response=response)
            
            # Get available tools if server is available
            tools = []
            if self.server_available:
                try:
                    tools = await self.tool_manager.get_tools()
                    logger.info(f"🧰 Available tools loaded: {[t.name for t in tools]}")
                    self.tool_logger.info(f"🧰 TOOLS AVAILABLE FOR QUERY: {[t.name for t in tools]}")
                except Exception as e:
                    logger.warning(f"Failed to get tools: {e}")
                    self.server_available = False
            
            # IMPROVED: Check for existing tools with better matching
            if tools:
                # Try direct matching first
                matching_tool = self.llm._find_matching_tool_improved(query, tools)
                if matching_tool:
                    logger.info(f"🎯 Found existing tool: {matching_tool.name}")
                    self.tool_logger.info(f"🎯 EXISTING TOOL MATCH: {matching_tool.name}")
                    
                    # Extract parameters and execute directly
                    parameters = self._extract_parameters_from_query(query, matching_tool.description)
                    if parameters:
                        self.tool_logger.info(f"✅ Executing existing tool: {matching_tool.name} with {parameters}")
                        result = await self.tool_manager.execute_tool(matching_tool.name, parameters)
                        if result.success:
                            self.conversation.add_message("assistant", result.response)
                            self.conversation.extract_and_store_context(query, result.response)
                            return result
                
                # Fallback to LLM-based tool execution
                self.tool_logger.info("🔄 Falling back to LLM-based tool execution...")
                tool_result = await self._try_tool_execution(query, tools)
                if tool_result and tool_result.success:
                    self.conversation.add_message("assistant", tool_result.response)
                    self.conversation.extract_and_store_context(query, tool_result.response)
                    return tool_result
            
            # Check if we need to create a new tool
            needs_tool, tool_topic = await self.llm.needs_tool_creation(query, tools)
            logger.info(f"🛠️ Needs tool creation: {needs_tool}, topic: {tool_topic}")
            
            if not needs_tool:
                # Direct LLM response for non-computational queries
                logger.info("📝 Generating direct LLM response...")
                response = await self.llm.generate_response(f"""
Please answer this query directly: {query}

Provide a helpful, informative response based on your knowledge.
""")
                self.conversation.add_message("assistant", response)
                self.conversation.extract_and_store_context(query, response)
                return QueryResult(success=True, response=response)
            
            # Need to create a tool
            logger.info(f"🔨 Creating tool for: {tool_topic}")
            self.tool_logger.info(f"🔨 TOOL CREATION INITIATED")
            self.tool_logger.info(f"   → Query: {query}")
            self.tool_logger.info(f"   → Tool topic: {tool_topic}")
            
            # Import the tool creation function
            try:
                from main import create_and_update_tool
            except ImportError:
                logger.error("Cannot import tool creation function")
                response = await self.llm.generate_response(query)
                self.conversation.add_message("assistant", response)
                return QueryResult(success=True, response=response)
            
            # Generate tool name
            tool_name = self._generate_tool_name(tool_topic)
            self.tool_logger.info(f"   → Generated tool name: {tool_name}")
            
            # Check if tool already exists (with better checking)
            existing_tools = {t.name.lower(): t for t in tools}
            if tool_name.lower() in existing_tools:
                logger.info(f"🔄 Tool '{tool_name}' already exists, using it directly")
                self.tool_logger.info(f"🔄 TOOL ALREADY EXISTS: {tool_name}")
                existing_tool = existing_tools[tool_name.lower()]
                
                # Extract parameters and execute
                parameters = self._extract_parameters_from_query(query, tool_topic)
                if parameters:
                    self.tool_logger.info(f"   → Using existing tool with parameters: {parameters}")
                    result = await self.tool_manager.execute_tool(existing_tool.name, parameters)
                    if result.success:
                        self.conversation.add_message("assistant", result.response)
                        self.conversation.extract_and_store_context(query, result.response)
                        return result
                
                # If direct execution failed, fall through to creation
                logger.warning(f"Tool '{tool_name}' exists but couldn't be executed")
                self.tool_logger.warning(f"⚠️ EXISTING TOOL EXECUTION FAILED: {tool_name}")
            
            # Classify query to determine next steps
            qtype, general_part, specific_part = await self.llm.classify_query(query)
            logger.info(f"📋 Query classification: {qtype} | {general_part} | {specific_part}")
            self.tool_logger.info(f"📋 QUERY CLASSIFICATION: {qtype}")
            self.tool_logger.info(f"   → General part: {general_part}")
            self.tool_logger.info(f"   → Specific part: {specific_part}")
            
            # Create the tool
            self.tool_logger.info(f"🔨 CREATING TOOL: {tool_topic}")
            create_and_update_tool(f"Create a function for {tool_topic}")
            
            if qtype == "general":
                # General query - just confirm tool creation
                response = f"✅ I've created a tool for '{tool_topic}'. The tool has been added to the server successfully!"
                self.conversation.add_message("assistant", response)
                self.tool_logger.info(f"✅ TOOL CREATED (GENERAL): {tool_topic}")
                return QueryResult(success=True, response=response, tool_used=False)
            
            else:
                # Direct query - create tool and then use it
                response = f"🛠️ Created tool for '{tool_topic}'. Let me use it to answer your question..."
                self.conversation.add_message("assistant", response)
                
                # Wait for server restart
                logger.info("⏳ Waiting for server restart after tool creation...")
                self.tool_logger.info("⏳ WAITING FOR SERVER RESTART...")
                await asyncio.sleep(8)
                
                # Refresh tools and try again
                try:
                    self.tool_logger.info("🔄 REFRESHING TOOLS AFTER CREATION...")
                    await self.tool_manager.refresh_tools()
                    updated_tools = await self.tool_manager.get_tools()
                    logger.info(f"🔄 Tools after refresh: {[t.name for t in updated_tools]}")
                    self.tool_logger.info(f"🔄 TOOLS AFTER REFRESH: {[t.name for t in updated_tools]}")
                    
                    # Try to execute the new tool
                    self.tool_logger.info("🔄 ATTEMPTING TO USE NEWLY CREATED TOOL...")
                    tool_result = await self._try_tool_execution(query, updated_tools)
                    if tool_result and tool_result.success:
                        final_response = f"{response}\n\n✅ {tool_result.response}"
                        self.conversation.add_message("assistant", tool_result.response)
                        self.conversation.extract_and_store_context(query, tool_result.response)
                        self.tool_logger.info("✅ NEWLY CREATED TOOL EXECUTED SUCCESSFULLY")
                        return QueryResult(success=True, response=final_response, tool_used=True)
                    else:
                        # Extract parameters manually if LLM didn't call the tool
                        self.tool_logger.info("🔧 ATTEMPTING MANUAL TOOL EXECUTION...")
                        parameters = self._extract_parameters_from_query(query, tool_topic)
                        if parameters:
                            # Try to find the tool by name
                            target_tool = None
                            for tool in updated_tools:
                                if tool.name.lower() == tool_name.lower():
                                    target_tool = tool
                                    break
                            
                            if target_tool:
                                self.tool_logger.info(f"🎯 FOUND TARGET TOOL: {target_tool.name}")
                                self.tool_logger.info(f"   → Manual execution with: {parameters}")
                                manual_result = await self.tool_manager.execute_tool(target_tool.name, parameters)
                                if manual_result.success:
                                    final_response = f"{response}\n\n✅ {manual_result.response}"
                                    self.conversation.add_message("assistant", manual_result.response)
                                    self.conversation.extract_and_store_context(query, manual_result.response)
                                    self.tool_logger.info("✅ MANUAL TOOL EXECUTION SUCCESSFUL")
                                    return QueryResult(success=True, response=final_response, tool_used=True)
                            else:
                                logger.error(f"Could not find tool '{tool_name}' in updated tools: {[t.name for t in updated_tools]}")
                                self.tool_logger.error(f"❌ TARGET TOOL NOT FOUND: {tool_name}")
                                self.tool_logger.error(f"   → Available tools: {[t.name for t in updated_tools]}")
                        
                        # If still no success, inform user
                        fallback_response = f"{response}\n\n⚠️ Tool created but couldn't be executed automatically. Please try your query again."
                        self.tool_logger.warning("⚠️ TOOL CREATED BUT EXECUTION FAILED")
                        return QueryResult(success=True, response=fallback_response, tool_used=False)
                
                except Exception as e:
                    logger.error(f"Error after tool creation: {e}")
                    self.tool_logger.error(f"❌ ERROR AFTER TOOL CREATION: {str(e)}", exc_info=True)
                    fallback_response = f"{response}\n\n❌ Tool created but there was an error using it: {str(e)}"
                    return QueryResult(success=False, response=fallback_response, error=str(e))
            
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            logger.error(error_msg)
            self.tool_logger.error(f"❌ QUERY PROCESSING ERROR: {str(e)}", exc_info=True)
            # Even in error case, try to give some response
            try:
                fallback_response = await self.llm.generate_response(f"I encountered an error, but let me try to help with: {query}")
                self.conversation.add_message("assistant", fallback_response)
                return QueryResult(success=False, response=fallback_response, error=str(e))
            except:
                return QueryResult(success=False, response="I'm sorry, I encountered an error and couldn't process your request.", error=str(e))
    
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