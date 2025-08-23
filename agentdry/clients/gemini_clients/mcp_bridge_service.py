import asyncio
import logging
import os
import sys
import threading
import time
import json
from contextlib import asynccontextmanager
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Union
from enum import Enum
import uuid
import subprocess
import tempfile

import httpx
from dotenv import load_dotenv
from google import generativeai as genai

# Updated imports for proper content handling
try:
    from google.generativeai.types import Content, Part, FunctionCall, FunctionResponse
except ImportError:
    try:
        from google.ai.generativelanguage import Content, Part, FunctionCall, FunctionResponse
    except ImportError:
        Content = genai.protos.Content
        Part = genai.protos.Part
        FunctionCall = genai.protos.FunctionCall
        FunctionResponse = genai.protos.FunctionResponse

from mcp import ClientSession
from mcp.client.sse import sse_client

# Add parent directory to system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("bridge_service.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class QueryStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing" 
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class MCPTool:
    """Represents an MCP tool with its metadata."""
    name: str
    description: str
    parameters: Dict[str, Any]


@dataclass
class QueryRequest:
    """Represents a query request."""
    id: str
    query: str
    timestamp: float
    status: QueryStatus = QueryStatus.PENDING
    
    def to_dict(self):
        return {
            "id": self.id,
            "query": self.query,
            "timestamp": self.timestamp,
            "status": self.status.value
        }


@dataclass 
class QueryResponse:
    """Represents a query response."""
    id: str
    success: bool
    response: str
    tool_used: bool = False
    tools_called: Optional[List[str]] = None
    error: Optional[str] = None
    processing_time: Optional[float] = None
    tool_created: bool = False
    
    def __post_init__(self):
        if self.tools_called is None:
            self.tools_called = []
    
    def to_dict(self):
        return {
            "id": self.id,
            "success": self.success,
            "response": self.response,
            "tool_used": self.tool_used,
            "tools_called": self.tools_called,
            "error": self.error,
            "processing_time": self.processing_time,
            "tool_created": self.tool_created
        }


class ConnectionManager:
    """Manages MCP server connections with health monitoring."""

    def __init__(self, server_url: str = "http://localhost:8000/sse"):
        self.server_url = server_url
        self._health_status = False
        self._last_health_check = 0
        self._health_check_interval = 30  # seconds

    async def check_server_health(self, force: bool = False) -> bool:
        """Check if the MCP server is responding with caching."""
        current_time = time.time()
        
        if not force and (current_time - self._last_health_check) < self._health_check_interval:
            return self._health_status
            
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.head(self.server_url)
                self._health_status = response.status_code == 200
                self._last_health_check = current_time
                return self._health_status
        except httpx.RequestError as e:
            logger.warning(f"Server health check failed: {e}")
            self._health_status = False
            self._last_health_check = current_time
            return False

    @asynccontextmanager
    async def get_session(self):
        """Get an MCP session with retry logic."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                streams = sse_client(url=self.server_url)
                async with streams as stream_pair:
                    session = ClientSession(*stream_pair)
                    async with session:
                        await asyncio.wait_for(session.initialize(), timeout=10.0)
                        yield session
                        return
            except Exception as e:
                logger.error(f"Connection attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)  # Exponential backoff


class GeminiClient:
    """Gemini API client with rate limiting."""

    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required")

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-1.5-flash")
        self._last_request_time = 0
        self._min_request_interval = 0.1  # 100ms between requests

    async def generate_response(self, prompt: str, temperature: float = 0.0) -> str:
        """Generate a response with rate limiting."""
        # Simple rate limiting
        current_time = time.time()
        time_since_last = current_time - self._last_request_time
        if time_since_last < self._min_request_interval:
            await asyncio.sleep(self._min_request_interval - time_since_last)
        
        try:
            response = await asyncio.to_thread(
                self.model.generate_content,
                prompt,
                generation_config={"temperature": temperature},
            )
            self._last_request_time = time.time()
            return response.text.strip()
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise


class ToolCreationManager:
    """Manages dynamic tool creation and server updates."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.temp_dir = tempfile.gettempdir()
    
    async def create_and_deploy_tool(self, query: str, server_context: str) -> Dict[str, Any]:
        """Create a new tool based on the query and deploy it to the server."""
        try:
            # Generate tool code
            tool_code = await self._generate_tool_code(query, server_context)
            if not tool_code:
                return {"success": False, "error": "Failed to generate tool code"}
            
            # Save tool to temporary file
            tool_file = os.path.join(self.temp_dir, f"dynamic_tool_{int(time.time())}.py")
            with open(tool_file, 'w') as f:
                f.write(tool_code)
            
            logger.info(f"Generated tool saved to: {tool_file}")
            
            # Try to restart the server with the new tool
            restart_success = await self._restart_server_with_tool(tool_file)
            
            if restart_success:
                # Give the server time to start
                await asyncio.sleep(3)
                return {
                    "success": True, 
                    "tool_file": tool_file,
                    "message": "Tool created and server restarted successfully"
                }
            else:
                return {
                    "success": False, 
                    "error": "Failed to restart server with new tool",
                    "tool_file": tool_file
                }
                
        except Exception as e:
            logger.error(f"Tool creation failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _generate_tool_code(self, query: str, server_context: str) -> Optional[str]:
        """Generate Python code for a new MCP tool."""
        
        generation_prompt = f"""
You are an expert Python developer creating MCP (Model Context Protocol) tools. 

Context: The MCP server is focused on the "{server_context}" domain.
User Query: "{query}"

Create a complete Python MCP tool that can handle this query. Follow this exact structure:

```python
import asyncio
import logging
from mcp.server.fastmcp import FastMCP

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create the MCP server
mcp = FastMCP("Dynamic Tool Server")

@mcp.tool()
def handle_query(query: str) -> str:
    \"\"\"
    Handle the user's specific query.
    
    Args:
        query: The user's question or request
        
    Returns:
        str: Response to the query
    \"\"\"
    try:
        # TODO: Implement the specific logic for this query
        # This should be tailored to handle: {query}
        
        result = f"Processed query: {{query}}"
        return result
        
    except Exception as e:
        logger.error(f"Error handling query: {{e}}")
        return f"Error processing query: {{str(e)}}"

if __name__ == "__main__":
    mcp.run()
```

Requirements:
1. The function must be named exactly "handle_query"
2. It must take a "query" parameter of type str
3. It must return a string response
4. Implement actual logic relevant to the query: "{query}"
5. Include proper error handling
6. Make the tool actually useful for this specific query

Generate ONLY the Python code, no explanations or markdown formatting.
"""

        try:
            client = GeminiClient(self.api_key)
            code = await client.generate_response(generation_prompt, temperature=0.1)
            
            # Clean up the response to ensure it's valid Python
            if "```python" in code:
                code = code.split("```python")[1].split("```")[0].strip()
            elif "```" in code:
                code = code.split("```")[1].split("```")[0].strip()
            
            # Basic validation
            if "def handle_query" not in code or "mcp.tool()" not in code:
                logger.error("Generated code doesn't contain required components")
                return None
                
            return code
            
        except Exception as e:
            logger.error(f"Code generation failed: {e}")
            return None
    
    async def _restart_server_with_tool(self, tool_file: str) -> bool:
        """Restart the MCP server with the new tool."""
        try:
            # This would typically involve:
            # 1. Stopping the current server
            # 2. Starting a new server with the tool file
            # For now, we'll simulate this by running the tool file directly
            
            # Try to run the tool as a separate process
            cmd = [sys.executable, tool_file]
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=os.path.dirname(tool_file)
            )
            
            # Give it a moment to start
            await asyncio.sleep(1)
            
            # Check if the process is still running
            if process.poll() is None:
                logger.info(f"Tool server started with PID: {process.pid}")
                return True
            else:
                stdout, stderr = process.communicate()
                logger.error(f"Tool server failed to start: {stderr.decode()}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to restart server with tool: {e}")
            return False


class MCPToolManager:
    """Manages MCP tools with intelligent caching."""

    def __init__(self, connection_manager: ConnectionManager):
        self.connection_manager = connection_manager
        self._tools_cache: List[MCPTool] = []
        self._cache_valid = False
        self._cache_timestamp = 0
        self._cache_ttl = 300  # 5 minutes

    def _is_cache_valid(self) -> bool:
        """Check if cache is still valid."""
        return (
            self._cache_valid and
            (time.time() - self._cache_timestamp) < self._cache_ttl
        )

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
                    "description": value.get("description", ""),
                }

        return transformed

    async def refresh_tools(self, force: bool = False) -> List[MCPTool]:
        """Refresh the tool cache from the server."""
        if not force and self._is_cache_valid():
            return self._tools_cache
            
        try:
            async with self.connection_manager.get_session() as session:
                tools_response = await asyncio.wait_for(
                    session.list_tools(), timeout=10.0
                )

                tools = []
                for tool in tools_response.tools:
                    try:
                        mcp_tool = MCPTool(
                            name=getattr(tool, "name", "unknown_tool"),
                            description=getattr(tool, "description", "No description"),
                            parameters=self._transform_schema(
                                getattr(tool, "inputSchema", {})
                            ),
                        )
                        if mcp_tool.name != "unknown_tool":
                            tools.append(mcp_tool)
                    except Exception as e:
                        logger.error(f"Error processing tool: {e}")

                self._tools_cache = tools
                self._cache_valid = True
                self._cache_timestamp = time.time()
                logger.info(f"Refreshed {len(tools)} tools")
                return tools

        except Exception as e:
            logger.error(f"Failed to refresh tools: {e}")
            self._cache_valid = False
            return self._tools_cache if self._tools_cache else []

    async def get_tools(self) -> List[MCPTool]:
        """Get tools, refreshing cache if needed."""
        return await self.refresh_tools()

    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool and return structured result."""
        start_time = time.time()
        try:
            async with self.connection_manager.get_session() as session:
                result = await asyncio.wait_for(
                    session.call_tool(tool_name, arguments=arguments), timeout=30.0
                )

                execution_time = time.time() - start_time
                
                if result and result.content:
                    response_text = (
                        result.content[0].text
                        if result.content[0].text
                        else "Tool executed successfully"
                    )
                    return {
                        "success": True,
                        "response": response_text,
                        "execution_time": execution_time
                    }
                else:
                    return {
                        "success": False,
                        "response": "Tool returned no result",
                        "error": "Empty result",
                        "execution_time": execution_time
                    }

        except asyncio.TimeoutError:
            execution_time = time.time() - start_time
            return {
                "success": False,
                "response": f"Tool '{tool_name}' timed out",
                "error": "timeout",
                "execution_time": execution_time
            }
        except Exception as e:
            execution_time = time.time() - start_time
            return {
                "success": False,
                "response": f"Tool '{tool_name}' execution failed: {str(e)}",
                "error": str(e),
                "execution_time": execution_time
            }


class MCPBridgeService:
    """
    Bridge service that handles all MCP operations in the background.
    Provides a clean interface for chat clients.
    """

    def __init__(
        self,
        api_key: str,
        server_url: str = "http://localhost:8000/sse",
        max_tool_iterations: int = 5,
    ):
        self.llm = GeminiClient(api_key)
        self.connection_manager = ConnectionManager(server_url)
        self.tool_manager = MCPToolManager(self.connection_manager)
        self.tool_creator = ToolCreationManager(api_key)
        self.server_available = False
        self.max_tool_iterations = max_tool_iterations
        self.server_context = ""
        
        # Query management
        self._active_queries: Dict[str, QueryRequest] = {}
        self._completed_queries: Dict[str, QueryResponse] = {}
        self._query_lock = asyncio.Lock()
        
        # Background tasks
        self._background_tasks = set()
        self._health_monitor_task = None
        self._cleanup_task = None
        
        # Service state
        self._initialized = False
        self._running = False

    async def initialize(self) -> bool:
        """Initialize the bridge service."""
        try:
            logger.info("Initializing MCP Bridge Service...")
            
            # Check server health
            self.server_available = await self.connection_manager.check_server_health(force=True)
            
            if self.server_available:
                await self.tool_manager.refresh_tools(force=True)
                await self._analyze_server_context()
                logger.info(f"Server context detected: {self.server_context}")
            else:
                logger.warning("MCP server not available - operating in direct mode")
            
            # Start background tasks
            await self._start_background_tasks()
            
            self._initialized = True
            self._running = True
            logger.info("MCP Bridge Service initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            return False

    async def _start_background_tasks(self):
        """Start background monitoring tasks."""
        # Health monitoring task
        self._health_monitor_task = asyncio.create_task(self._health_monitor())
        self._background_tasks.add(self._health_monitor_task)
        
        # Cleanup task for old queries
        self._cleanup_task = asyncio.create_task(self._cleanup_old_queries())
        self._background_tasks.add(self._cleanup_task)

    async def _health_monitor(self):
        """Background task to monitor server health."""
        while self._running:
            try:
                old_status = self.server_available
                self.server_available = await self.connection_manager.check_server_health()
                
                if old_status != self.server_available:
                    status_msg = "available" if self.server_available else "unavailable"
                    logger.info(f"Server status changed: now {status_msg}")
                    
                    if self.server_available:
                        await self.tool_manager.refresh_tools(force=True)
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
                await asyncio.sleep(60)  # Wait longer on error

    async def _cleanup_old_queries(self):
        """Cleanup old completed queries to prevent memory leaks."""
        while self._running:
            try:
                current_time = time.time()
                cutoff_time = current_time - 3600  # Keep queries for 1 hour
                
                async with self._query_lock:
                    # Remove old completed queries
                    old_query_ids = [
                        qid for qid, response in self._completed_queries.items()
                        if response.processing_time and (current_time - response.processing_time) > 3600
                    ]
                    
                    for qid in old_query_ids:
                        del self._completed_queries[qid]
                    
                    if old_query_ids:
                        logger.info(f"Cleaned up {len(old_query_ids)} old queries")
                
                await asyncio.sleep(1800)  # Cleanup every 30 minutes
                
            except Exception as e:
                logger.error(f"Cleanup task error: {e}")
                await asyncio.sleep(3600)  # Wait longer on error

    async def _analyze_server_context(self):
        """Analyze the server context based on available tools."""
        tools = await self.tool_manager.get_tools()
        if not tools:
            self.server_context = "unknown"
            return
            
        tool_descriptions = [f"{tool.name}: {tool.description}" for tool in tools]
        tools_text = "\n".join(tool_descriptions)
        
        analysis_prompt = f"""
        Based on these available tools, identify the domain/context of this MCP server:
        
        {tools_text}
        
        Respond with just one word describing the domain (e.g., "math", "file", "web", "database", "api", etc.)
        """
        
        try:
            context = await self.llm.generate_response(analysis_prompt)
            self.server_context = context.lower().strip()
            logger.info(f"Detected server context: {self.server_context}")
        except Exception as e:
            logger.error(f"Failed to analyze server context: {e}")
            self.server_context = "unknown"

    async def submit_query(self, query: str) -> str:
        """
        Submit a query for processing and return a query ID.
        This is the main interface for chat clients.
        """
        if not self._initialized:
            raise RuntimeError("Bridge service not initialized")
        
        query_id = str(uuid.uuid4())
        request = QueryRequest(
            id=query_id,
            query=query,
            timestamp=time.time()
        )
        
        async with self._query_lock:
            self._active_queries[query_id] = request
        
        # Process query in background
        task = asyncio.create_task(self._process_query_background(query_id))
        self._background_tasks.add(task)
        
        logger.info(f"Query submitted with ID: {query_id}")
        return query_id

    async def get_query_status(self, query_id: str) -> Optional[Dict[str, Any]]:
        """Get the current status of a query."""
        async with self._query_lock:
            # Check active queries
            if query_id in self._active_queries:
                return self._active_queries[query_id].to_dict()
            
            # Check completed queries
            if query_id in self._completed_queries:
                response = self._completed_queries[query_id]
                return {
                    **response.to_dict(),
                    "status": QueryStatus.COMPLETED.value
                }
        
        return None

    async def get_query_result(self, query_id: str) -> Optional[QueryResponse]:
        """Get the result of a completed query."""
        async with self._query_lock:
            return self._completed_queries.get(query_id)

    async def _process_query_background(self, query_id: str):
        """Process a query in the background."""
        start_time = time.time()
        
        try:
            async with self._query_lock:
                if query_id not in self._active_queries:
                    return
                
                request = self._active_queries[query_id]
                request.status = QueryStatus.PROCESSING
            
            # Process the query
            result = await self._process_query_internal(request.query)
            result.id = query_id
            result.processing_time = time.time() - start_time
            
            # Move to completed queries
            async with self._query_lock:
                if query_id in self._active_queries:
                    del self._active_queries[query_id]
                self._completed_queries[query_id] = result
            
            logger.info(f"Query {query_id} completed in {result.processing_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Error processing query {query_id}: {e}")
            
            # Create error response
            error_result = QueryResponse(
                id=query_id,
                success=False,
                response="An error occurred while processing your query.",
                error=str(e),
                processing_time=time.time() - start_time
            )
            
            async with self._query_lock:
                if query_id in self._active_queries:
                    del self._active_queries[query_id]
                self._completed_queries[query_id] = error_result

    async def _process_query_internal(self, query: str) -> QueryResponse:
        """Internal query processing logic with proper classification."""
        try:
            if not self.server_available:
                # No server available, respond directly
                response = await self.llm.generate_response(f"Please answer this query: {query}")
                return QueryResponse(id="", success=True, response=response)

            # Get available tools and classify the query
            tools = await self.tool_manager.get_tools()
            classification = await self._classify_query(query, tools)
            
            logger.info(f"Query classification: {classification}")
            
            if classification == "use_existing_tools" and tools:
                # Use existing tools
                tool_result = await self._iterative_tool_execution(query, tools)
                if tool_result and tool_result.success:
                    return tool_result
                else:
                    # Tool execution failed, fall back to direct response
                    response = await self.llm.generate_response(f"Please answer this query: {query}")
                    return QueryResponse(id="", success=True, response=response)
                    
            elif classification == "create_new_tool":
                # Create a new tool for this query
                logger.info(f"Creating new tool for query: {query}")
                tool_creation_result = await self.tool_creator.create_and_deploy_tool(query, self.server_context)
                
                if tool_creation_result["success"]:
                    # Tool created successfully, wait a moment for server to restart
                    await asyncio.sleep(5)
                    
                    # Check if server is available again and refresh tools
                    self.server_available = await self.connection_manager.check_server_health(force=True)
                    if self.server_available:
                        await self.tool_manager.refresh_tools(force=True)
                        new_tools = await self.tool_manager.get_tools()
                        
                        # Try to use the new tool
                        if new_tools:
                            tool_result = await self._iterative_tool_execution(query, new_tools)
                            if tool_result and tool_result.success:
                                tool_result.tool_created = True
                                return tool_result
                    
                    # If tool creation succeeded but execution failed, still report success
                    return QueryResponse(
                        id="",
                        success=True,
                        response=f"I created a new tool to handle your request. {tool_creation_result.get('message', '')}",
                        tool_created=True
                    )
                else:
                    # Tool creation failed, fall back to direct response
                    logger.warning(f"Tool creation failed: {tool_creation_result.get('error')}")
                    response = await self.llm.generate_response(f"Please answer this query: {query}")
                    return QueryResponse(id="", success=True, response=response)
            
            else:  # classification == "respond_directly"
                # Generate direct response
                response = await self.llm.generate_response(f"Please answer this query: {query}")
                return QueryResponse(id="", success=True, response=response)

        except Exception as e:
            logger.error(f"Internal query processing error: {e}")
            return QueryResponse(
                id="",
                success=False,
                response="Sorry, I encountered an error processing your request.",
                error=str(e)
            )

    async def _classify_query(self, query: str, available_tools: List[MCPTool]) -> str:
        """Classify the query to determine the best approach."""
        if not available_tools:
            return "respond_directly"
            
        tools_context = "\n".join([f"- {tool.name}: {tool.description}" for tool in available_tools])
        
        classification_prompt = f"""
You are an intelligent query classifier for an MCP (Model Context Protocol) system.

Current Server Context: {self.server_context}
Available Tools:
{tools_context}

User Query: "{query}"

Analyze the query and classify it into one of these categories:

1. "use_existing_tools" - The query can be directly handled by one or more of the existing tools
2. "create_new_tool" - The query requires functionality that doesn't exist in current tools but is clearly within the {self.server_context} domain and would benefit from a specialized tool
3. "respond_directly" - The query is either:
   - A general question that doesn't need tools
   - Outside the {self.server_context} domain completely
   - Better handled with a direct conversational response

Guidelines:
- Only choose "create_new_tool" if the query requires specific functionality missing from existing tools AND is clearly within the server domain
- Choose "use_existing_tools" if any current tool can handle the request
- Choose "respond_directly" for general questions, greetings, or queries completely outside the server domain

Respond with ONLY one of these three options: use_existing_tools, create_new_tool, or respond_directly
"""
        
        try:
            classification = await self.llm.generate_response(classification_prompt, temperature=0.0)
            classification = classification.strip().lower()
            
            valid_classifications = ["use_existing_tools", "create_new_tool", "respond_directly"]
            if classification in valid_classifications:
                return classification
            else:
                logger.warning(f"Invalid classification: {classification}, defaulting to respond_directly")
                return "respond_directly"
        except Exception as e:
            logger.error(f"Error in query classification: {e}")
            return "respond_directly"

    async def _iterative_tool_execution(self, query: str, tools: List[MCPTool]) -> Optional[QueryResponse]:
        """Execute tools iteratively using Gemini function calling with proper multi-turn conversation."""
        if not tools:
            return None

        tools_called = []
        conversation_history = []
        current_query = query
        
        try:
            # Prepare tools for Gemini API
            tool_declarations = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": self._convert_to_gemini_schema(tool.parameters),
                }
                for tool in tools
            ]
            
            if not tool_declarations:
                return None

            # Start conversation with initial query
            conversation_history.append({"role": "user", "parts": [{"text": current_query}]})
            
            # Iterative tool execution loop
            for iteration in range(self.max_tool_iterations):
                logger.info(f"Tool execution iteration {iteration + 1}")
                
                # Generate response with tools
                response = await asyncio.to_thread(
                    self.llm.model.generate_content,
                    conversation_history,
                    tools={"function_declarations": tool_declarations},
                    generation_config={"temperature": 0.0},
                )
                
                # Add the model's response to conversation
                conversation_history.append({
                    "role": "model", 
                    "parts": [{"text": response.text}] if response.text else []
                })
                
                # Check for function calls in the response
                function_calls_made = False
                
                if hasattr(response.candidates[0].content, 'parts'):
                    parts = response.candidates[0].content.parts
                    
                    for part in parts:
                        if hasattr(part, 'function_call') and part.function_call:
                            function_calls_made = True
                            fc = part.function_call
                            tool_name = fc.name
                            arguments = self._convert_proto_map_to_dict(fc.args)
                            
                            # 🔥 Add this line for clean logging
                            logger.info(f"Executing tool: {tool_name} with args: {arguments}")
                            
                            tools_called.append(tool_name)
                            
                            # Execute the tool
                            function_response_content = await self.tool_manager.execute_tool(tool_name, arguments)

                            
                            # Instead of storing proto objects, store plain dicts
                            conversation_history[-1]["parts"].append({
                                "text": f"[Function Call] {tool_name}({arguments})"
                            })

                            conversation_history.append({
                                "role": "function",
                                "parts": [{
                                    "text": f"[Function Response] {tool_name} -> {function_response_content}"
                                }]
                            })

                            logger.info(f"Executing tool: {tool_name} with args: {arguments}")
                            logger.info(f"Tool {tool_name} result: {function_response_content}")

                            
                            logger.info(f"Tool {tool_name} result: {function_response_content}")
                
                # If no function calls were made, we're done
                if not function_calls_made:
                    logger.info(f"No more function calls needed. Final response: {response.text}")
                    break
                
                # Continue the conversation to let the model use the tool results
                if iteration < self.max_tool_iterations - 1:  # Don't add for last iteration
                    conversation_history.append({
                        "role": "user", 
                        "parts": [{"text": "Please continue with the calculation using the results above."}]
                    })
            
            # Get the final response
            if response.text:
                final_response = response.text.strip()
            else:
                final_response = "Calculation completed using tools."
            
            return QueryResponse(
                id="",
                success=True,
                response=final_response,
                tool_used=len(tools_called) > 0,
                tools_called=tools_called
            )

        except Exception as e:
            logger.error(f"Iterative tool execution error: {e}")
            return QueryResponse(
                id="",
                success=False,
                response=f"Tool execution failed: {str(e)}",
                error=str(e),
                tools_called=tools_called
            )

    def _convert_to_gemini_schema(self, mcp_schema: Dict[str, Any]) -> Dict[str, Any]:
        """Convert MCP schema to Gemini function calling schema format."""
        if not isinstance(mcp_schema, dict):
            return {"type": "object", "properties": {}, "required": []}

        properties = mcp_schema.get("properties", {})
        required = mcp_schema.get("required", [])

        if not isinstance(properties, dict):
            return {"type": "object", "properties": {}, "required": []}

        gemini_properties = {}
        for key, value in properties.items():
            if isinstance(value, dict):
                gemini_properties[key] = {
                    "type": value.get("type", "string"),
                    "description": value.get("description", ""),
                }

        return {
            "type": "object",
            "properties": gemini_properties,
            "required": required if isinstance(required, list) else [],
        }

    def _convert_proto_map_to_dict(self, proto_map: Any) -> Any:
        """Convert proto map to Python dictionary."""
        try:
            from proto.marshal.collections.maps import MapComposite
            from proto.marshal.collections.repeated import RepeatedComposite

            if isinstance(proto_map, MapComposite):
                return {
                    key: self._convert_proto_map_to_dict(value)
                    for key, value in proto_map.items()
                }
            if isinstance(proto_map, RepeatedComposite):
                return [self._convert_proto_map_to_dict(item) for item in proto_map]
            return proto_map
        except ImportError:
            if hasattr(proto_map, 'items'):
                return dict(proto_map.items())
            elif hasattr(proto_map, '__iter__') and not isinstance(proto_map, str):
                return list(proto_map)
            return proto_map

    async def get_server_status(self) -> Dict[str, Any]:
        """Get current server status and statistics."""
        tools = await self.tool_manager.get_tools() if self.server_available else []
        
        async with self._query_lock:
            active_count = len(self._active_queries)
            completed_count = len(self._completed_queries)
        
        return {
            "server_available": self.server_available,
            "server_context": self.server_context,
            "available_tools": len(tools),
            "tool_names": [tool.name for tool in tools],
            "active_queries": active_count,
            "completed_queries": completed_count,
            "service_running": self._running
        }

    async def shutdown(self):
        """Gracefully shutdown the bridge service."""
        logger.info("Shutting down MCP Bridge Service...")
        self._running = False
        
        # Cancel background tasks
        for task in self._background_tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        
        logger.info("MCP Bridge Service shutdown complete")


# Example usage and testing
async def main():
    """Test the bridge service."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY not found")
        return

    bridge = MCPBridgeService(api_key)
    
    if not await bridge.initialize():
        print("Failed to initialize bridge service")
        return

    print("🌉 Enhanced MCP Bridge Service is running!")
    print("✨ Now supports dynamic tool creation!")
    
    try:
        # Test different types of queries
        test_queries = [
            "What is 2 + 3?",  # Should classify as create_new_tool or respond_directly
            "Hello, how are you?",  # Should classify as respond_directly
            "Calculate the square root of 16",  # Should classify based on available tools
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n--- Test Query {i}: {query} ---")
            query_id = await bridge.submit_query(query)
            print(f"Query submitted with ID: {query_id}")
            
            # Poll for result
            print("Waiting for result...")
            for _ in range(60):  # Wait up to 60 seconds
                result = await bridge.get_query_result(query_id)
                if result:
                    print(f"Success: {result.success}")
                    print(f"Response: {result.response}")
                    if result.tool_used:
                        print(f"Tools used: {result.tools_called}")
                    if result.tool_created:
                        print("🎉 New tool was created for this query!")
                    break
                await asyncio.sleep(1)
            else:
                print("Query timed out")
        
        # Show server status
        print("\n--- Server Status ---")
        status = await bridge.get_server_status()
        print(f"Server Available: {status['server_available']}")
        print(f"Server Context: {status['server_context']}")
        print(f"Available Tools: {status['available_tools']}")
        print(f"Tool Names: {status['tool_names']}")
        print(f"Active Queries: {status['active_queries']}")
        print(f"Completed Queries: {status['completed_queries']}")
        
    except KeyboardInterrupt:
        pass
    finally:
        await bridge.shutdown()


if __name__ == "__main__":
    asyncio.run(main())