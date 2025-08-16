import asyncio
import json
import logging
import os
import sys
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx
from dotenv import load_dotenv
from groq import Groq, APIError

from mcp import ClientSession
from mcp.client.sse import sse_client

# Add parent directory to system path for local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# --- Local Imports ---
# Assuming these utility functions exist in the specified paths
try:
    from utils.openai_format_check import transform_schema
    from main import create_and_update_tool, delete_tool_from_server
except ImportError:
    print("Error: Could not import necessary utility functions from parent directories.")
    print("Please ensure 'utils/openai_format_check.py' and 'main.py' are accessible.")
    # Define dummy functions to allow the script to be parsed
    def transform_schema(schema): return schema
    def create_and_update_tool(query): pass
    def delete_tool_from_server(tool_name): pass


# --- Setup Logging ---
LOG_DIR = "logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "groq_client_improved.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Load Environment ---
load_dotenv()


# --- Dataclasses for Structure ---
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
    tools_called: List[str] = field(default_factory=list)
    error: Optional[str] = None


# --- Core Components ---

class ConnectionManager:
    """Manages MCP server connections."""
    def __init__(self, server_url: str = "http://localhost:8000/sse"):
        self.server_url = server_url

    async def check_server_health(self) -> bool:
        """Check if the MCP server is responding."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.head(self.server_url)
                return response.status_code == 200
        except httpx.RequestError as e:
            logger.warning(f"Server health check failed: {e}")
            return False

    @asynccontextmanager
    async def get_session(self):
        """Get an MCP session."""
        try:
            streams = sse_client(url=self.server_url)
            async with streams as stream_pair:
                session = ClientSession(*stream_pair)
                async with session:
                    await asyncio.wait_for(session.initialize(), timeout=10.0)
                    yield session
        except Exception as e:
            logger.error(f"MCP connection failed: {e}")
            raise


class GroqClient:
    """Groq API client."""
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("GROQ_API_KEY is required")
        self.client = Groq(api_key=api_key)
        self.model_name = "llama3-8b-8192"

    async def generate_response(self, prompt: str, temperature: float = 0.0) -> str:
        """Generate a simple, non-tool response."""
        messages = [{"role": "user", "content": prompt}]
        try:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                messages=messages,
                model=self.model_name,
                temperature=temperature,
            )
            return response.choices[0].message.content.strip()
        except APIError as e:
            logger.error(f"Groq non-tool generation failed: {e}")
            return "I'm sorry, I couldn't generate a response. Please try again."


class MCPToolManager:
    """Manages MCP tools, including fetching, caching, executing, and deleting."""
    def __init__(self, connection_manager: ConnectionManager):
        self.connection_manager = connection_manager
        self._tools_cache: List[MCPTool] = []
        self._cache_valid = False

    async def refresh_tools(self) -> List[MCPTool]:
        """Refresh the tool cache from the server."""
        try:
            async with self.connection_manager.get_session() as session:
                tools_response = await asyncio.wait_for(session.list_tools(), timeout=10.0)
                tools = [
                    MCPTool(
                        name=tool.name,
                        description=tool.description,
                        parameters=transform_schema(tool.inputSchema),
                    )
                    for tool in tools_response.tools
                ]
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

    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> QueryResult:
        """Execute a tool on the MCP server."""
        try:
            async with self.connection_manager.get_session() as session:
                result = await asyncio.wait_for(
                    session.call_tool(tool_name, arguments=arguments), timeout=30.0
                )
                response_text = "Tool executed successfully but returned no text content."
                if result and result.content and result.content[0].text:
                    response_text = result.content[0].text
                return QueryResult(success=True, response=response_text, tool_used=True)
        except asyncio.TimeoutError:
            return QueryResult(success=False, response=f"Tool '{tool_name}' timed out.", error="timeout")
        except Exception as e:
            return QueryResult(success=False, response=f"Tool '{tool_name}' failed: {e}", error=str(e))

    async def delete_tool(self, tool_name: str) -> bool:
        """Delete a tool from the server."""
        try:
            delete_tool_from_server(tool_name)
            await asyncio.sleep(2)  # Wait for server to reflect changes
            await self.refresh_tools()
            return True
        except Exception as e:
            logger.error(f"Failed to delete tool '{tool_name}': {e}")
            return False


class ConversationManager:
    """Manages conversation history."""
    def __init__(self, max_history: int = 20):
        self.max_history = max_history
        self.history: List[Dict[str, Any]] = []

    def add_message(self, message: Dict[str, Any]):
        """Add a message to history."""
        self.history.append(message)
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

    def get_history(self) -> List[Dict[str, Any]]:
        """Return the current conversation history."""
        return self.history

    def clear(self):
        """Clear conversation history."""
        self.history.clear()


class Agent:
    """Main agent class for Groq with intelligent tool calling and creation."""
    def __init__(
        self,
        api_key: str,
        server_url: str = "http://localhost:8000/sse",
        max_tool_iterations: int = 5,
    ):
        self.llm = GroqClient(api_key)
        self.connection_manager = ConnectionManager(server_url)
        self.tool_manager = MCPToolManager(self.connection_manager)
        self.conversation = ConversationManager()
        self.server_available = False
        self.max_tool_iterations = max_tool_iterations
        self.server_context = "unknown"

    async def initialize(self) -> bool:
        """Initialize the agent, check server health, and analyze context."""
        try:
            self.server_available = await self.connection_manager.check_server_health()
            if self.server_available:
                await self.tool_manager.refresh_tools()
                await self._analyze_server_context()
            return True
        except Exception as e:
            logger.error(f"Agent initialization failed: {e}")
            return False

    async def _analyze_server_context(self):
        """Analyze server context based on available tools."""
        tools = await self.tool_manager.get_tools()
        if not tools:
            self.server_context = "unknown"
            return

        tool_descriptions = "\n".join([f"- {tool.name}: {tool.description}" for tool in tools])
        prompt = (
            "Based on the following tools, what is the primary domain of this server?\n"
            f"{tool_descriptions}\n"
            "Respond with a single word (e.g., 'math', 'file', 'web', 'database')."
        )
        self.server_context = await self.llm.generate_response(prompt)
        logger.info(f"Detected server context: {self.server_context}")

    async def _should_use_tools_or_create(self, query: str) -> str:
        """Decide whether to use tools, create a new tool, or respond directly."""
        tools = await self.tool_manager.get_tools()
        if not tools:
            return "respond_directly"

        tool_descriptions = "\n".join([f"- {tool.name}: {tool.description}" for tool in tools])
        prompt = f"""
        You are an intelligent assistant for an MCP server in the '{self.server_context}' domain.
        Available tools:
        {tool_descriptions}

        User query: "{query}"

        Analyze the query and choose the best approach:
        1. "use_tools": If the query can be answered using the available tools.
        2. "create_tool": If the query needs a new tool that fits the '{self.server_context}' domain, or if the user explicitly asks to create a tool.
        3. "respond_directly": If the query is a general question, a greeting, or outside the server's domain.

        Respond with ONLY ONE of the three options.
        """
        decision = await self.llm.generate_response(prompt)
        decision = decision.strip().lower()

        if decision in ["use_tools", "create_tool", "respond_directly"]:
            logger.info(f"LLM decision for query '{query}': {decision}")
            return decision
        else:
            logger.warning(f"Invalid LLM decision '{decision}', defaulting to respond_directly.")
            return "respond_directly"

    async def _iterative_tool_execution(self, query: str) -> QueryResult:
        """Executes tools iteratively to solve a user's query."""
        tools = await self.tool_manager.get_tools()
        if not tools or not self.server_available:
            return QueryResult(success=False, response="No tools available to process the request.")

        tool_schemas = [{'type': 'function', 'function': tool.__dict__} for tool in tools]
        self.conversation.add_message({"role": "user", "content": query})
        tools_called = []

        for i in range(self.max_tool_iterations):
            logger.info(f"Tool iteration {i + 1}/{self.max_tool_iterations}")
            try:
                response = await asyncio.to_thread(
                    self.llm.client.chat.completions.create,
                    model=self.llm.model_name,
                    messages=self.conversation.get_history(),
                    tools=tool_schemas,
                    tool_choice="auto",
                )
                response_message = response.choices[0].message
                self.conversation.add_message(response_message.dict())

                if not response_message.tool_calls:
                    final_response = response_message.content or "Task completed successfully."
                    return QueryResult(
                        success=True, response=final_response,
                        tool_used=bool(tools_called), tools_called=list(set(tools_called))
                    )

                tool_results = []
                for tool_call in response_message.tool_calls:
                    tool_name = tool_call.function.name
                    tools_called.append(tool_name)
                    try:
                        args = json.loads(tool_call.function.arguments)
                        logger.info(f"LLM calling tool: {tool_name} with args: {args}")
                        result = await self.tool_manager.execute_tool(tool_name, args)
                        tool_results.append({
                            "tool_call_id": tool_call.id, "role": "tool",
                            "name": tool_name, "content": result.response
                        })
                    except json.JSONDecodeError as e:
                        logger.error(f"JSON decoding failed for tool {tool_name}: {e}")
                        tool_results.append({
                            "tool_call_id": tool_call.id, "role": "tool",
                            "name": tool_name, "content": f"Error: Invalid arguments provided. {e}"
                        })

                for res in tool_results:
                    self.conversation.add_message(res)

            except APIError as e:
                logger.error(f"Groq API error during tool execution: {e}")
                return QueryResult(success=False, response=f"API Error: {e}", error=str(e), tools_called=tools_called)
            except Exception as e:
                logger.error(f"Unexpected error during tool execution: {e}", exc_info=True)
                return QueryResult(success=False, response=f"An error occurred: {e}", error=str(e), tools_called=tools_called)

        return QueryResult(success=True, response="Max tool iterations reached.", tool_used=True, tools_called=tools_called)

    async def process_query(self, query: str) -> QueryResult:
        """Process a user query with intelligent decision making."""
        try:
            if not self.server_available:
                response = await self.llm.generate_response(query)
                self.conversation.add_message({"role": "user", "content": query})
                self.conversation.add_message({"role": "assistant", "content": response})
                return QueryResult(success=True, response=response)

            decision = await self._should_use_tools_or_create(query)

            if decision == "use_tools":
                return await self._iterative_tool_execution(query)

            elif decision == "create_tool":
                logger.info("Decision: Create a new tool.")
                try:
                    create_and_update_tool(query=f"Create a function to handle: {query}")
                    await asyncio.sleep(3) # Wait for server restart
                    await self.tool_manager.refresh_tools()
                    await self._analyze_server_context() # Re-analyze context
                    response = "I've created a new tool to handle that. Now, let's try answering your original query again."
                    logger.info(response)
                    # Now attempt to answer the original query with the new tool
                    return await self._iterative_tool_execution(query)
                except Exception as e:
                    logger.error(f"Tool creation failed: {e}")
                    return await self.process_query(query) # Fallback

            else: # decision == "respond_directly"
                response = await self.llm.generate_response(query)
                self.conversation.add_message({"role": "user", "content": query})
                self.conversation.add_message({"role": "assistant", "content": response})
                return QueryResult(success=True, response=response)

        except Exception as e:
            error_msg = f"Error processing query: {e}"
            logger.error(error_msg, exc_info=True)
            return QueryResult(success=False, response="Sorry, I encountered an error.", error=str(e))


async def main():
    """Enhanced CLI interface for the Groq agent."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Error: GROQ_API_KEY environment variable not found.")
        return

    agent = Agent(api_key)
    if not await agent.initialize():
        print("Failed to initialize agent. Please check server connection and logs.")
        return

    print("🤖 Groq Agent Ready!")
    print(f"🔧 Detected server context: {agent.server_context}")
    print("Type 'quit' to exit, 'tools' to list, 'clear' for new session, 'delete <tool_name>' to remove a tool.")

    while True:
        try:
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue
            
            if user_input.lower() == "quit":
                break
            elif user_input.lower() == "clear":
                agent.conversation.clear()
                print("Conversation history cleared.")
                continue
            elif user_input.lower() == "tools":
                tools = await agent.tool_manager.get_tools()
                if tools:
                    print("\nAvailable tools:")
                    for tool in tools:
                        print(f"  - {tool.name}: {tool.description}")
                else:
                    print("No tools available on the server.")
                continue
            elif user_input.lower().startswith("delete "):
                tool_name_to_delete = user_input.split(maxsplit=1)[1]
                print(f"Attempting to delete tool '{tool_name_to_delete}'...")
                if await agent.tool_manager.delete_tool(tool_name_to_delete):
                    print(f"Tool '{tool_name_to_delete}' deleted successfully.")
                else:
                    print(f"Failed to delete tool '{tool_name_to_delete}'.")
                continue

            result = await agent.process_query(user_input)
            print(f"Assistant: {result.response}")

            if result.tool_used and result.tools_called:
                unique_tools = list(set(result.tools_called))
                tool_counts = {tool: result.tools_called.count(tool) for tool in unique_tools}
                print("\n🔧 Tool Usage Summary:")
                for tool, count in tool_counts.items():
                    print(f"  - '{tool}' was called {count} time(s).")

        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"An unexpected error occurred in the main loop: {e}", exc_info=True)

    print("\nGoodbye!")


if __name__ == "__main__":
    asyncio.run(main())