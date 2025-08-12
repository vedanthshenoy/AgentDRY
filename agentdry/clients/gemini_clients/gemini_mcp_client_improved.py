import asyncio
import logging
import os
import sys
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx
from dotenv import load_dotenv
from google import generativeai as genai
from mcp import ClientSession
from mcp.client.sse import sse_client

# Add parent directory to system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# Simple logging setup
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
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
    tools_called: Optional[List[str]] = None
    error: Optional[str] = None

    def __post_init__(self):
        if self.tools_called is None:
            self.tools_called = []


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
            logger.error(f"Connection failed: {e}")
            raise


class GeminiClient:
    """Gemini API client."""

    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required")

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-1.5-flash")

    async def generate_response(self, prompt: str, temperature: float = 0.0) -> str:
        """Generate a response."""
        try:
            response = await asyncio.to_thread(
                self.model.generate_content,
                prompt,
                generation_config={"temperature": temperature},
            )
            return response.text.strip()
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return "I'm sorry, I couldn't generate a response. Please try again."


class MCPToolManager:
    """Manages MCP tools."""

    def __init__(self, connection_manager: ConnectionManager):
        self.connection_manager = connection_manager
        self._tools_cache: List[MCPTool] = []
        self._cache_valid = False

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

    async def refresh_tools(self) -> List[MCPTool]:
        """Refresh the tool cache from the server."""
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

    async def execute_tool(
        self, tool_name: str, arguments: Dict[str, Any]
    ) -> QueryResult:
        """Execute a tool."""
        try:
            async with self.connection_manager.get_session() as session:
                result = await asyncio.wait_for(
                    session.call_tool(tool_name, arguments=arguments), timeout=30.0
                )

                if result and result.content:
                    response_text = (
                        result.content[0].text
                        if result.content[0].text
                        else "Tool executed successfully"
                    )
                    return QueryResult(
                        success=True, response=response_text, tool_used=True
                    )
                else:
                    return QueryResult(
                        success=False, response="Tool returned no result", error="Empty result"
                    )

        except asyncio.TimeoutError:
            error_msg = f"Tool '{tool_name}' timed out"
            return QueryResult(success=False, response=error_msg, error="timeout")
        except Exception as e:
            error_msg = f"Tool '{tool_name}' execution failed: {str(e)}"
            return QueryResult(success=False, response=error_msg, error=str(e))

    async def delete_tool(self, tool_name: str) -> bool:
        """Delete a tool from the server."""
        try:
            # Import the delete function
            from main import delete_tool_from_server

            delete_tool_from_server(tool_name)
            await asyncio.sleep(2)  # Wait for server restart
            await self.refresh_tools()
            return True
        except Exception as e:
            logger.error(f"Failed to delete tool: {e}")
            return False


class ConversationManager:
    """Manages conversation history."""

    def __init__(self, max_history: int = 20):
        self.max_history = max_history
        self.history: List[Dict[str, str]] = []

    def add_message(self, role: str, content: str):
        """Add a message to history."""
        self.history.append({"role": role, "content": content})
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history :]

    def get_gemini_history(self) -> List[Dict[str, Any]]:
        """Convert history to Gemini API format."""
        return [
            {
                "role": "model" if msg["role"] == "assistant" else msg["role"],
                "parts": [{"text": msg["content"]}],
            }
            for msg in self.history
        ]

    def clear(self):
        """Clear conversation history."""
        self.history.clear()


class AgentDRY:
    """Main agent class with iterative tool calling."""

    def __init__(
        self,
        api_key: str,
        server_url: str = "http://localhost:8000/sse",
        max_tool_iterations: int = 5,
    ):
        self.llm = GeminiClient(api_key)
        self.connection_manager = ConnectionManager(server_url)
        self.tool_manager = MCPToolManager(self.connection_manager)
        self.conversation = ConversationManager()
        self.server_available = False
        self.max_tool_iterations = max_tool_iterations  # Prevent infinite loops

    async def initialize(self) -> bool:
        """Initialize the agent."""
        try:
            self.server_available = await self.connection_manager.check_server_health()
            if self.server_available:
                await self.tool_manager.refresh_tools()
            return True
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            return False

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
        """Recursively converts a proto map to a Python dictionary."""
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

    async def _iterative_tool_execution(
        self, query: str, tools: List[MCPTool]
    ) -> Optional[QueryResult]:
        """Execute tools iteratively until the LLM decides it's done."""
        if not tools or not self.server_available:
            return None

        tools_called = []
        conversation_parts = []

        try:
            # Prepare tools for LLM function calling
            tool_declarations = []
            for tool in tools:
                tool_declarations.append(
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": self._convert_to_gemini_schema(tool.parameters),
                    }
                )

            if not tool_declarations:
                return None

            # Start with user query
            conversation_parts = [{"role": "user", "parts": [{"text": query}]}]
            combined_tools = {"function_declarations": tool_declarations}

            # Iterative tool calling loop
            for iteration in range(self.max_tool_iterations):
                logger.info(f"Tool calling iteration {iteration + 1}")

                # Get LLM response
                response = await asyncio.to_thread(
                    self.llm.model.generate_content,
                    conversation_parts,
                    tools=combined_tools,
                    generation_config={"temperature": 0.0},
                )

                if not (hasattr(response, "candidates") and response.candidates):
                    break

                candidate = response.candidates[0]
                if not hasattr(candidate.content, "parts") or not candidate.content.parts:
                    break
                
                # Add model's response to conversation
                conversation_parts.append(candidate.content)

                # Check if any part contains a function call
                function_calls_made = False
                tool_responses = []

                for part in candidate.content.parts:
                    if hasattr(part, "function_call") and part.function_call:
                        function_calls_made = True
                        tool_name = part.function_call.name
                        arguments = self._convert_proto_map_to_dict(
                            part.function_call.args
                        )

                        logger.info(
                            f"LLM called tool: {tool_name} with args: {arguments}"
                        )
                        tools_called.append(tool_name)

                        # Execute the tool
                        tool_result = await self.tool_manager.execute_tool(
                            tool_name, arguments
                        )

                        # Append the tool response in the correct format
                        if tool_result.success:
                            tool_responses.append(
                                {
                                    "role": "tool",
                                    "parts": [
                                        {
                                            "function_response": {
                                                "name": tool_name,
                                                "response": {"result": tool_result.response},
                                            }
                                        }
                                    ],
                                }
                            )
                        else:
                            tool_responses.append(
                                {
                                    "role": "tool",
                                    "parts": [
                                        {
                                            "function_response": {
                                                "name": tool_name,
                                                "response": {
                                                    "result": f"Tool failed: {tool_result.error}"
                                                },
                                            }
                                        }
                                    ],
                                }
                            )

                # If no function calls were made, we're done
                if not function_calls_made:
                    # Extract final response
                    final_text = ""
                    for part in candidate.content.parts:
                        if hasattr(part, "text") and part.text:
                            final_text += part.text

                    return QueryResult(
                        success=True,
                        response=final_text.strip() or "Task completed successfully",
                        tool_used=len(tools_called) > 0,
                        tools_called=tools_called,
                    )

                # Add tool responses to conversation for the next iteration
                conversation_parts.extend(tool_responses)

            # If we exit the loop due to max iterations
            logger.warning(f"Reached maximum tool iterations ({self.max_tool_iterations})")
            return QueryResult(
                success=True,
                response="Task completed after maximum tool iterations.",
                tool_used=len(tools_called) > 0,
                tools_called=tools_called,
            )

        except Exception as e:
            logger.error(f"Iterative tool execution failed: {e}")
            return QueryResult(
                success=False,
                response=f"Tool execution failed: {str(e)}",
                error=str(e),
                tools_called=tools_called,
            )

    async def process_query(self, query: str) -> QueryResult:
        """Process a user query with iterative tool calling."""
        try:
            self.conversation.add_message("user", query)

            # Try existing tools first if server is available
            if self.server_available:
                tools = await self.tool_manager.get_tools()
                tool_result = await self._iterative_tool_execution(query, tools)
                if tool_result and tool_result.success:
                    # Log which tools were used
                    if tool_result.tools_called:
                        logger.info(f"Tools used: {', '.join(tool_result.tools_called)}")

                    self.conversation.add_message("assistant", tool_result.response)
                    return tool_result

            # Check if we should create a new tool
            should_create_tool = await self._should_create_tool(query)

            if should_create_tool and self.server_available:
                # Create tool
                try:
                    from main import create_and_update_tool

                    create_and_update_tool(f"Create a function for: {query}")

                    # Wait for server restart and refresh tools
                    await asyncio.sleep(3)
                    await self.tool_manager.refresh_tools()

                    # Try using the new tool iteratively
                    updated_tools = await self.tool_manager.get_tools()
                    tool_result = await self._iterative_tool_execution(
                        query, updated_tools
                    )
                    if tool_result and tool_result.success:
                        response = f"Created and used new tool(s): {tool_result.response}"
                        if tool_result.tools_called:
                            response += (
                                f" (Tools used: {', '.join(tool_result.tools_called)})"
                            )

                        self.conversation.add_message("assistant", response)
                        return QueryResult(
                            success=True,
                            response=response,
                            tool_used=True,
                            tools_called=tool_result.tools_called,
                        )

                except ImportError:
                    logger.error("Cannot import tool creation function")
                except Exception as e:
                    logger.error(f"Tool creation failed: {e}")

            # Generate direct response
            response = await self.llm.generate_response(
                f"Please answer this query: {query}"
            )
            self.conversation.add_message("assistant", response)
            return QueryResult(success=True, response=response)

        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            logger.error(error_msg)
            return QueryResult(
                success=False, response="Sorry, I encountered an error.", error=str(e)
            )

    async def _should_create_tool(self, query: str) -> bool:
        """Simple heuristic to determine if a tool should be created."""
        query_lower = query.lower()

        # Check for computational keywords
        computational_keywords = [
            "calculate",
            "compute",
            "find",
            "determine",
            "factorial",
            "area",
            "circumference",
            "perimeter",
            "volume",
            "fibonacci",
            "prime",
        ]

        return any(keyword in query_lower for keyword in computational_keywords)

    async def delete_tool(self, tool_name: str) -> bool:
        """Delete a tool."""
        return await self.tool_manager.delete_tool(tool_name)

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get conversation history."""
        return self.conversation.history.copy()

    def clear_conversation(self):
        """Clear conversation history."""
        self.conversation.clear()

    async def process_query_with_details(self, query: str) -> QueryResult:
        """Process query and return detailed information about tool usage."""
        result = await self.process_query(query)

        if result.tool_used and result.tools_called:
            print(f"Tools executed: {', '.join(result.tools_called)}")
            print(f"Total tool calls: {len(result.tools_called)}")

        return result


# Enhanced CLI with tool usage tracking
async def main():
    """Enhanced CLI interface with tool usage tracking."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY not found")
        return

    agent = AgentDRY(api_key, max_tool_iterations=10)  # Allow up to 10 tool iterations

    if not await agent.initialize():
        print("Failed to initialize agent")
        return

    print("Enhanced Agent ready with iterative tool calling!")
    print("Type 'quit' to exit, 'tools' to list available tools, 'clear' to clear history")

    while True:
        try:
            user_input = input("\nYou: ").strip()

            if user_input.lower() == "quit":
                break
            elif user_input.lower() == "clear":
                agent.clear_conversation()
                print("Conversation history cleared.")
                continue
            elif user_input.lower() == "tools":
                tools = await agent.tool_manager.get_tools()
                if tools:
                    print(f"\nAvailable tools ({len(tools)}):")
                    for tool in tools:
                        print(f"  - {tool.name}: {tool.description}")
                else:
                    print("No tools available")
                continue

            if not user_input:
                continue

            result = await agent.process_query_with_details(user_input)
            print(f"Assistant: {result.response}")

            # Show tool usage summary
            if result.tool_used and result.tools_called:
                unique_tools = list(set(result.tools_called))
                tool_counts = {
                    tool: result.tools_called.count(tool) for tool in unique_tools
                }

                print("\n🔧 Tool Usage Summary:")
                for tool, count in tool_counts.items():
                    print(f"  - {tool}: called {count} time(s)")

        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Unexpected error: {e}")

    print("Goodbye!")


if __name__ == "__main__":
    asyncio.run(main())