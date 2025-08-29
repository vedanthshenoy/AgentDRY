import asyncio
import logging
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

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

# Import client components
from gemini_client import ConnectionManager, GeminiClient, ConversationManager

# MCP Sampling imports
from fastmcp.client.sampling import SamplingMessage, SamplingParams, RequestContext

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

    async def create_tool_via_server(self, query: str) -> QueryResult:
        """Create a new tool using the server's create_tool function."""
        try:
            result = await self.execute_tool("create_tool", {"query": query})
            if result.success:
                logger.info(f"Tool creation request processed: {query}")
                # Wait a bit for server restart
                await asyncio.sleep(5)
                # Refresh tools cache to get the new tool
                await self.refresh_tools()
            return result
        except Exception as e:
            error_msg = f"Failed to create tool via server: {str(e)}"
            logger.error(error_msg)
            return QueryResult(success=False, response=error_msg, error=str(e))


class SamplingHandler:
    """Handles server-initiated sampling requests using Gemini."""
    
    def __init__(self, gemini_client: GeminiClient):
        self.gemini_client = gemini_client
        self.logger = logger
    
    async def handle_sampling(
        self,
        messages: List[SamplingMessage],
        params: SamplingParams,
        context: RequestContext
    ) -> str:
        """Handle sampling request from server."""
        try:
            # Extract conversation from messages
            conversation_parts = []
            for message in messages:
                role = message.role
                # Extract text content from message
                if hasattr(message.content, 'text'):
                    content = message.content.text
                elif isinstance(message.content, str):
                    content = message.content
                else:
                    content = str(message.content)
                
                conversation_parts.append(f"{role}: {content}")
            
            # Build the prompt
            system_prompt = params.systemPrompt or "You are a helpful assistant."
            conversation_text = "\n".join(conversation_parts)
            
            full_prompt = f"{system_prompt}\n\nConversation:\n{conversation_text}"
            
            # Use sampling parameters
            temperature = params.temperature if params.temperature is not None else 0.0
            
            self.logger.info(f"Processing sampling request with temperature: {temperature}")
            
            # Generate response using Gemini
            response = await self.gemini_client.generate_response(
                full_prompt, 
                temperature=temperature
            )
            
            return response
            
        except Exception as e:
            error_msg = f"Sampling handler error: {str(e)}"
            self.logger.error(error_msg)
            return "Error: Could not process sampling request"


class GeminiBridge:
    """Bridge class with intelligent tool calling and FastMCP sampling support."""

    def __init__(
        self,
        api_key: str,
        server_url: str = "http://localhost:8000/sse",
        max_tool_iterations: int = 5,
    ):
        self.llm = GeminiClient(api_key)
        self.sampling_handler = SamplingHandler(self.llm)
        
        # Create connection manager with sampling handler
        sampling_handler_func = self.sampling_handler.handle_sampling
        self.connection_manager = ConnectionManager(server_url, sampling_handler_func)
        
        self.tool_manager = MCPToolManager(self.connection_manager)
        self.conversation = ConversationManager()
        self.server_available = False
        self.max_tool_iterations = max_tool_iterations
        self.server_context = ""  # Store server context/domain

    async def initialize(self) -> bool:
        """Initialize the bridge with sampling support."""
        try:
            self.server_available = await self.connection_manager.check_server_health()
            if self.server_available:
                await self.tool_manager.refresh_tools()
                # Analyze server context based on available tools
                await self._analyze_server_context()
            return True
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            return False

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
            # Fallback if proto modules are not available
            if hasattr(proto_map, 'items'):
                return dict(proto_map.items())
            elif hasattr(proto_map, '__iter__') and not isinstance(proto_map, str):
                return list(proto_map)
            return proto_map

    def _create_content_objects(self, role: str, parts: List[Any]) -> Any:
        """Create content objects compatible with Gemini API."""
        try:
            # Try using the imported classes first
            return Content(role=role, parts=parts)
        except (NameError, AttributeError):
            # Fallback to dictionary format
            return {
                "role": role,
                "parts": parts
            }

    def _create_part_objects(self, text: str = None, function_call: Any = None, function_response: Any = None) -> Any:
        """Create part objects compatible with Gemini API."""
        try:
            # Try using the imported classes first
            if text:
                return Part(text=text)
            elif function_call:
                return Part(function_call=function_call)
            elif function_response:
                return Part(function_response=function_response)
        except (NameError, AttributeError):
            # Fallback to dictionary format
            if text:
                return {"text": text}
            elif function_call:
                return {"function_call": function_call}
            elif function_response:
                return {"function_response": function_response}

    async def _should_use_tools_or_create(self, query: str, available_tools: List[MCPTool]) -> str:
        """
        Let LLM decide whether to use existing tools, create new tool, or respond directly.
        Returns: 'use_tools', 'create_tool', or 'respond_directly'
        """
        if not available_tools:
            return "respond_directly"
            
        tools_context = "\n".join([f"- {tool.name}: {tool.description}" for tool in available_tools])
        
        decision_prompt = f"""
        You are an intelligent assistant with access to an MCP server that supports dynamic tool creation.

        Available tools:
        {tools_context}

        User query: "{query}"

        The server has a 'create_tool' function that can generate new tools using LLM sampling.

        Analyze the query and determine the best approach:

        1. If the query can be handled by existing tools, respond with: "use_tools"
        2. If the query requires functionality not available in existing tools AND could benefit from a dedicated function, respond with: "create_tool"  
        3. If the query is asking you to explicitly create a function/tool, respond with: "create_tool"
        4. If the query is a general question that doesn't need tools, respond with: "respond_directly"

        Note: The server can create mathematical, utility, and computational functions. It's better to create a tool for:
        - Calculations that could be reused
        - Complex operations that benefit from dedicated functions
        - User requests for specific functionality

        Respond with ONLY one of these three options: use_tools, create_tool, or respond_directly
        """
        
        try:
            decision = await self.llm.generate_response(decision_prompt, temperature=0.0)
            decision = decision.strip().lower()
            
            if decision in ["use_tools", "create_tool", "respond_directly"]:
                logger.info(f"LLM decision for query '{query}': {decision}")
                return decision
            else:
                logger.warning(f"Invalid LLM decision: {decision}, defaulting to respond_directly")
                return "respond_directly"
        except Exception as e:
            logger.error(f"Error in decision making: {e}")
            return "respond_directly"

    async def _iterative_tool_execution(
        self, query: str, tools: List[MCPTool]
    ) -> Optional[QueryResult]:
        """
        Executes tools iteratively with improved compatibility for different Gemini API versions.
        """
        if not tools or not self.server_available:
            return None

        tools_called = []
        
        try:
            # 1. Prepare tools for the Gemini API
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

            # 2. Start conversation history - use simpler approach
            chat_history = [
                self._create_content_objects("user", [self._create_part_objects(text=query)])
            ]

            # 3. Start the iterative tool calling loop
            for iteration in range(self.max_tool_iterations):
                logger.info(f"Tool iteration {iteration + 1}/{self.max_tool_iterations}")
                
                response = await asyncio.to_thread(
                    self.llm.model.generate_content,
                    chat_history,
                    tools={"function_declarations": tool_declarations},
                    generation_config={"temperature": 0.0},
                )
                
                model_response_content = response.candidates[0].content
                chat_history.append(model_response_content)

                # Check if model wants to call functions
                function_calls = []
                for part in model_response_content.parts:
                    if hasattr(part, 'function_call') and part.function_call:
                        function_calls.append(part)

                if not function_calls:
                    # Model is done, extract final response
                    break

                # 5. Execute function calls
                tasks = []
                for part in function_calls:
                    fc = part.function_call
                    tool_name = fc.name
                    arguments = self._convert_proto_map_to_dict(fc.args)
                    
                    logger.info(f"LLM calling tool: {tool_name} with args: {arguments}")
                    print(f"[TOOL INVOKE] {tool_name} with args: {arguments}")
                    tools_called.append(tool_name)
                    tasks.append(self.tool_manager.execute_tool(tool_name, arguments))
                
                results = await asyncio.gather(*tasks)

                # 6. Create function response parts
                function_response_parts = []
                for tool_result, part in zip(results, function_calls):
                    if tool_result.success:
                        for tool_result, part in zip(results, function_calls):
                            print(f"[TOOL RESULT] {part.function_call.name} -> Success={tool_result.success}, Response={tool_result.response}, Error={tool_result.error}")

                        output_data = tool_result.response
                    else:
                        output_data = f"Tool failed: {tool_result.error}"

                    # Create function response
                    try:
                        response_obj = FunctionResponse(
                            name=part.function_call.name,
                            response={"output": output_data},
                        )
                        function_response_parts.append(self._create_part_objects(function_response=response_obj))
                    except (NameError, AttributeError):
                        # Fallback to dictionary format
                        function_response_parts.append({
                            "function_response": {
                                "name": part.function_call.name,
                                "response": {"output": output_data}
                            }
                        })
                
                # Append the tool response turn
                chat_history.append(
                    self._create_content_objects("tool", function_response_parts)
                )

            # 7. Extract final text response
            final_content = chat_history[-1]
            final_text_response = ""
            
            if hasattr(final_content, 'parts'):
                parts = final_content.parts
            else:
                parts = final_content.get('parts', [])
            
            for part in parts:
                if hasattr(part, 'text') and part.text:
                    final_text_response += part.text
                elif isinstance(part, dict) and 'text' in part:
                    final_text_response += part['text']
            
            return QueryResult(
                success=True,
                response=final_text_response.strip() or "Task completed successfully.",
                tool_used=len(tools_called) > 0,
                tools_called=list(set(tools_called)),
            )

        except Exception as e:
            logger.error(f"Iterative tool execution failed: {e}", exc_info=True)
            return QueryResult(
                success=False,
                response=f"An error occurred during tool execution: {str(e)}",
                error=str(e),
                tools_called=list(set(tools_called)),
            )

    async def process_query(self, query: str) -> QueryResult:
        """Process a user query with intelligent decision making and server-side tool creation."""
        try:
            self.conversation.add_message("user", query)

            if not self.server_available:
                # No server available, respond directly
                response = await self.llm.generate_response(f"Please answer this query: {query}")
                self.conversation.add_message("assistant", response)
                return QueryResult(success=True, response=response)

            # Get available tools and let LLM decide what to do
            tools = await self.tool_manager.get_tools()
            decision = await self._should_use_tools_or_create(query, tools)
            
            if decision == "use_tools" and tools:
                # Use existing tools
                tool_result = await self._iterative_tool_execution(query, tools)
                if tool_result and tool_result.success:
                    if tool_result.tools_called:
                        logger.info(f"Tools used: {', '.join(tool_result.tools_called)}")
                    self.conversation.add_message("assistant", tool_result.response)
                    return tool_result
                else:
                    # Tool execution failed, fall back to direct response
                    response = await self.llm.generate_response(f"Please answer this query: {query}")
                    self.conversation.add_message("assistant", response)
                    return QueryResult(success=True, response=response)
                    
            elif decision == "create_tool":
                # Create a new tool using server's create_tool function
                try:
                    # Check if this is an explicit tool creation request
                    if any(phrase in query.lower() for phrase in ["create a function", "create a tool", "make a function"]):
                        # Just create the tool and exit
                        creation_result = await self.tool_manager.create_tool_via_server(query)
                        if creation_result.success:
                            response = f"Tool created successfully: {creation_result.response}"
                        else:
                            response = f"Tool creation failed: {creation_result.response}"
                        
                        self.conversation.add_message("assistant", response)
                        return QueryResult(success=creation_result.success, response=response, tool_used=False)
                    else:
                        # Create tool and then use it to answer the query
                        logger.info(f"Creating new tool for query: {query}")
                        creation_result = await self.tool_manager.create_tool_via_server(f"Create a function to handle: {query}")
                        
                        if not creation_result.success:
                            # Tool creation failed, respond directly
                            response = await self.llm.generate_response(f"I couldn't create a tool for this, but I can help: {query}")
                            self.conversation.add_message("assistant", response)
                            return QueryResult(success=True, response=response)

                        # Try using the new tool by re-executing the query with updated tools
                        logger.info("Tool created, waiting for server restart and trying to use new tools...")
                        await asyncio.sleep(8)  # Wait for server restart
                        
                        # Refresh tools and try again
                        updated_tools = await self.tool_manager.get_tools()
                        tool_result = await self._iterative_tool_execution(query, updated_tools)
                        
                        if tool_result and tool_result.success:
                            response = f"Created new tool and executed: {tool_result.response}"
                            if tool_result.tools_called:
                                response += f" (New tools used: {', '.join(tool_result.tools_called)})"
                            self.conversation.add_message("assistant", response)
                            return QueryResult(
                                success=True,
                                response=response,
                                tool_used=True,
                                tools_called=tool_result.tools_called,
                            )
                        else:
                            # Tool creation succeeded but execution failed, respond with creation status
                            response = f"Tool was created ({creation_result.response}) but I'll answer your question directly: {await self.llm.generate_response(query)}"
                            self.conversation.add_message("assistant", response)
                            return QueryResult(success=True, response=response)

                except Exception as e:
                    logger.error(f"Tool creation via server failed: {e}")
                    response = await self.llm.generate_response(f"Please answer this query: {query}")
                    self.conversation.add_message("assistant", response)
                    return QueryResult(success=True, response=response)
            
            else:  # decision == "respond_directly"
                # Generate direct response
                response = await self.llm.generate_response(f"Please answer this query: {query}")
                self.conversation.add_message("assistant", response)
                return QueryResult(success=True, response=response)

        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            logger.error(error_msg)
            return QueryResult(
                success=False, response="Sorry, I encountered an error.", error=str(e)
            )

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


# Enhanced CLI with server-side tool creation
async def main():
    """Enhanced CLI interface with server-side tool creation via sampling."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY not found")
        return

    bridge = GeminiBridge(api_key, max_tool_iterations=10)

    if not await bridge.initialize():
        print("Failed to initialize bridge")
        return

    print("🤖 Enhanced Bridge ready with FastMCP sampling support!")
    print(f"🔧 Detected server context: {bridge.server_context}")
    print("Type 'quit' to exit, 'tools' to list available tools, 'clear' to clear history")
    print("The server can now create new tools dynamically using client-side LLM sampling!")

    while True:
        try:
            user_input = input("\nYou: ").strip()

            if user_input.lower() == "quit":
                break
            elif user_input.lower() == "clear":
                bridge.clear_conversation()
                print("Conversation history cleared.")
                continue
            elif user_input.lower() == "tools":
                tools = await bridge.tool_manager.get_tools()
                if tools:
                    print(f"\nAvailable tools ({len(tools)}):")
                    for tool in tools:
                        print(f"  - {tool.name}: {tool.description}")
                else:
                    print("No tools available")
                continue

            if not user_input:
                continue

            result = await bridge.process_query_with_details(user_input)
            print(f"Assistant: {result.response}")

            # Show tool usage summary if tools were used
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