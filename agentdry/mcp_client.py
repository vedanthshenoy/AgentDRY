import asyncio
import json
import logging
import subprocess
import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime

import google.generativeai as genai

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class QueryResult:
    """Result of processing a query"""
    response: str
    tool_used: bool = False
    tool_name: Optional[str] = None
    tool_result: Optional[Any] = None
    sampling_used: bool = False

class MCPClient:
    """MCP Client that connects to multiple servers with sampling capabilities"""
    
    def __init__(self, gemini_api_key: str):
        self.gemini_api_key = gemini_api_key
        self.server_processes: Dict[str, subprocess.Popen] = {}
        self.tools: Dict[str, Dict[str, Any]] = {}
        self.conversation_history = []
        self.servers_running = False
        
        # Initialize Gemini
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Server configurations
        self.server_configs = {
            "agentdry": {
                "command": "C:/greendex/greendex/Scripts/python.exe",
                "args": ["C:/prass/agentdry/servers/claude_mcp_server.py"],
                "env": {
                    "PYTHONPATH": "C:/prass/agentdry;C:/greendex/greendex/Lib/site-packages",
                    "PATH": "C:/greendex/greendex/Scripts;C:/greendex/greendex/Scripts/Scripts",
                    "VIRTUAL_ENV": "C:/greendex/greendex",
                }
            },
            "math": {
                "command": "C:/greendex/greendex/Scripts/python.exe",
                "args": ["C:/prass/agentdry/servers/math_server_q.py"],
                "env": {
                    "PYTHONPATH": "C:/prass/agentdry;C:/greendex/greendx/Lib/site-packages",
                    "PATH": "C:/greendex/greendx/Scripts;C:/greendex/greendx/Scripts/Scripts",
                    "VIRTUAL_ENV": "C:/greendex/greendex",
                }
            },
            "seeker": {
                "command": "C:/greendex/greendex/Scripts/python.exe", 
                "args": ["C:/prass/agentdry/servers/seeker_server.py"],
                "env": {
                    "PYTHONPATH": "C:/prass/agentdry;C:/greendex/greendex/Lib/site-packages",
                    "PATH": "C:/greendex/greendex/Scripts;C:/greendx/greendx/Scripts/Scripts",
                    "VIRTUAL_ENV": "C:/greendex/greendx",
                }
            }
        }
        
        # Mock tools data since we can't connect to actual servers easily
        self.mock_tools = {
            "agentdry:create_dynamic_tool": {
                "name": "create_dynamic_tool", 
                "description": "Create a new tool dynamically based on a description",
                "server": "agentdry"
            },
            "agentdry:list_dynamic_tools": {
                "name": "list_dynamic_tools",
                "description": "List all dynamically created tools", 
                "server": "agentdry"
            },
            "agentdry:execute_python_code": {
                "name": "execute_python_code",
                "description": "Execute Python code safely",
                "server": "agentdry"
            },
            "math:add": {
                "name": "add",
                "description": "Add two numbers",
                "server": "math"
            },
            "math:subtract": {
                "name": "subtract", 
                "description": "Subtract b from a",
                "server": "math"
            },
            "math:multiply": {
                "name": "multiply",
                "description": "Multiply two numbers", 
                "server": "math"
            },
            "math:divide": {
                "name": "divide",
                "description": "Divide a by b",
                "server": "math"
            },
            "seeker:hey_agent_dry": {
                "name": "hey_agent_dry",
                "description": "Greet AgentDRY and discover MCP ecosystem",
                "server": "seeker"
            },
            "seeker:show_servers": {
                "name": "show_servers", 
                "description": "Show detailed information about discovered servers",
                "server": "seeker"
            },
            "seeker:refresh_discovery": {
                "name": "refresh_discovery",
                "description": "Manually refresh the MCP ecosystem discovery",
                "server": "seeker"
            }
        }
    
    async def initialize(self) -> bool:
        """Initialize connections to all MCP servers"""
        try:
            # For now, we'll simulate the connection and use the mock tools
            # In a real implementation, this would establish MCP connections
            logger.info("Simulating MCP server connections...")
            
            self.tools = self.mock_tools.copy()
            self.servers_running = True
            
            logger.info(f"Successfully initialized MCP client simulation")
            logger.info(f"Total tools available: {len(self.tools)}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize MCP client: {e}")
            return False
    
    async def sample(self, messages: List[Dict[str, str]], max_tokens: int = 1000, temperature: float = 0.7) -> Dict[str, Any]:
        """Implement sampling using Gemini API"""
        try:
            # Convert messages to Gemini format
            gemini_messages = []
            for msg in messages:
                if msg["role"] == "user":
                    gemini_messages.append(msg["content"])
                elif msg["role"] == "assistant":
                    # For multi-turn, we'd need to handle this properly
                    pass
            
            # Use the last user message
            prompt = gemini_messages[-1] if gemini_messages else ""
            
            # Configure generation
            generation_config = genai.types.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=temperature,
            )
            
            # Generate response
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.model.generate_content(
                    prompt,
                    generation_config=generation_config
                )
            )
            
            return {
                "choices": [{
                    "message": {
                        "content": response.text
                    }
                }]
            }
            
        except Exception as e:
            logger.error(f"Sampling failed: {e}")
            return {
                "choices": [{
                    "message": {
                        "content": f"Sampling error: {str(e)}"
                    }
                }]
            }
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool - simulated for now"""
        try:
            if tool_name not in self.tools:
                return {"error": f"Tool {tool_name} not found"}
            
            tool_info = self.tools[tool_name]
            server = tool_info["server"]
            actual_tool_name = tool_info["name"]
            
            # Simulate tool execution based on tool type
            if "math:" in tool_name:
                return await self._simulate_math_tool(actual_tool_name, arguments)
            elif "agentdry:" in tool_name:
                return await self._simulate_agentdry_tool(actual_tool_name, arguments)
            elif "seeker:" in tool_name:
                return await self._simulate_seeker_tool(actual_tool_name, arguments)
            else:
                return {"result": f"Simulated execution of {tool_name} with args: {arguments}"}
                
        except Exception as e:
            logger.error(f"Tool call failed: {e}")
            return {"error": str(e)}
    
    async def _simulate_math_tool(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate math tool execution"""
        try:
            a = float(args.get("a", 0))
            b = float(args.get("b", 0))
            
            if tool_name == "add":
                result = a + b
            elif tool_name == "subtract":
                result = a - b
            elif tool_name == "multiply":
                result = a * b
            elif tool_name == "divide":
                if b == 0:
                    return {"error": "Division by zero"}
                result = a / b
            else:
                return {"error": f"Unknown math operation: {tool_name}"}
            
            return {"result": result}
            
        except Exception as e:
            return {"error": f"Math calculation error: {str(e)}"}
    
    async def _simulate_agentdry_tool(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate agentdry tool execution"""
        if tool_name == "execute_python_code":
            code = args.get("code", "")
            try:
                # WARNING: This is unsafe - only for demonstration
                # In production, use a sandboxed environment
                result = eval(code) if code else "No code provided"
                return {"result": str(result)}
            except Exception as e:
                return {"error": f"Python execution error: {str(e)}"}
        
        elif tool_name == "create_dynamic_tool":
            description = args.get("query_description", "")
            tool_name_new = f"dynamic_tool_{len(self.tools)}"
            return {
                "result": f"Created dynamic tool '{tool_name_new}' for: {description}",
                "tool_name": tool_name_new
            }
        
        elif tool_name == "list_dynamic_tools":
            dynamic_tools = [k for k in self.tools.keys() if "dynamic_tool" in k]
            return {"result": dynamic_tools}
        
        return {"result": f"Simulated agentdry tool: {tool_name}"}
    
    async def _simulate_seeker_tool(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate seeker tool execution"""
        if tool_name == "hey_agent_dry":
            return {
                "greeting": "Hey there! I'm AgentDRY - Your MCP ecosystem mapper!",
                "servers_discovered": 3,
                "servers": ["agentdry", "math", "seeker"]
            }
        elif tool_name == "show_servers":
            return {
                "servers": [
                    {"name": "agentdry", "tools": 3, "status": "connected"},
                    {"name": "math", "tools": 4, "status": "connected"},
                    {"name": "seeker", "tools": 3, "status": "connected"}
                ]
            }
        elif tool_name == "refresh_discovery":
            return {"status": "discovery_refreshed", "servers": 3}
        
        return {"result": f"Simulated seeker tool: {tool_name}"}
    
    async def process_query(self, query: str) -> QueryResult:
        """Process a user query, potentially using tools and sampling"""
        try:
            self.conversation_history.append({"role": "user", "content": query})
            
            # First, let the AI decide if it needs to use tools
            tool_decision_prompt = f"""
            User query: {query}
            
            Available tools:
            {json.dumps([{"name": name, "description": tool["description"]} for name, tool in self.tools.items()], indent=2)}
            
            Should I use any tools to answer this query? If yes, which tool and with what arguments?
            
            Respond with JSON:
            {{
                "use_tool": true/false,
                "tool_name": "tool_name_if_needed",
                "arguments": {{"arg1": "value1"}} if tool needed,
                "reasoning": "why this tool is needed"
            }}
            """
            
            # Use sampling to decide on tool usage
            decision_response = await self.sample([
                {"role": "user", "content": tool_decision_prompt}
            ], max_tokens=500, temperature=0.1)
            
            decision_text = decision_response["choices"][0]["message"]["content"]
            
            # Try to extract JSON from decision
            import re
            json_match = re.search(r'\{.*\}', decision_text, re.DOTALL)
            
            tool_used = False
            tool_result = None
            tool_name = None
            
            if json_match:
                try:
                    decision = json.loads(json_match.group())
                    
                    if decision.get("use_tool", False):
                        tool_name = decision.get("tool_name")
                        arguments = decision.get("arguments", {})
                        
                        if tool_name and tool_name in self.tools:
                            logger.info(f"Using tool: {tool_name} with args: {arguments}")
                            tool_result = await self.call_tool(tool_name, arguments)
                            tool_used = True
                            
                except json.JSONDecodeError:
                    logger.warning("Could not parse tool decision JSON")
            
            # Generate final response
            if tool_used and tool_result:
                final_prompt = f"""
                User query: {query}
                Tool used: {tool_name}
                Tool result: {json.dumps(tool_result, indent=2)}
                
                Please provide a helpful response to the user based on the tool result.
                Be conversational and explain what you did.
                """
            else:
                final_prompt = f"""
                User query: {query}
                
                Please provide a helpful response. No tools were needed for this query.
                """
            
            # Use sampling for final response
            final_response = await self.sample([
                {"role": "user", "content": final_prompt}
            ], max_tokens=1000, temperature=0.7)
            
            response_text = final_response["choices"][0]["message"]["content"]
            
            # Add to conversation history
            self.conversation_history.append({"role": "assistant", "content": response_text})
            
            return QueryResult(
                response=response_text,
                tool_used=tool_used,
                tool_name=tool_name,
                tool_result=tool_result,
                sampling_used=True
            )
            
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            error_response = f"I encountered an error processing your query: {str(e)}"
            return QueryResult(response=error_response)
    
    async def get_available_tools(self) -> List[Dict[str, str]]:
        """Get list of available tools"""
        return [
            {
                "name": name,
                "description": tool["description"],
                "server": tool["server"]
            }
            for name, tool in self.tools.items()
        ]
    
    async def get_server_status(self) -> Dict[str, str]:
        """Get status of all connected servers"""
        if self.servers_running:
            return {
                "agentdry": "Connected",
                "math": "Connected", 
                "seeker": "Connected"
            }
        else:
            return {
                "agentdry": "Disconnected",
                "math": "Disconnected",
                "seeker": "Disconnected"
            }
    
    async def close(self):
        """Close all connections"""
        self.servers_running = False
        logger.info("MCP client closed")
    
    def clear_conversation(self):
        """Clear conversation history"""
        self.conversation_history = []