import json
import yaml
import os
import asyncio
from typing import Dict, Any
from datetime import datetime

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp import Context

# Initialize the FastMCP server
mcp = FastMCP("Seeker Server")

# Global context map and YAML file path
_context_map: Dict[str, Any] = {}
# _yaml_file_path = r"C:\prass\agentdry\servers\logs\mcp_servers_discovered.yaml"
_yaml_file_path = os.path.join(os.path.expanduser("~"), "mcp_servers_discovered.yaml")
_discovery_completed = False  # Fixed variable name

async def _discover_and_save_servers(ctx: Context) -> None:
    """Discover MCP servers using LLM Sampling and save to YAML file."""
    global _context_map, _discovery_completed  # Fixed variable name
    try:
        # Use LLM sampling to discover connected MCP servers
        discovery_prompt = """
        Please provide a comprehensive list of all MCP servers currently connected to this client.
        For each server, include:
        - Server name
        - Description of what it does
        - List of available tools/functions
        - Primary domain/category
        
        Format your response as valid JSON:
        {
            "servers": [
                {
                    "name": "server-name",
                    "description": "detailed description of server capabilities",
                    "tools": ["tool1", "tool2", "tool3"],
                    "primary_domain": "category"
                }
            ],
            "discovery_timestamp": "current_time",
            "total_count": "number"
        }
        
        Be thorough and include ALL connected servers with their complete tool lists.
        """
        
        # Sample the LLM for server discovery
        response = await ctx.sample(
            messages=[{"role": "user", "content": discovery_prompt}],
            max_tokens=2000,
            temperature=0.1
        )
        
        response_text = response.choices[0].message.content
        
        # Extract JSON from response
        import re
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            discovered_data = json.loads(json_match.group())
        else:
            # Fallback if JSON parsing fails
            discovered_data = {
                "servers": [],
                "discovery_timestamp": datetime.now().isoformat(),
                "total_count": 0,
                "error": "Could not parse server discovery response"
            }
        
        # Enhance each server with detailed capabilities using LLM Sampling
        for server in discovered_data.get("servers", []):
            if server.get("name"):
                capabilities_prompt = f"""
                Provide detailed information about the MCP server named "{server['name']}"
                that has these tools: {server.get('tools', [])}.
                
                Include:
                1. Detailed description of what this server does
                2. Full capabilities and features
                3. What problems it solves
                4. When someone would use it
                5. Tool description if possible
                
                Format as JSON:
                {{
                    "detailed_description": "comprehensive description",
                    "capabilities": ["capability1", "capability2"],
                    "use_cases": ["use_case1", "use_case2"],
                    "tool_descriptions": {{"tool_name": "What it does"}}
                }}
                """
                
                try:
                    cap_response = await ctx.sample(
                        messages=[{"role": "user", "content": capabilities_prompt}],  # Fixed "context" to "content"
                        max_tokens=500,
                        temperature=0.2
                    )
                    
                    cap_text = cap_response.choices[0].message.content  # Fixed "choice" to "choices"
                    cap_json_match = re.search(r'\{.*\}', cap_text, re.DOTALL)
                    
                    if cap_json_match:
                        capabilities_data = json.loads(cap_json_match.group())
                        server.update(capabilities_data)
                    else:
                        server["detailed_description"] = f"Enhanced description for {server['name']} server"
                        server["capabilities"] = ["Server capabilities analysis"]
                        
                except Exception as e:
                    server["detailed_description"] = f"Server: {server.get('name', 'Unknown')} - Analysis failed"
                    server["error"] = str(e)[:100]  # Truncate error message
                    
        # Add metadata
        discovered_data["discovery_timestamp"] = datetime.now().isoformat()
        discovered_data["discovery_method"] = "llm_sampling_with_capabilities"
        discovered_data["auto_discovery"] = True
        
        # Update global context map
        _context_map = discovered_data
        
        # Save to YAML file with UTF-8 encoding to handle any Unicode
        with open(_yaml_file_path, "w", encoding='utf-8') as f:
            yaml.dump(discovered_data, f, default_flow_style=False, allow_unicode=True, indent=2)
            
        # Also create a simple text log
        # with open(r"C:\prass\agentdry\servers\logs\discovery_log.txt", "w", encoding='utf-8') as f:
        log_path = os.path.join(os.path.expanduser("~"), "discovery_log.txt")
        with open(log_path, "w", encoding='utf-8') as f:
            f.write(f"AUTO-Discovery completed at {datetime.now().isoformat()}\n")
            f.write(f"Found {len(discovered_data.get('servers', []))} servers\n")
            f.write(f"Saved to {_yaml_file_path}\n")
            f.write(f"Discovery triggered: Server startup\n")
            
        _discovery_completed = True
        
    except Exception as e:
        # Error handling - create minimal YAML file
        error_data = {
            "servers": [],
            "discovery_timestamp": datetime.now().isoformat(),
            "total_count": 0,
            "error": str(e).encode('ascii', 'ignore').decode('ascii'),
            "discovery_method": "llm_sampling_failed",
            "auto_discovery": True
        } 
        
        _context_map = error_data
        
        # Save error data to YAML
        with open(_yaml_file_path, "w", encoding='utf-8') as f:
            yaml.dump(error_data, f, default_flow_style=False, allow_unicode=True, indent=2)
            
        # Create error log
        # with open("discovery_error.txt", 'w', encoding='utf-8') as f:
        error_log_path = os.path.join(os.path.expanduser("~"), "discovery_error.txt")
        with open(error_log_path, 'w', encoding='utf-8') as f:
            f.write(f"AUTO-Discovery failed at {datetime.now().isoformat()}\n")
            f.write(f"Error: {str(e).encode('ascii', 'ignore').decode('ascii')}\n")
            
        _discovery_completed = True
        
@mcp.resource("context://map")
async def get_context_map() -> str:  # Removed ctx parameter
    """Get the current MCP ecosystem context map"""
    if os.path.exists(_yaml_file_path):
        with open(_yaml_file_path, 'r', encoding='utf-8') as f:
            return f.read()
    return yaml.dump(_context_map, default_flow_style=False, allow_unicode=True, indent=2)
    
@mcp.tool()
async def show_servers(ctx: Context) -> Dict[str, Any]:
    """Show detailed information about discovered servers"""
    if not _context_map.get("servers"):
        return {"message": "No servers discovered yet. Try saying 'Hey AgentDRY' first!"}
    
    return {
        "servers": _context_map.get("servers", []),
        "total_count": len(_context_map.get("servers", [])),
        "discovery_timestamp": _context_map.get("discovery_timestamp"),
        "yaml_file": _yaml_file_path
    }
    
@mcp.tool()
async def refresh_discovery(ctx: Context) -> Dict[str, Any]:
    """Manually refresh the MCP ecosystem discovery."""
    global _discovery_completed
    _discovery_completed = False
    await _discover_and_save_servers(ctx)
    return {
        "status": "discovery_refreshed",
        "servers": len(_context_map.get("servers", [])),
        "yaml_file": _yaml_file_path,
        "timestamp": _context_map.get("discovery_timestamp")
    }
    
@mcp.tool()
async def hey_agent_dry(ctx: Context) -> Dict[str, Any]:
    """Responds to casual greetings and automatically discovers MCP ecosystem."""
    # Automatically run discovery if not completed
    if not _discovery_completed:
        await _discover_and_save_servers(ctx)
        
    greeting_response = f"""
    Hey there! I'm AgentDRY - Your MCP ecosystem mapper!
    
    I've automatically discovered {len(_context_map.get('servers', []))} MCP servers connected to this client.
    
    Files Created:
    - {_yaml_file_path} - Complete server inventory
    - discovery_log.txt - Discovery timestamp log
    
    What I found:
    """
    
    # Add server summary
    for server in _context_map.get("servers", []):
        server_name = server.get("name", "Unknown")
        tool_count = len(server.get("tools", []))
        domain = server.get("primary_domain", "general")
        greeting_response += f"\n   - {server_name} ({tool_count} tools) - {domain}"
        
    if not _context_map.get("servers", []):
        greeting_response += f"\n   No additional servers found (just me!)"
        
    greeting_response += f"""
    
    Available commands:
    - "refresh discovery" - Update the server list
    - "show servers" - View detailed server info
    - Check the YAML file for complete details!
    
    Discovery completed at: {_context_map.get('discovery_timestamp', 'Unknown')}
    """
    
    return {
        "greeting": greeting_response,
        "servers_discovered": len(_context_map.get("servers", [])),
        "yaml_file": _yaml_file_path,
        "auto_discovery_triggered": not _discovery_completed
    }
    
if __name__ == "__main__":
    mcp.run()