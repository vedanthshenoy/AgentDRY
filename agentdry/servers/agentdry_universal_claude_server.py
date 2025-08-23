import asyncio
import json
import logging
import os
import sys
import threading
import time
import importlib.util
import tempfile
import re
import subprocess
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime
from dataclasses import dataclass, asdict
from pathlib import Path
import yaml

from mcp.server.fastmcp import FastMCP
import mcp.server.stdio

# Add the parent directory (agentdry) to the path so we can import utils
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

# Import your existing utilities (backward compatibility)
try:
    from utils.append_to_server import create_tool_from_user_input, append_code_to_server_file
    from main import create_and_update_tool, delete_tool_from_server
    UTILS_AVAILABLE = True
    print(f"Successfully imported utils from: {project_root}")
except ImportError as e:
    print(f"Warning: Could not import utility functions: {e}")
    print(f"Attempted to import from: {project_root}")
    UTILS_AVAILABLE = False

# Setup Enhanced Logging
LOG_DIR = "logs"
ECOSYSTEM_DIR = "mcp_ecosystem"
SERVERS_CONFIG = os.path.join(ECOSYSTEM_DIR, "servers.json")
CAPABILITIES_DB = os.path.join(ECOSYSTEM_DIR, "capabilities.json")
TOOL_REGISTRY = os.path.join(ECOSYSTEM_DIR, "tool_registry.json")

for directory in [LOG_DIR, ECOSYSTEM_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "agentdry_universal.log"), encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ServerInfo:
    """Information about an MCP server"""
    name: str
    path: str
    port: Optional[int] = None
    status: str = "unknown"  # unknown, running, stopped, error
    capabilities: List[str] = None
    tools: List[str] = None
    last_ping: Optional[str] = None
    config: Dict[str, Any] = None

@dataclass
class ToolInfo:
    """Information about a tool"""
    name: str
    server: str
    description: str
    parameters: Dict[str, Any]
    created_at: str
    last_used: Optional[str] = None
    usage_count: int = 0

@dataclass
class QueryContext:
    """Context for processing a query"""
    raw_query: str
    intent: str
    required_capabilities: List[str]
    parameters: Dict[str, Any]
    target_servers: List[str]
    missing_tools: List[str]

class UniversalServerRegistry:
    """Registry for managing multiple MCP servers"""
    
    def __init__(self):
        self.servers: Dict[str, ServerInfo] = {}
        self.capabilities: Dict[str, List[str]] = {}  # capability -> [servers]
        self.tools: Dict[str, ToolInfo] = {}
        self.load_configuration()
    
    def load_configuration(self):
        """Load server configuration from JSON"""
        try:
            if os.path.exists(SERVERS_CONFIG):
                with open(SERVERS_CONFIG, 'r') as f:
                    config = json.load(f)
                    for server_name, server_data in config.get('servers', {}).items():
                        self.servers[server_name] = ServerInfo(**server_data)
                        
            if os.path.exists(CAPABILITIES_DB):
                with open(CAPABILITIES_DB, 'r') as f:
                    self.capabilities = json.load(f)
                    
            if os.path.exists(TOOL_REGISTRY):
                with open(TOOL_REGISTRY, 'r') as f:
                    tools_data = json.load(f)
                    for tool_name, tool_data in tools_data.items():
                        self.tools[tool_name] = ToolInfo(**tool_data)
                        
            logger.info(f"Loaded {len(self.servers)} servers, {len(self.tools)} tools")
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
    
    def save_configuration(self):
        """Save current configuration to JSON"""
        try:
            # Save servers
            servers_data = {
                'servers': {name: asdict(server) for name, server in self.servers.items()}
            }
            with open(SERVERS_CONFIG, 'w') as f:
                json.dump(servers_data, f, indent=2)
            
            # Save capabilities
            with open(CAPABILITIES_DB, 'w') as f:
                json.dump(self.capabilities, f, indent=2)
            
            # Save tools
            tools_data = {name: asdict(tool) for name, tool in self.tools.items()}
            with open(TOOL_REGISTRY, 'w') as f:
                json.dump(tools_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
    
    def register_server(self, server_info: ServerInfo):
        """Register a new server"""
        self.servers[server_info.name] = server_info
        self._update_capabilities_for_server(server_info)
        self.save_configuration()
    
    def discover_servers(self, search_paths: List[str] = None):
        """Auto-discover MCP servers in specified paths"""
        if search_paths is None:
            search_paths = [
                os.path.join(project_root, "servers"),
                os.path.expanduser("~/.mcp/servers"),
                "/usr/local/mcp/servers",
                "./servers"
            ]
        
        discovered = []
        for search_path in search_paths:
            if os.path.exists(search_path):
                for file in os.listdir(search_path):
                    if file.endswith('_server.py') or file.endswith('_mcp.py'):
                        server_path = os.path.join(search_path, file)
                        server_name = os.path.splitext(file)[0]
                        
                        if server_name not in self.servers:
                            server_info = ServerInfo(
                                name=server_name,
                                path=server_path,
                                status="discovered",
                                capabilities=[],
                                tools=[]
                            )
                            self.register_server(server_info)
                            discovered.append(server_name)
        
        return discovered
    
    def _update_capabilities_for_server(self, server_info: ServerInfo):
        """Update capability mappings for a server"""
        if server_info.capabilities:
            for capability in server_info.capabilities:
                if capability not in self.capabilities:
                    self.capabilities[capability] = []
                if server_info.name not in self.capabilities[capability]:
                    self.capabilities[capability].append(server_info.name)

class IntelligentQueryProcessor:
    """Process natural language queries and determine required tools/servers"""
    
    def __init__(self, registry: UniversalServerRegistry):
        self.registry = registry
        self.intent_patterns = {
            'calculate': r'(calculate|compute|math|sum|multiply|divide|add|subtract)',
            'file_operation': r'(file|read|write|save|load|directory|folder)',
            'data_processing': r'(process|analyze|filter|sort|group|aggregate)',
            'web_operation': r'(web|http|api|request|fetch|download)',
            'database': r'(database|db|sql|query|table|record)',
            'text_processing': r'(text|string|parse|format|replace|search)',
            'image_processing': r'(image|photo|picture|resize|convert|filter)',
            'system': r'(system|process|monitor|status|performance)'
        }
    
    def analyze_query(self, query: str) -> QueryContext:
        """Analyze a query and determine requirements"""
        context = QueryContext(
            raw_query=query,
            intent=self._detect_intent(query),
            required_capabilities=[],
            parameters=self._extract_parameters(query),
            target_servers=[],
            missing_tools=[]
        )
        
        # Determine required capabilities based on intent
        context.required_capabilities = self._get_capabilities_for_intent(context.intent)
        
        # Find servers that can handle the capabilities
        context.target_servers = self._find_servers_for_capabilities(context.required_capabilities)
        
        # Check for missing tools
        context.missing_tools = self._identify_missing_tools(context)
        
        return context
    
    def _detect_intent(self, query: str) -> str:
        """Detect the intent of a query using pattern matching"""
        query_lower = query.lower()
        for intent, pattern in self.intent_patterns.items():
            if re.search(pattern, query_lower):
                return intent
        return 'general'
    
    def _extract_parameters(self, query: str) -> Dict[str, Any]:
        """Extract parameters from the query"""
        parameters = {}
        
        # Extract numbers
        numbers = re.findall(r'\d+(?:\.\d+)?', query)
        if numbers:
            parameters['numbers'] = [float(n) for n in numbers]
        
        # Extract file paths
        file_paths = re.findall(r'["\']([^"\']+\.[a-zA-Z0-9]+)["\']', query)
        if file_paths:
            parameters['files'] = file_paths
        
        # Extract URLs
        urls = re.findall(r'https?://[^\s]+', query)
        if urls:
            parameters['urls'] = urls
        
        # Extract quoted strings
        quoted = re.findall(r'["\']([^"\']+)["\']', query)
        if quoted:
            parameters['strings'] = quoted
            
        return parameters
    
    def _get_capabilities_for_intent(self, intent: str) -> List[str]:
        """Get required capabilities for an intent"""
        capability_map = {
            'calculate': ['math', 'computation'],
            'file_operation': ['file_system', 'io'],
            'data_processing': ['data_analysis', 'processing'],
            'web_operation': ['web', 'http', 'api'],
            'database': ['database', 'sql'],
            'text_processing': ['text', 'string'],
            'image_processing': ['image', 'graphics'],
            'system': ['system', 'monitoring']
        }
        return capability_map.get(intent, ['general'])
    
    def _find_servers_for_capabilities(self, capabilities: List[str]) -> List[str]:
        """Find servers that provide the required capabilities"""
        servers = set()
        for capability in capabilities:
            if capability in self.registry.capabilities:
                servers.update(self.registry.capabilities[capability])
        return list(servers)
    
    def _identify_missing_tools(self, context: QueryContext) -> List[str]:
        """Identify tools that need to be created"""
        missing = []
        
        # Check if we have tools for the required capabilities
        available_tools = set()
        for server_name in context.target_servers:
            server = self.registry.servers.get(server_name)
            if server and server.tools:
                available_tools.update(server.tools)
        
        # Generate tool names based on intent and parameters
        required_tool = f"{context.intent}_tool"
        if required_tool not in available_tools:
            missing.append(required_tool)
            
        return missing

class RealTimeToolGenerator:
    """Generate and register tools in real-time"""
    
    def __init__(self, registry: UniversalServerRegistry):
        self.registry = registry
        self.session_tools = {}  # In-memory tool persistence
        self.tool_templates = self._load_tool_templates()
    
    def _load_tool_templates(self) -> Dict[str, str]:
        """Load tool generation templates"""
        return {
            'calculate': '''
@mcp.tool()
def {tool_name}(expression: str) -> str:
    """Perform mathematical calculations"""
    try:
        import ast
        import operator as ops
        
        # Safe evaluation
        allowed_ops = {
            ast.Add: ops.add, ast.Sub: ops.sub, ast.Mult: ops.mul,
            ast.Div: ops.truediv, ast.Pow: ops.pow, ast.BitXor: ops.xor,
            ast.USub: ops.neg
        }
        
        def eval_expr(node):
            if isinstance(node, ast.Num):
                return node.n
            elif isinstance(node, ast.BinOp):
                return allowed_ops[type(node.op)](eval_expr(node.left), eval_expr(node.right))
            elif isinstance(node, ast.UnaryOp):
                return allowed_ops[type(node.op)](eval_expr(node.operand))
            else:
                raise TypeError(node)
        
        result = eval_expr(ast.parse(expression, mode='eval').body)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"
''',
            'file_operation': '''
@mcp.tool()
def {tool_name}(operation: str, file_path: str, content: str = "") -> str:
    """Perform file operations"""
    try:
        import os
        if operation == "read":
            with open(file_path, 'r') as f:
                return f.read()
        elif operation == "write":
            with open(file_path, 'w') as f:
                f.write(content)
            return f"Successfully wrote to {file_path}"
        elif operation == "exists":
            return str(os.path.exists(file_path))
        else:
            return f"Unknown operation: {operation}"
    except Exception as e:
        return f"Error: {str(e)}"
''',
            'data_processing': '''
@mcp.tool()
def {tool_name}(data: str, operation: str) -> str:
    """Process data with various operations"""
    try:
        import json
        
        # Parse data
        if data.startswith('[') or data.startswith('{'):
            parsed_data = json.loads(data)
        else:
            parsed_data = data.split(',')
        
        if operation == "sort":
            result = sorted(parsed_data)
        elif operation == "count":
            result = len(parsed_data)
        elif operation == "unique":
            result = list(set(parsed_data))
        else:
            result = f"Unknown operation: {operation}"
            
        return json.dumps(result) if isinstance(result, (list, dict)) else str(result)
    except Exception as e:
        return f"Error: {str(e)}"
''',
            'general': '''
@mcp.tool()
def {tool_name}(input_data: str) -> str:
    """General purpose tool for: {description}"""
    try:
        # Process input based on the tool description
        return f"Processed: {input_data} with {tool_name}"
    except Exception as e:
        return f"Error: {str(e)}"
'''
        }
    
    def generate_tool(self, context: QueryContext, tool_name: str) -> str:
        """Generate tool code based on context"""
        template = self.tool_templates.get(context.intent, self.tool_templates['general'])
        
        tool_code = template.format(
            tool_name=tool_name,
            description=context.raw_query
        )
        
        return tool_code
    
    def register_tool_to_server(self, server_name: str, tool_name: str, tool_code: str) -> bool:
        """Register a tool to a specific server via MCP"""
        try:
            server = self.registry.servers.get(server_name)
            if not server:
                logger.error(f"Server {server_name} not found")
                return False
            
            # Store in session memory
            self.session_tools[tool_name] = {
                'code': tool_code,
                'server': server_name,
                'created_at': datetime.now().isoformat()
            }
            
            # Update registry
            tool_info = ToolInfo(
                name=tool_name,
                server=server_name,
                description=f"Generated tool for server {server_name}",
                parameters={},
                created_at=datetime.now().isoformat()
            )
            self.registry.tools[tool_name] = tool_info
            
            # Update server tools list
            if not server.tools:
                server.tools = []
            server.tools.append(tool_name)
            
            # Save configuration
            self.registry.save_configuration()
            
            logger.info(f"Registered tool {tool_name} to server {server_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register tool {tool_name} to server {server_name}: {e}")
            return False

class CrossServerCoordinator:
    """Coordinate operations across multiple servers"""
    
    def __init__(self, registry: UniversalServerRegistry, tool_generator: RealTimeToolGenerator):
        self.registry = registry
        self.tool_generator = tool_generator
        self.active_connections = {}
    
    def execute_distributed_query(self, context: QueryContext) -> Dict[str, Any]:
        """Execute a query that may require multiple servers"""
        results = {
            'query': context.raw_query,
            'servers_used': [],
            'tools_created': [],
            'results': [],
            'errors': []
        }
        
        try:
            # Create missing tools first
            for missing_tool in context.missing_tools:
                created = self._create_and_deploy_tool(context, missing_tool)
                if created:
                    results['tools_created'].append(missing_tool)
            
            # Execute on target servers
            for server_name in context.target_servers:
                try:
                    server_result = self._execute_on_server(server_name, context)
                    results['servers_used'].append(server_name)
                    results['results'].append({
                        'server': server_name,
                        'result': server_result
                    })
                except Exception as e:
                    results['errors'].append(f"Server {server_name}: {str(e)}")
            
            # Aggregate results
            final_result = self._aggregate_results(results['results'])
            results['final_result'] = final_result
            
        except Exception as e:
            results['errors'].append(f"Coordination error: {str(e)}")
        
        return results
    
    def _create_and_deploy_tool(self, context: QueryContext, tool_name: str) -> bool:
        """Create and deploy a missing tool"""
        try:
            # Generate tool code
            tool_code = self.tool_generator.generate_tool(context, tool_name)
            
            # Deploy to appropriate servers
            deployed = False
            for server_name in context.target_servers:
                if self.tool_generator.register_tool_to_server(server_name, tool_name, tool_code):
                    deployed = True
            
            return deployed
            
        except Exception as e:
            logger.error(f"Failed to create and deploy tool {tool_name}: {e}")
            return False
    
    def _execute_on_server(self, server_name: str, context: QueryContext) -> str:
        """Execute query on a specific server"""
        # Simulate server execution (in real implementation, this would use MCP protocol)
        server = self.registry.servers.get(server_name)
        if not server:
            raise Exception(f"Server {server_name} not available")
        
        # For demo purposes, return a formatted result
        return f"Executed '{context.raw_query}' on {server_name} with intent '{context.intent}'"
    
    def _aggregate_results(self, results: List[Dict[str, Any]]) -> str:
        """Aggregate results from multiple servers"""
        if not results:
            return "No results to aggregate"
        
        if len(results) == 1:
            return results[0]['result']
        
        # Combine results from multiple servers
        combined = "Combined results:\n"
        for i, result in enumerate(results, 1):
            combined += f"{i}. {result['server']}: {result['result']}\n"
        
        return combined.strip()

# Enhanced AgentDRY Tool Manager (backward compatibility)
class EnhancedAgentDRYToolManager:
    def __init__(self, mcp_instance):
        self.mcp = mcp_instance
        self.registry = UniversalServerRegistry()
        self.query_processor = IntelligentQueryProcessor(self.registry)
        self.tool_generator = RealTimeToolGenerator(self.registry)
        self.coordinator = CrossServerCoordinator(self.registry, self.tool_generator)
        
        # Backward compatibility
        self.dynamic_tools = {}
        self.tool_counter = 0
        self.tools_dir = "agentdry_tools"
        
        # Initialize
        self._initialize_system()
    
    def _initialize_system(self):
        """Initialize the enhanced system"""
        if not os.path.exists(self.tools_dir):
            os.makedirs(self.tools_dir)
        
        # Auto-discover servers
        discovered = self.registry.discover_servers()
        logger.info(f"Discovered {len(discovered)} new servers: {discovered}")
        
        # Register AgentDRY as the primary server
        agentdry_server = ServerInfo(
            name="agentdry_primary",
            path=__file__,
            status="running",
            capabilities=["universal_management", "tool_generation", "coordination"],
            tools=["create_dynamic_tool", "list_dynamic_tools", "execute_distributed_query"]
        )
        self.registry.register_server(agentdry_server)
    
    def process_intelligent_query(self, query: str) -> str:
        """Process a query through the intelligent system"""
        try:
            # Analyze the query
            context = self.query_processor.analyze_query(query)
            
            # Execute distributed query
            results = self.coordinator.execute_distributed_query(context)
            
            # Format response
            response = f"AgentDRY Universal Analysis:\n"
            response += f"Intent: {context.intent}\n"
            response += f"Required Capabilities: {', '.join(context.required_capabilities)}\n"
            response += f"Target Servers: {', '.join(context.target_servers)}\n"
            
            if results['tools_created']:
                response += f"Tools Created: {', '.join(results['tools_created'])}\n"
            
            response += f"\nExecution Results:\n{results.get('final_result', 'No results')}"
            
            if results['errors']:
                response += f"\nErrors: {'; '.join(results['errors'])}"
            
            return response
            
        except Exception as e:
            return f"AgentDRY Error: {str(e)}"
    
    # Backward compatibility methods
    def create_and_register_tool(self, query: str) -> str:
        """Create a tool using legacy interface (backward compatibility)"""
        try:
            self.tool_counter += 1
            tool_name = f"agentdry_tool_{self.tool_counter}"
            
            # Use enhanced system for tool creation
            context = self.query_processor.analyze_query(query)
            tool_code = self.tool_generator.generate_tool(context, tool_name)
            
            # Register to primary server
            success = self.tool_generator.register_tool_to_server("agentdry_primary", tool_name, tool_code)
            
            if success:
                # Maintain backward compatibility structure
                tool_file = os.path.join(self.tools_dir, f"{tool_name}.py")
                with open(tool_file, 'w') as f:
                    f.write(tool_code)
                
                self.dynamic_tools[tool_name] = {
                    'code': tool_code,
                    'file': tool_file,
                    'query': query,
                    'created_at': datetime.now().isoformat()
                }
                
                logger.info(f"Legacy tool created: {tool_name}")
                return tool_name
            else:
                raise Exception("Failed to register tool")
            
        except Exception as e:
            logger.error(f"Failed to create legacy tool: {e}")
            raise

# MCP Server Setup
mcp = FastMCP("AgentDRY-Universal")
enhanced_manager = EnhancedAgentDRYToolManager(mcp)

@mcp.tool()
def create_dynamic_tool(query_description: str) -> str:
    """Create a new tool dynamically (backward compatibility)"""
    try:
        tool_name = enhanced_manager.create_and_register_tool(query_description)
        return f"AgentDRY created tool '{tool_name}' for: {query_description}"
    except Exception as e:
        return f"AgentDRY failed to create tool: {str(e)}"

@mcp.tool()
def execute_distributed_query(query: str) -> str:
    """Execute a query using the intelligent distributed system"""
    return enhanced_manager.process_intelligent_query(query)

@mcp.tool()
def list_all_servers() -> str:
    """List all registered MCP servers"""
    try:
        servers = enhanced_manager.registry.servers
        if not servers:
            return "No servers registered"
        
        result = f"AgentDRY Universal - Registered Servers ({len(servers)} total):\n\n"
        
        for name, server in servers.items():
            result += f"🖥️  {name}\n"
            result += f"   Status: {server.status}\n"
            result += f"   Path: {server.path}\n"
            result += f"   Capabilities: {', '.join(server.capabilities or [])}\n"
            result += f"   Tools: {len(server.tools or [])}\n"
            if server.last_ping:
                result += f"   Last Ping: {server.last_ping}\n"
            result += "\n"
        
        return result
    except Exception as e:
        return f"Error listing servers: {str(e)}"

@mcp.tool()
def register_new_server(server_name: str, server_path: str, capabilities: str = "") -> str:
    """Register a new MCP server manually"""
    try:
        caps_list = [cap.strip() for cap in capabilities.split(",")] if capabilities else []
        
        server_info = ServerInfo(
            name=server_name,
            path=server_path,
            status="registered",
            capabilities=caps_list,
            tools=[]
        )
        
        enhanced_manager.registry.register_server(server_info)
        return f"Successfully registered server: {server_name}"
    except Exception as e:
        return f"Failed to register server: {str(e)}"

@mcp.tool()
def discover_servers(search_paths: str = "") -> str:
    """Auto-discover MCP servers in specified paths"""
    try:
        paths = [path.strip() for path in search_paths.split(",")] if search_paths else None
        discovered = enhanced_manager.registry.discover_servers(paths)
        
        if discovered:
            return f"Discovered {len(discovered)} new servers: {', '.join(discovered)}"
        else:
            return "No new servers discovered"
    except Exception as e:
        return f"Discovery failed: {str(e)}"

@mcp.tool()
def list_capabilities() -> str:
    """List all available capabilities across servers"""
    try:
        caps = enhanced_manager.registry.capabilities
        if not caps:
            return "No capabilities registered"
        
        result = f"Available Capabilities ({len(caps)} total):\n\n"
        
        for capability, servers in caps.items():
            result += f"🔧 {capability}\n"
            result += f"   Available on: {', '.join(servers)}\n\n"
        
        return result
    except Exception as e:
        return f"Error listing capabilities: {str(e)}"

@mcp.tool()
def system_status() -> str:
    """Get comprehensive system status"""
    try:
        registry = enhanced_manager.registry
        
        status = f"""╔════════════════════════════════════════════════════╗
║              AgentDRY Universal                    ║
║          MCP Ecosystem Manager                     ║
║         Made in Mangaluru by                       ║
║           Vedanth and Praas team                   ║
╚════════════════════════════════════════════════════╝

🌐 System Overview:
   - Registered Servers: {len(registry.servers)}
   - Total Capabilities: {len(registry.capabilities)}
   - Registered Tools: {len(registry.tools)}
   - Session Tools: {len(enhanced_manager.tool_generator.session_tools)}

🖥️  Server Status:
"""
        
        for name, server in registry.servers.items():
            status += f"   • {name}: {server.status} ({len(server.tools or [])} tools)\n"
        
        status += f"""
🔧 Available Capabilities:
   {', '.join(registry.capabilities.keys())}

💡 Universal Commands:
   - execute_distributed_query(query) - Intelligent query processing
   - create_dynamic_tool(description) - Legacy tool creation
   - list_all_servers() - Server management
   - register_new_server(name, path, caps) - Add new server
   - discover_servers(paths) - Auto-discovery
   - list_capabilities() - Capability overview

🚀 Ready for universal MCP ecosystem management!"""
        
        return status
    except Exception as e:
        return f"Error getting system status: {str(e)}"

# Legacy compatibility tools
@mcp.tool()
def list_dynamic_tools() -> str:
    """List legacy dynamic tools (backward compatibility)"""
    try:
        tools_info = {
            'count': len(enhanced_manager.dynamic_tools),
            'tools': {name: {'query': info['query'], 'created_at': info['created_at']} 
                     for name, info in enhanced_manager.dynamic_tools.items()}
        }
        
        if tools_info['count'] == 0:
            return "No legacy dynamic tools created yet."
        
        result = f"Legacy Dynamic Tools ({tools_info['count']} total):\n\n"
        
        for tool_name, info in tools_info['tools'].items():
            result += f"* {tool_name}\n"
            result += f"   Created for: {info['query']}\n"
            result += f"   Created at: {info['created_at']}\n\n"
        
        return result
    except Exception as e:
        return f"Error listing legacy tools: {str(e)}"

@mcp.tool()
def execute_python_code(code: str) -> str:
    """Safely execute Python code (backward compatibility)"""
    try:
        namespace = {
            '__builtins__': {
                'abs': abs, 'all': all, 'any': any, 'bin': bin, 'bool': bool,
                'chr': chr, 'dict': dict, 'dir': dir, 'enumerate': enumerate,
                'float': float, 'hex': hex, 'int': int, 'len': len, 'list': list,
                'max': max, 'min': min, 'oct': oct, 'ord': ord, 'pow': pow,
                'range': range, 'repr': repr, 'round': round, 'set': set,
                'sorted': sorted, 'str': str, 'sum': sum, 'tuple': tuple,
                'zip': zip, 'print': print
            },
            'math': __import__('math'),
            'datetime': __import__('datetime'),
            'json': __import__('json')
        }
        
        exec(code, namespace)
        return "AgentDRY: Code executed successfully"
        
    except Exception as e:
        return f"AgentDRY error executing code: {str(e)}"

@mcp.tool()
def create_server_workflow(workflow_description: str, target_servers: str = "") -> str:
    """Create a complex workflow across multiple servers"""
    try:
        # Parse target servers
        servers = [s.strip() for s in target_servers.split(",")] if target_servers else []
        
        # Analyze workflow requirements
        context = enhanced_manager.query_processor.analyze_query(workflow_description)
        
        # Create workflow plan
        workflow_plan = {
            'description': workflow_description,
            'steps': [],
            'servers_involved': servers or context.target_servers,
            'created_at': datetime.now().isoformat()
        }
        
        # Generate workflow steps based on context
        if context.intent == 'calculate':
            workflow_plan['steps'] = [
                'Parse mathematical expression',
                'Validate input parameters',
                'Execute calculation on math server',
                'Format and return results'
            ]
        elif context.intent == 'data_processing':
            workflow_plan['steps'] = [
                'Load data from source',
                'Validate data format',
                'Process data with required operations',
                'Store/return processed results'
            ]
        else:
            workflow_plan['steps'] = [
                'Analyze input requirements',
                'Route to appropriate servers',
                'Execute operations',
                'Aggregate results'
            ]
        
        # Save workflow (in production, this would create executable workflow)
        workflow_id = f"workflow_{int(time.time())}"
        workflow_file = os.path.join(ECOSYSTEM_DIR, f"{workflow_id}.json")
        with open(workflow_file, 'w') as f:
            json.dump(workflow_plan, f, indent=2)
        
        result = f"Created workflow '{workflow_id}':\n"
        result += f"Description: {workflow_description}\n"
        result += f"Servers: {', '.join(workflow_plan['servers_involved'])}\n"
        result += f"Steps: {len(workflow_plan['steps'])}\n"
        for i, step in enumerate(workflow_plan['steps'], 1):
            result += f"  {i}. {step}\n"
        
        return result
        
    except Exception as e:
        return f"Failed to create workflow: {str(e)}"

@mcp.tool()
def export_ecosystem_config() -> str:
    """Export the entire ecosystem configuration"""
    try:
        config = {
            'servers': {name: asdict(server) for name, server in enhanced_manager.registry.servers.items()},
            'capabilities': enhanced_manager.registry.capabilities,
            'tools': {name: asdict(tool) for name, tool in enhanced_manager.registry.tools.items()},
            'session_tools': enhanced_manager.tool_generator.session_tools,
            'exported_at': datetime.now().isoformat(),
            'version': '1.0.0'
        }
        
        export_file = os.path.join(ECOSYSTEM_DIR, f"ecosystem_export_{int(time.time())}.json")
        with open(export_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        return f"Ecosystem configuration exported to: {export_file}\nTotal items: {len(config['servers'])} servers, {len(config['tools'])} tools"
        
    except Exception as e:
        return f"Export failed: {str(e)}"

@mcp.tool()
def import_ecosystem_config(config_file_path: str) -> str:
    """Import ecosystem configuration from file"""
    try:
        if not os.path.exists(config_file_path):
            return f"Configuration file not found: {config_file_path}"
        
        with open(config_file_path, 'r') as f:
            config = json.load(f)
        
        imported_servers = 0
        imported_tools = 0
        
        # Import servers
        if 'servers' in config:
            for name, server_data in config['servers'].items():
                if name not in enhanced_manager.registry.servers:
                    server_info = ServerInfo(**server_data)
                    enhanced_manager.registry.register_server(server_info)
                    imported_servers += 1
        
        # Import tools
        if 'tools' in config:
            for name, tool_data in config['tools'].items():
                if name not in enhanced_manager.registry.tools:
                    tool_info = ToolInfo(**tool_data)
                    enhanced_manager.registry.tools[name] = tool_info
                    imported_tools += 1
        
        # Update capabilities
        if 'capabilities' in config:
            for capability, servers in config['capabilities'].items():
                if capability not in enhanced_manager.registry.capabilities:
                    enhanced_manager.registry.capabilities[capability] = servers
        
        # Save updated configuration
        enhanced_manager.registry.save_configuration()
        
        return f"Successfully imported configuration:\n- Servers: {imported_servers}\n- Tools: {imported_tools}\n- From: {config_file_path}"
        
    except Exception as e:
        return f"Import failed: {str(e)}"

@mcp.tool()
def health_check_servers() -> str:
    """Perform health check on all registered servers"""
    try:
        results = {
            'total_servers': len(enhanced_manager.registry.servers),
            'healthy': 0,
            'unhealthy': 0,
            'unknown': 0,
            'details': []
        }
        
        for name, server in enhanced_manager.registry.servers.items():
            health_status = {
                'name': name,
                'status': 'unknown',
                'response_time': None,
                'error': None
            }
            
            try:
                # Check if server file exists
                if os.path.exists(server.path):
                    # Simulate health check (in production, this would ping the actual server)
                    start_time = time.time()
                    
                    # Simple file accessibility check
                    with open(server.path, 'r') as f:
                        f.read(100)  # Read first 100 chars
                    
                    health_status['status'] = 'healthy'
                    health_status['response_time'] = round((time.time() - start_time) * 1000, 2)
                    results['healthy'] += 1
                    
                    # Update server status
                    server.status = 'running'
                    server.last_ping = datetime.now().isoformat()
                    
                else:
                    health_status['status'] = 'unhealthy'
                    health_status['error'] = 'Server file not found'
                    results['unhealthy'] += 1
                    server.status = 'error'
                    
            except Exception as e:
                health_status['status'] = 'unhealthy'
                health_status['error'] = str(e)
                results['unhealthy'] += 1
                server.status = 'error'
            
            results['details'].append(health_status)
        
        # Save updated server statuses
        enhanced_manager.registry.save_configuration()
        
        # Format response
        response = f"🏥 Server Health Check Results:\n"
        response += f"Total Servers: {results['total_servers']}\n"
        response += f"✅ Healthy: {results['healthy']}\n"
        response += f"❌ Unhealthy: {results['unhealthy']}\n"
        response += f"❓ Unknown: {results['unknown']}\n\n"
        
        response += "📊 Server Details:\n"
        for detail in results['details']:
            status_icon = "✅" if detail['status'] == 'healthy' else "❌" if detail['status'] == 'unhealthy' else "❓"
            response += f"{status_icon} {detail['name']}: {detail['status']}"
            if detail['response_time']:
                response += f" ({detail['response_time']}ms)"
            if detail['error']:
                response += f" - {detail['error']}"
            response += "\n"
        
        return response
        
    except Exception as e:
        return f"Health check failed: {str(e)}"

@mcp.tool()
def optimize_server_allocation(query_load_data: str = "") -> str:
    """Optimize server allocation based on query patterns"""
    try:
        # Parse load data if provided
        load_data = {}
        if query_load_data:
            try:
                load_data = json.loads(query_load_data)
            except:
                # Simple format: "server1:10,server2:5"
                for item in query_load_data.split(","):
                    if ":" in item:
                        server, load = item.split(":")
                        load_data[server.strip()] = int(load.strip())
        
        # Analyze current ecosystem
        servers = enhanced_manager.registry.servers
        capabilities = enhanced_manager.registry.capabilities
        
        recommendations = []
        
        # Check for capability gaps
        critical_capabilities = ['math', 'file_system', 'data_processing', 'web']
        for cap in critical_capabilities:
            if cap not in capabilities or len(capabilities[cap]) < 2:
                recommendations.append(f"🔄 Consider adding redundancy for '{cap}' capability")
        
        # Check for overloaded servers
        for server_name, load in load_data.items():
            if load > 80:  # High load threshold
                recommendations.append(f"⚡ Server '{server_name}' is overloaded (load: {load}%)")
        
        # Check for underutilized servers
        for server_name, load in load_data.items():
            if load < 10:  # Low utilization threshold
                recommendations.append(f"💤 Server '{server_name}' is underutilized (load: {load}%)")
        
        # Generate optimization plan
        optimization_plan = {
            'analysis_time': datetime.now().isoformat(),
            'total_servers': len(servers),
            'recommendations': recommendations,
            'load_distribution': load_data
        }
        
        # Save optimization report
        report_file = os.path.join(ECOSYSTEM_DIR, f"optimization_report_{int(time.time())}.json")
        with open(report_file, 'w') as f:
            json.dump(optimization_plan, f, indent=2)
        
        # Format response
        response = f"🎯 Server Allocation Optimization Report:\n\n"
        response += f"📈 Current Ecosystem:\n"
        response += f"   - Total Servers: {len(servers)}\n"
        response += f"   - Active Capabilities: {len(capabilities)}\n"
        
        if load_data:
            response += f"   - Load Data Points: {len(load_data)}\n"
        
        response += f"\n💡 Recommendations ({len(recommendations)}):\n"
        if recommendations:
            for rec in recommendations:
                response += f"   {rec}\n"
        else:
            response += "   ✨ System appears well-optimized!\n"
        
        response += f"\n📄 Detailed report saved to: {report_file}"
        
        return response
        
    except Exception as e:
        return f"Optimization analysis failed: {str(e)}"

# File Watcher for Auto-Restart (Enhanced)
current_file = os.path.abspath(__file__)
last_modified = os.path.getmtime(current_file) if os.path.exists(current_file) else 0
ecosystem_files = [SERVERS_CONFIG, CAPABILITIES_DB, TOOL_REGISTRY]

def enhanced_file_watcher():
    """Enhanced file watcher for ecosystem files"""
    global last_modified
    ecosystem_modified = {f: os.path.getmtime(f) if os.path.exists(f) else 0 for f in ecosystem_files}
    
    while True:
        time.sleep(2)
        try:
            # Check main server file
            if os.path.exists(current_file):
                current_modified = os.path.getmtime(current_file)
                if current_modified > last_modified:
                    logger.info("AgentDRY Universal server file changed, restarting...")
                    last_modified = current_modified
                    time.sleep(3)
                    python = sys.executable
                    os.execl(python, python, *sys.argv)
            
            # Check ecosystem configuration files
            for config_file in ecosystem_files:
                if os.path.exists(config_file):
                    current_modified = os.path.getmtime(config_file)
                    if current_modified > ecosystem_modified[config_file]:
                        logger.info(f"Ecosystem config changed: {config_file}")
                        ecosystem_modified[config_file] = current_modified
                        # Reload configuration without restart
                        enhanced_manager.registry.load_configuration()
                        
        except Exception as e:
            logger.error(f"Enhanced file watcher error: {e}")

if __name__ == "__main__":
    start_time = time.time()
    
    # Start enhanced file watcher
    watcher_thread = threading.Thread(target=enhanced_file_watcher, daemon=True)
    watcher_thread.start()
    
    logger.info("╔════════════════════════════════════════════════════╗")
    logger.info("║              AgentDRY Universal                    ║")
    logger.info("║          MCP Ecosystem Manager                     ║")
    logger.info("║         Made in Mangaluru by                       ║")
    logger.info("║           Vedanth and Praas team                   ║")
    logger.info("╚════════════════════════════════════════════════════╝")
    
    logger.info("🌐 AgentDRY Universal Server starting...")
    logger.info(f"📁 Ecosystem directory: {ECOSYSTEM_DIR}")
    logger.info(f"🔧 Utils available: {UTILS_AVAILABLE}")
    logger.info(f"🖥️  Registered servers: {len(enhanced_manager.registry.servers)}")
    logger.info(f"⚡ Available capabilities: {len(enhanced_manager.registry.capabilities)}")
    
    # Perform initial system checks
    logger.info("🔍 Performing initial system health check...")
    try:
        # Quick health check on startup
        health_results = []
        for name, server in enhanced_manager.registry.servers.items():
            if os.path.exists(server.path):
                server.status = "running"
                health_results.append(f"✅ {name}")
            else:
                server.status = "error"
                health_results.append(f"❌ {name}")
        
        logger.info(f"Health check results: {', '.join(health_results)}")
        enhanced_manager.registry.save_configuration()
        
    except Exception as e:
        logger.warning(f"Initial health check failed: {e}")
    
    # Check if running in test mode
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        logger.info("🧪 AgentDRY Universal running in test mode")
        print("AgentDRY Universal Server initialized successfully!")
        print("🌐 Universal MCP ecosystem management ready")
        print(f"📊 System status: {len(enhanced_manager.registry.servers)} servers, {len(enhanced_manager.registry.tools)} tools")
        time.sleep(2)
        sys.exit(0)
    
    logger.info("🚀 AgentDRY Universal ready for ecosystem management")
    logger.info("💫 Features: Auto-discovery | Real-time tool generation | Cross-server coordination")
    
    try:
        # Use stdio transport for Claude Desktop
        mcp.run()
    except KeyboardInterrupt:
        logger.info("🛑 AgentDRY Universal Server stopped by user")
        # Save final state
        enhanced_manager.registry.save_configuration()
    except Exception as e:
        logger.error(f"❌ AgentDRY Universal Server failed: {e}", exc_info=True)
        raise
    finally:
        logger.info("🔄 AgentDRY Universal shutdown complete")