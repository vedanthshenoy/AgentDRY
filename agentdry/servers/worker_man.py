from typing import Dict, Any, List, Optional
import os, sys, ast
from datetime import datetime

class MCPServerAnalyzer:
    """
    Analyzer class for MCP Server Python files.
    
    This class loads a server file, parses its AST,
    extracts metadata, tools, imports, and statistics,
    and then generates a detailed report.
    """

    def __init__(self, server_file_path: str):
        """Initialize analyzer with server file path."""
        self.server_file_path = os.path.abspath(server_file_path)
        self.code_content = None
        self.ast_tree = None
        
    def analyze_and_save(self, output_file_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform full analysis and save report to file.
        """
        try:
            if not self._load_file():
                return {"success": False, "error": f"File not found : {self.server_file_path}"}
            if not self._parse_code():
                return {"success": False, "error": "Syntax error in file"}
            
            tools = self._extract_tools()
            imports = self._extract_imports()
            metadata = self._extract_metadata()
            stats = self._calculate_stats()
            
            if output_file_path is None:
                base_name = os.path.splitext(os.path.basename(self.server_file_path))[0]
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file_path = f"{base_name}_analysis_{timestamp}.txt"
            
            report_content = self._generate_report(metadata, tools, imports, stats)
            
            with open(output_file_path, "w", encoding='utf-8') as f:
                f.write(report_content)
                
            return {
                "success": True,
                "server_file": self.server_file_path,
                "output_file": output_file_path,
                "server_name": metadata['server_name'],
                "tools_count": len(tools),
                "file_size": os.path.getsize(output_file_path),
                "generated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"success": False, "error": f"Analysis failed: {str(e)}"}
        
    def _load_file(self) -> bool:
        """Load the target Python file into memory."""
        try:
            if not os.path.exists(self.server_file_path):
                return False
            with open(self.server_file_path, 'r', encoding='utf-8') as f:
                self.code_content = f.read()
            return True
        except Exception:
            return False
        
    def _parse_code(self) -> bool:
        """Parse file content into an AST."""
        try:
            self.ast_tree = ast.parse(self.code_content)
            return True
        except SyntaxError:
            return False
        
    def _extract_tools(self) -> List[Dict[str, Any]]:
        """
        Extract tool functions decorated with @tool, @server.tool, @mcp.tool, or @mcp.tool().
        """
        tools = []
        
        for node in ast.walk(self.ast_tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for decorator in node.decorator_list:
                    is_tool = False
                    
                    # Handle decorators like @server.tool, @mcp.tool
                    if isinstance(decorator, ast.Attribute):
                        if decorator.attr == 'tool' and isinstance(decorator.value, ast.Name):
                            is_tool = True
                    
                    # Handle decorators like @mcp.tool()
                    elif isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Attribute):
                        if decorator.func.attr == 'tool' and isinstance(decorator.func.value, ast.Name):
                            is_tool = True
                    
                    # Handle decorators like @tool
                    elif isinstance(decorator, ast.Name) and decorator.id == 'tool':
                        is_tool = True
                        
                    if is_tool:
                        tools.append({
                            "name": node.name,
                            "docstring": ast.get_docstring(node) or "No description available",
                            "parameters": self._extract_parameters(node),
                            "line_number": node.lineno,
                            "is_async": isinstance(node, ast.AsyncFunctionDef)
                        })
                        break
        return tools
    
    def _extract_parameters(self, func_node) -> List[Dict[str, str]]:
        """Extract parameters and annotations from function definition."""
        params = []
        
        for arg in func_node.args.args:
            annotation = "Any"
            if arg.annotation:
                if isinstance(arg.annotation, ast.Name):
                    annotation = arg.annotation.id
                elif hasattr(ast, 'unparse'):
                    annotation = ast.unparse(arg.annotation)
                else:
                    annotation = "complex_type"
                    
            params.append({
                "name": arg.arg,
                "annotation": annotation
            })
            
        return params
    
    def _extract_imports(self) -> List[str]:
        """Extract all imports from the server file."""
        imports = []
        
        for node in ast.walk(self.ast_tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(f"import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    imports.append(f"from {module} import {alias.name}")
                    
        return imports
    
    def _extract_metadata(self) -> Dict[str, str]:
        """
        Extract server metadata such as name, description, and framework.
        Supports variable names other than 'server' (e.g., 'mcp').
        """
        metadata = {
            "server_name": "Unknown",
            "description": "No description available",
            "framework": "Unknown"
        }
        
        # Module-level docstring as description
        if (self.ast_tree.body and 
            isinstance(self.ast_tree.body[0], ast.Expr) and
            isinstance(self.ast_tree.body[0].value, ast.Constant) and 
            isinstance(self.ast_tree.body[0].value.value, str)):
            metadata["description"] = self.ast_tree.body[0].value.value
            
        # Look for assignment like: mcp = FastMCP("Name")
        for node in ast.walk(self.ast_tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        if isinstance(node.value, ast.Call):
                            if (isinstance(node.value.func, ast.Name) and
                                node.value.func.id == "FastMCP"):
                                metadata["framework"] = "FastMCP"
                                if (node.value.args and 
                                    isinstance(node.value.args[0], ast.Constant)):
                                    metadata["server_name"] = node.value.args[0].value
        return metadata
    
    def _calculate_stats(self) -> Dict[str, int]:
        """Calculate basic file statistics."""
        return {
            "total_lines": len(self.code_content.split('\n')),
            "file_size_bytes": len(self.code_content.encode('utf-8')),
            "functions_count": len([n for n in ast.walk(self.ast_tree) if isinstance(n, ast.FunctionDef)]),
            "classes_count": len([n for n in ast.walk(self.ast_tree) if isinstance(n, ast.ClassDef)])
        }
        
    def _generate_report(self, metadata: Dict, tools: List, imports: List, stats: Dict) -> str:
        """Generate formatted report string."""
        timestamp = datetime.now().isoformat()
        content = f"""MCP SERVER ANALYSIS CONTEXT
{"=" * 50}

Generated : {timestamp}
Server File : {self.server_file_path}

SERVER INFORMATION
{"=" * 20}
Name : {metadata['server_name']}
Framework : {metadata['framework']}
Description : {metadata['description']}

CODE STATISTICS
{"=" * 20}
Total Lines : {stats['total_lines']}
File Size : {stats['file_size_bytes']} bytes
Function : {stats['functions_count']}
Classes : {stats['classes_count']}

AVAILABLE TOOLS ({len(tools)})
{"=" * 25}
"""
        for i, tool in enumerate(tools, start=1):
            content += f"\n{i}. {tool['name']}\n"
            content += f"     Description : {tool['docstring']}\n"
            content += f"     Async : {tool['is_async']}\n"
            content += f"     Parameters : {len(tool['parameters'])}\n"
            
            for param in tool['parameters']:
                content += f"     - {param['name']} : {param['annotation']}\n"
                
            content += f"     Line : {tool['line_number']}\n"
            
        content += f"\nIMPORTS ({len(imports)})\n"
        content += "=" * 15 + "\n"
        for imp in imports:
            content += f"- {imp}\n"
            
        return content
    
def analyze_mcp_server(server_file_path: str, output_file_path: Optional[str] = None) -> Dict[str, Any]:
    """Convenience function to run analysis on a server file."""
    analyzer = MCPServerAnalyzer(server_file_path)
    return analyzer.analyze_and_save(output_file_path)

def main():
    """CLI entry point for MCP Server Analyzer."""
    if len(sys.argv) < 2:
        print("MCP SERVER ANALYZER")
        return 
    
    server_file = sys.argv[1].strip('"').strip("'")
    output_file = sys.argv[2].strip('"').strip("'") if len(sys.argv) > 2 else None
    
    print(f"analyzing : {server_file}")
    
    result = analyze_mcp_server(server_file, output_file)

    if result["success"]:
        print(f"Tools found : {result['tools_count']}")
    else:
        print(f"FAILED : {result['error']}")
        
if __name__ == "__main__":
    main()
