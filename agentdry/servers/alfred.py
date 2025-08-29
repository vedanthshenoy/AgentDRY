"""
Alfred - The MCP Server's Faithful Assistant

This module contains all the helper classes and utilities needed for dynamic
tool creation and server management. Like Batman's Alfred, it handles all the
behind-the-scenes work so the main server can focus on its core mission.
"""

import logging
import os
import sys
import threading
import time
import ast
import re
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from pathlib import Path
from mcp.server.fastmcp import FastMCP, Context



@dataclass
class FunctionInfo:
    """Information about a function extracted from code."""
    name: str
    parameters: List[str]
    docstring: str
    return_type: str = "Any"


class Logger:
    """Master Wayne's logging system - keeps track of everything that happens."""
    
    @staticmethod
    def setup(log_dir: str = "logs") -> logging.Logger:
        """Setup comprehensive logging with file and console handlers."""
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(log_dir, "server.log")),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)


class FileWatcher:
    """
    The vigilant guardian watching over server files.
    Automatically restarts the server when changes are detected.
    """
    
    def __init__(self, file_path: str, logger: logging.Logger):
        self.file_path = file_path
        self.logger = logger
        self.last_modified = os.path.getmtime(file_path)
        self._stop_event = threading.Event()
    
    def start_watching(self):
        """Begin the eternal watch over the server file."""
        watcher_thread = threading.Thread(target=self._watch_file, daemon=True)
        watcher_thread.start()
        self.logger.info(f"File watcher activated for: {self.file_path}")
    
    def _watch_file(self):
        """The main watching loop - vigilant and patient."""
        while not self._stop_event.is_set():
            time.sleep(1)
            try:
                current_modified = os.path.getmtime(self.file_path)
                if current_modified > self.last_modified:
                    self.logger.info("File modification detected. Initiating server restart sequence...")
                    self.last_modified = current_modified
                    time.sleep(3)  # Allow graceful resource cleanup
                    python = sys.executable
                    os.execl(python, python, *sys.argv)
            except FileNotFoundError:
                self.logger.warning(f"Watched file {self.file_path} has vanished. Ceasing watch.")
                break
            except Exception as e:
                self.logger.error(f"Error in file watcher: {e}", exc_info=True)
    
    def stop_watching(self):
        """Stop the file watcher when duty is complete."""
        self._stop_event.set()


class CodeValidator:
    """
    The code quality inspector - ensures all generated code meets standards.
    No syntax errors shall pass through these gates.
    """
    
    @staticmethod
    def validate_syntax(code: str) -> bool:
        """Rigorously validate Python code syntax."""
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False
    
    @staticmethod
    def extract_function_info(code: str) -> Optional[FunctionInfo]:
        """Extract comprehensive function information from Python code."""
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Analyze parameters with type hints
                    params = []
                    for arg in node.args.args:
                        arg_type = "Any"
                        if arg.annotation:
                            if hasattr(arg.annotation, 'id'):
                                arg_type = arg.annotation.id
                            elif hasattr(arg.annotation, 'attr'):
                                arg_type = arg.annotation.attr
                        params.append(f"{arg.arg}: {arg_type}")
                    
                    # Determine return type
                    return_type = "Any"
                    if node.returns:
                        if hasattr(node.returns, 'id'):
                            return_type = node.returns.id
                        elif hasattr(node.returns, 'attr'):
                            return_type = node.returns.attr
                    
                    return FunctionInfo(
                        name=node.name,
                        parameters=params,
                        docstring=ast.get_docstring(node) or "No description available",
                        return_type=return_type
                    )
        except Exception:
            pass
        return None


class ServerFileManager:
    """
    The file operations specialist - handles all server file modifications
    with precision and care.
    """
    
    def __init__(self, file_path: str, logger: logging.Logger):
        self.file_path = file_path
        self.logger = logger
    
    def function_exists(self, function_name: str) -> bool:
        """Check if a function already exists in the server file."""
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Search for MCP tool decorator followed by function definition
            pattern = rf'@mcp\.tool\(\)\s*\ndef\s+{re.escape(function_name)}\s*\('
            return bool(re.search(pattern, content))
        except Exception as e:
            self.logger.error(f"Error checking function existence: {e}")
            return False
    
    def append_function(self, code: str) -> bool:
        """Carefully append a new function to the server file."""
        try:
            with open(self.file_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()

            # Locate the main execution block
            main_index = self._find_main_block_index(lines)
            if main_index is None:
                self.logger.error("Could not locate main execution block for function insertion")
                return False

            # Format and insert the new function
            formatted_code = f"\n@mcp.tool()\n{code}\n"
            lines = lines[:main_index] + [formatted_code] + lines[main_index:]
            
            # Write the updated content
            with open(self.file_path, 'w', encoding='utf-8') as file:
                file.writelines(lines)
            
            function_info = CodeValidator.extract_function_info(code)
            function_name = function_info.name if function_info else "unknown"
            self.logger.info(f"Successfully integrated new function: {function_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error appending function to server: {e}")
            return False
    
    def _find_main_block_index(self, lines: List[str]) -> Optional[int]:
        """Locate the main execution block in the file."""
        for i, line in enumerate(lines):
            stripped_line = ' '.join(line.strip().split())
            if stripped_line.startswith('if __name__ == "__main__"'):
                return i
        return None
    
    def list_mcp_functions(self) -> List[FunctionInfo]:
        """Catalog all MCP tool functions in the server file."""
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            functions = []
            
            for i, node in enumerate(tree.body):
                if (isinstance(node, ast.FunctionDef) and 
                    self._has_mcp_tool_decorator(tree.body, i)):
                    
                    func_info = self._extract_function_info_from_node(node)
                    if func_info:
                        functions.append(func_info)
            
            return functions
            
        except Exception as e:
            self.logger.error(f"Error cataloging functions: {e}")
            return []
    
    def _has_mcp_tool_decorator(self, body: List[ast.stmt], index: int) -> bool:
        """Verify if a function has the @mcp.tool() decorator."""
        if index <= 0:
            return False
        
        prev_node = body[index - 1]
        if not (isinstance(prev_node, ast.Expr) and isinstance(prev_node.value, ast.Call)):
            return False
        
        call_node = prev_node.value
        return (hasattr(call_node.func, 'attr') and 
                call_node.func.attr == 'tool' and
                hasattr(call_node.func, 'value') and
                hasattr(call_node.func.value, 'id') and
                call_node.func.value.id == 'mcp')
    
    def _extract_function_info_from_node(self, node: ast.FunctionDef) -> Optional[FunctionInfo]:
        """Extract detailed function information from an AST node."""
        try:
            # Process parameters (excluding 'ctx' parameter)
            params = []
            for arg in node.args.args:
                if arg.arg != 'ctx':
                    arg_type = "Any"
                    if arg.annotation:
                        if hasattr(arg.annotation, 'id'):
                            arg_type = arg.annotation.id
                        elif hasattr(arg.annotation, 'attr'):
                            arg_type = arg.annotation.attr
                    params.append(f"{arg.arg}: {arg_type}")
            
            # Determine return type
            return_type = "Any"
            if node.returns:
                if hasattr(node.returns, 'id'):
                    return_type = node.returns.id
                elif hasattr(node.returns, 'attr'):
                    return_type = node.returns.attr
            
            return FunctionInfo(
                name=node.name,
                parameters=params,
                docstring=ast.get_docstring(node) or "No description available",
                return_type=return_type
            )
        except Exception:
            return None


class LLMCodeGenerator:
    """
    The code creation specialist - transforms natural language requests
    into working Python functions using the client's LLM.
    """
    
    SYSTEM_PROMPT = """You are a Python code generator specialized in creating reusable functions for mathematical and utility operations.

Generate ONLY the Python function definition (starting with 'def') based on the user's request. Follow these guidelines:

1. Create a GENERALIZED function that can handle similar requests
2. If asked for specific calculations (like "factorial of 5"), create a general function (like "calculate_factorial")
3. If asked for specific operations, generalize them
4. Include proper type hints and docstrings
5. Start directly from the 'def' keyword - NO decorators, headers, or language indicators
6. Make functions robust with error handling
7. Return meaningful results and error messages
8. Use descriptive function and parameter names

Example transformations:
- "factorial of 5" → def calculate_factorial(n: int) -> int:
- "convert 100 USD to EUR" → def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
- "square root of number" → def calculate_square_root(number: float) -> float:

Generate ONLY the function definition - no explanations or additional text."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def generate_function(self, query: str, ctx: Context) -> Tuple[bool, str]:
        """
        Transform a natural language request into Python function code.
        
        Returns:
            Tuple[bool, str]: (success, code_or_error_message)
        """
        try:
            user_message = f"Create a Python function for: {query}"
            
            self.logger.info(f"Initiating code generation for query: {query}")
            generated_code = ctx.sample(
                messages=[user_message],
                system_prompt=self.SYSTEM_PROMPT,
                temperature=0.1,
                max_tokens=500
            )
            
            if not generated_code or not generated_code.strip():
                return False, "LLM sampling returned empty response"
            
            # Clean and validate the generated code
            cleaned_code = self._clean_generated_code(generated_code)
            if not cleaned_code:
                return False, "No valid function found in generated code"
            
            if not CodeValidator.validate_syntax(cleaned_code):
                return False, "Generated code contains syntax errors"
            
            return True, cleaned_code
            
        except Exception as e:
            error_msg = f"Error in code generation process: {str(e)}"
            self.logger.error(error_msg)
            return False, error_msg
    
    def _clean_generated_code(self, code: str) -> Optional[str]:
        """Extract and clean the function definition from generated code."""
        code_lines = code.strip().split('\n')
        cleaned_lines = []
        in_function = False
        
        for line in code_lines:
            # Begin capturing when we find the function definition
            if line.strip().startswith('def ') and not in_function:
                in_function = True
            
            if in_function:
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines) if cleaned_lines else None


class DynamicToolManager:
    """
    The master orchestrator - Alfred's main brain that coordinates all
    operations for dynamic tool creation and server management.
    """
    
    def __init__(self, server_file_path: str):
        self.server_file_path = server_file_path
        self.logger = Logger.setup()
        self.file_manager = ServerFileManager(server_file_path, self.logger)
        self.code_generator = LLMCodeGenerator(self.logger)
        self.file_watcher = FileWatcher(server_file_path, self.logger)
    
    def initialize(self):
        """Initialize all systems and begin operations."""
        self.file_watcher.start_watching()
        self.logger.info("Alfred systems initialized and operational")
    
    def create_tool(self, query: str, ctx: Context) -> str:
        """
        The main tool creation orchestrator - handles the entire process
        from query to deployed function.
        
        Args:
            query (str): Description of the tool to create
            ctx (Context): FastMCP context for LLM sampling
        
        Returns:
            str: Status message about the operation
        """
        try:
            # Generate the function code using LLM
            success, result = self.code_generator.generate_function(query, ctx)
            if not success:
                return f"Code generation failed: {result}"
            
            code = result
            
            # Extract and validate function information
            function_info = CodeValidator.extract_function_info(code)
            if not function_info:
                return "Error: Could not extract function information from generated code"
            
            # Check for existing function conflicts
            if self.file_manager.function_exists(function_info.name):
                return f"Function '{function_info.name}' already exists in the server"
            
            # Deploy the function to the server
            if self.file_manager.append_function(code):
                return (f"Successfully created and deployed tool '{function_info.name}'. "
                       f"Server will restart automatically to activate the new function.")
            else:
                return "Error: Failed to deploy function to server file"
                
        except Exception as e:
            error_msg = f"Tool creation failed: {str(e)}"
            self.logger.error(error_msg)
            return error_msg
    
    def list_available_functions(self) -> str:
        """Provide a comprehensive catalog of all available server functions."""
        try:
            functions = self.file_manager.list_mcp_functions()
            
            if not functions:
                return "No functions currently deployed in the server"
            
            function_descriptions = []
            for func in functions:
                param_str = ", ".join(func.parameters) if func.parameters else "no parameters"
                desc = func.docstring.split('.')[0] + '.' if func.docstring else "No description"
                function_descriptions.append(f"• {func.name}({param_str})\n  {desc}")
            
            return f"Available functions in this MCP server:\n\n" + "\n\n".join(function_descriptions)
            
        except Exception as e:
            error_msg = f"Error cataloging functions: {str(e)}"
            self.logger.error(error_msg)
            return error_msg
    
    def get_server_info(self) -> str:
        """Provide comprehensive information about server capabilities."""
        return f"""Enhanced MCP Server with Dynamic Tool Creation
Powered by Alfred - The Faithful Assistant

Server Capabilities:
• Mathematical operations and calculations
• Dynamic tool creation using client-side LLM sampling
• Automatic server restart when new tools are deployed
• Comprehensive function catalog and server information
• Code validation and error handling

Key Features:
- Zero server-side AI costs (uses client's LLM)
- Rigorous code validation before deployment
- Duplicate function prevention
- Automatic file monitoring and restart
- Detailed logging and error reporting

Commands:
- create_tool("description") - Create new functionality
- list_available_functions() - View all available tools
- get_server_info() - Display this information

Technical Details:
Server file: {Path(self.server_file_path).name}
Log directory: logs/
Helper module: alfred.py
"""

    def shutdown(self):
        """Gracefully shutdown all Alfred systems."""
        self.file_watcher.stop_watching()
        self.logger.info("Alfred systems shutting down gracefully")