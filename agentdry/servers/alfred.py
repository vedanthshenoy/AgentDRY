"""
Alfred - helper utilities for dynamic tool creation and server management.
"""

import ast
import asyncio
import logging
import os
import re
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple


# ---------- Logging ----------

class Logger:
    @staticmethod
    def setup(log_dir: str = "logs") -> logging.Logger:
        os.makedirs(log_dir, exist_ok=True)
        logger = logging.getLogger("alfred")
        logger.setLevel(logging.INFO)

        # Avoid duplicate handlers if reloaded
        if not logger.handlers:
            fh = logging.FileHandler(os.path.join(log_dir, "server.log"), encoding="utf-8")
            ch = logging.StreamHandler()
            fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            fh.setFormatter(fmt)
            ch.setFormatter(fmt)
            logger.addHandler(fh)
            logger.addHandler(ch)
        return logger


# ---------- File Watcher (auto-restart) ----------

class FileWatcher:
    """Restarts the current Python process when the watched file changes."""

    def __init__(self, file_path: str, logger: logging.Logger):
        self.file_path = file_path
        self.logger = logger
        self.last_modified = os.path.getmtime(file_path)
        self._stop_event = threading.Event()

    def start_watching(self):
        t = threading.Thread(target=self._watch, daemon=True)
        t.start()
        self.logger.info(f"File watcher activated for: {self.file_path}")

    def stop_watching(self):
        self._stop_event.set()

    def _watch(self):
        while not self._stop_event.is_set():
            time.sleep(1)
            try:
                current = os.path.getmtime(self.file_path)
                if current > self.last_modified:
                    self.logger.info("File changed. Restarting server...")
                    self.last_modified = current
                    time.sleep(1.0)
                    python = sys.executable
                    os.execl(python, python, *sys.argv)
            except FileNotFoundError:
                self.logger.warning(f"Watched file {self.file_path} not found; stopping watcher.")
                break
            except Exception as e:
                self.logger.error(f"File watcher error: {e}", exc_info=True)


# ---------- Code validation & extraction ----------

@dataclass
class FunctionInfo:
    name: str
    parameters: List[str]
    docstring: str
    return_type: str = "Any"


class CodeValidator:
    @staticmethod
    def validate_syntax(code: str) -> bool:
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False

    @staticmethod
    def extract_function_info(code: str) -> Optional[FunctionInfo]:
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    params = []
                    for arg in node.args.args:
                        ann = "Any"
                        if arg.annotation:
                            if hasattr(arg.annotation, "id"):
                                ann = arg.annotation.id
                            elif hasattr(arg.annotation, "attr"):
                                ann = arg.annotation.attr
                        params.append(f"{arg.arg}: {ann}")
                    ret = "Any"
                    if node.returns:
                        if hasattr(node.returns, "id"):
                            ret = node.returns.id
                        elif hasattr(node.returns, "attr"):
                            ret = node.returns.attr
                    return FunctionInfo(
                        name=node.name,
                        parameters=params,
                        docstring=ast.get_docstring(node) or "No description available",
                        return_type=ret,
                    )
        except Exception:
            pass
        return None


# ---------- Server file manager ----------

class ServerFileManager:
    """Reads/writes the main server file to insert new @mcp.tool functions."""

    def __init__(self, file_path: str, logger: logging.Logger):
        self.file_path = file_path
        self.logger = logger

    def _parse(self) -> ast.Module:
        with open(self.file_path, "r", encoding="utf-8") as f:
            return ast.parse(f.read())

    @staticmethod
    def _has_mcp_tool_decorator(fn: ast.FunctionDef) -> bool:
        for dec in getattr(fn, "decorator_list", []):
            # Matches @mcp.tool or @mcp.tool()
            # dec can be Attribute (mcp.tool) or Call(Attribute(...))
            target = dec.func if isinstance(dec, ast.Call) else dec
            if isinstance(target, ast.Attribute) and target.attr == "tool":
                base = target.value
                if isinstance(base, ast.Name) and base.id == "mcp":
                    return True
        return False

    def function_exists(self, function_name: str) -> bool:
        try:
            tree = self._parse()
            for node in tree.body:
                if isinstance(node, ast.FunctionDef) and node.name == function_name:
                    # Any def with this name counts, to avoid duplicates
                    return True
            return False
        except Exception as e:
            self.logger.error(f"Error checking existing functions: {e}", exc_info=True)
            return False

    def list_mcp_functions(self) -> List[FunctionInfo]:
        out: List[FunctionInfo] = []
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                code = f.read()
            tree = ast.parse(code)
            for node in tree.body:
                if isinstance(node, ast.FunctionDef) and self._has_mcp_tool_decorator(node):
                    params: List[str] = []
                    for arg in node.args.args:
                        if arg.arg == "ctx":
                            continue
                        ann = "Any"
                        if arg.annotation:
                            if hasattr(arg.annotation, "id"):
                                ann = arg.annotation.id
                            elif hasattr(arg.annotation, "attr"):
                                ann = arg.annotation.attr
                        params.append(f"{arg.arg}: {ann}")
                    ret = "Any"
                    if node.returns:
                        if hasattr(node.returns, "id"):
                            ret = node.returns.id
                        elif hasattr(node.returns, "attr"):
                            ret = node.returns.attr
                    out.append(
                        FunctionInfo(
                            name=node.name,
                            parameters=params,
                            docstring=ast.get_docstring(node) or "No description available",
                            return_type=ret,
                        )
                    )
        except Exception as e:
            self.logger.error(f"Error listing MCP functions: {e}", exc_info=True)
        return out

    def append_function(self, code: str) -> bool:
        """Insert the new function (decorated) above the __main__ block."""
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            # Find the __main__ block to insert before it
            insert_idx: Optional[int] = None
            for i, line in enumerate(lines):
                if line.strip().startswith('if __name__ == "__main__"'):
                    insert_idx = i
                    break
            if insert_idx is None:
                self.logger.error("Cannot find __main__ block to insert function.")
                return False

            # Ensure the generated code starts with 'def '
            stripped = code.lstrip()
            if not stripped.startswith("def "):
                # If the LLM returned with backticks or extra text, try to extract def block
                def_block = self._extract_def_block(stripped)
                if def_block:
                    code = def_block
                else:
                    self.logger.error("Generated code doesn't contain a function starting with 'def '.")
                    return False

            decorated = f"\n@mcp.tool()\n{code.strip()}\n"
            new_lines = lines[:insert_idx] + [decorated] + lines[insert_idx:]
            with open(self.file_path, "w", encoding="utf-8") as f:
                f.writelines(new_lines)

            info = CodeValidator.extract_function_info(code)
            fn_name = info.name if info else "unknown"
            self.logger.info(f"Successfully integrated new function: {fn_name}")
            return True
        except Exception as e:
            self.logger.error(f"Error appending function: {e}", exc_info=True)
            return False

    @staticmethod
    def _extract_def_block(text: str) -> Optional[str]:
        """Grab a function block beginning at the first 'def ' and return until the end."""
        # Remove code fences if present
        text = re.sub(r"^```(?:python)?\s*", "", text.strip(), flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text.strip())
        # Find first def
        m = re.search(r"(^|\n)def\s+\w+\s*\(", text)
        if not m:
            return None
        start = m.start() if m.start() == 0 else m.start() + 1
        return text[start:]


# ---------- LLM code generator (via MCP sampling) ----------

class LLMCodeGenerator:
    SYSTEM_PROMPT = (
        "You are a Python code generator for MCP tools.\n"
        "Output ONLY a single Python function definition starting with 'def '.\n"
        "- Generalize specific queries (e.g., 'factorial of 4' -> def calculate_factorial(n: int) -> int:)\n"
        "- Add type hints and a concise docstring\n"
        "- No imports, no decorators, no explanations, no backticks\n"
        "- Robust error handling for invalid inputs\n"
    )

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    async def _sample(self, ctx: Any, **kwargs) -> Any:
        """
        Call the client's LLM via the server context, supporting multiple SDK variants:
        - Preferred: await ctx.sample(...)
        - Fallbacks observed in older drafts: await ctx.call_sampling(...), await ctx.request_sampling(...)
        """
        method = None
        for name in ("sample", "call_sampling", "request_sampling"):
            method = getattr(ctx, name, None)
            if method:
                break
        if method is None:
            raise AttributeError(
                "This FastMCP Context has no sampling method (sample/call_sampling/request_sampling). "
                "Upgrade fastmcp to >= 2.0.0."
            )
        result = method(**kwargs)
        if asyncio.iscoroutine(result):
            result = await result
        return result

    async def generate_function(self, query: str, ctx: Any) -> Tuple[bool, str]:
        try:
            user_prompt = f"Create a Python function to handle: {query}"
            self.logger.info(f"Initiating code generation for query: {query}")

            # Most-compatible form (per docs): messages can be a string
            response = await self._sample(
                ctx,
                messages=user_prompt,
                system_prompt=self.SYSTEM_PROMPT,
                temperature=0.1,
                max_tokens=500,
            )

            # Parse response into plain text
            generated = None
            # New fastmcp returns a TextContent with .text
            if hasattr(response, "text") and isinstance(response.text, str):
                generated = response.text
            # If some other structure is returned
            elif isinstance(response, dict) and "text" in response:
                generated = response["text"]
            elif isinstance(response, str):
                generated = response
            else:
                # Try common patterns (list of parts with .text)
                text_parts: List[str] = []
                for attr in ("content", "parts", "choices"):
                    val = getattr(response, attr, None) if not isinstance(response, dict) else response.get(attr)
                    if isinstance(val, list):
                        for item in val:
                            if isinstance(item, str):
                                text_parts.append(item)
                            elif hasattr(item, "text"):
                                text_parts.append(item.text)
                            elif isinstance(item, dict) and "text" in item:
                                text_parts.append(item["text"])
                if text_parts:
                    generated = "\n".join(text_parts)

            if not generated:
                return False, "LLM sampling returned empty response"

            cleaned = self._clean_generated_code(generated)
            if not cleaned:
                return False, "No valid function found in generated output"

            if not CodeValidator.validate_syntax(cleaned):
                return False, "Generated code contains syntax errors"

            return True, cleaned
        except Exception as e:
            self.logger.error(f"Error in code generation process: {e}", exc_info=True)
            return False, f"Error in code generation process: {e}"

    @staticmethod
    def _clean_generated_code(code: str) -> Optional[str]:
        # Remove code fences
        code = re.sub(r"^```(?:python)?\s*", "", code.strip(), flags=re.IGNORECASE)
        code = re.sub(r"\s*```$", "", code.strip())

        # Keep everything from the first 'def ' onward
        lines = code.splitlines()
        in_def = False
        kept: List[str] = []
        for line in lines:
            if not in_def and line.strip().startswith("def "):
                in_def = True
            if in_def:
                kept.append(line)
        return "\n".join(kept).strip() if kept else None


# ---------- Orchestrator ----------

class DynamicToolManager:
    def __init__(self, server_file_path: str):
        self.server_file_path = server_file_path
        self.logger = Logger.setup()
        self.file_manager = ServerFileManager(server_file_path, self.logger)
        self.code_generator = LLMCodeGenerator(self.logger)
        self.file_watcher = FileWatcher(server_file_path, self.logger)

    def initialize(self):
        self.file_watcher.start_watching()
        self.logger.info("Alfred systems initialized and operational")

    async def create_tool(self, query: str, ctx: Any) -> str:
        # Generate
        success, result = await self.code_generator.generate_function(query, ctx)
        if not success:
            return f"Code generation failed: {result}"

        code = result

        # Extract function info
        info = CodeValidator.extract_function_info(code)
        if not info:
            return "Error: Could not extract function information from generated code"

        # Avoid duplicates
        if self.file_manager.function_exists(info.name):
            return f"Function '{info.name}' already exists in the server"

        # Append to server
        if self.file_manager.append_function(code):
            return (
                f"Successfully created and deployed tool '{info.name}'. "
                f"Server will restart automatically to activate the new function."
            )
        else:
            return "Error: Failed to deploy function to server file"

    def list_available_functions(self) -> str:
        funcs = self.file_manager.list_mcp_functions()
        if not funcs:
            return "No functions currently deployed in the server"
        parts = []
        for f in funcs:
            params = ", ".join(f.parameters) if f.parameters else "no parameters"
            desc = (f.docstring or "No description").split(".")[0] + "."
            parts.append(f"• {f.name}({params})\n  {desc}")
        return "Available functions in this MCP server:\n\n" + "\n\n".join(parts)

    def get_server_info(self) -> str:
        return f"""Enhanced MCP Server with Dynamic Tool Creation
Powered by Alfred

Server Capabilities:
• Dynamic tool creation using client-side LLM sampling
• Auto-restart on file changes
• Function cataloging and duplicate prevention
• Code validation + error logging

Server file: {Path(self.server_file_path).name}
Log dir: logs/
"""

    def shutdown(self):
        self.file_watcher.stop_watching()
        self.logger.info("Alfred systems shutting down gracefully")
