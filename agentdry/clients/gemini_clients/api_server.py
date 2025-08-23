import sys
import os
import asyncio
import logging
import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Tuple

# Adjust sys.path to find new_gemi_client.py in the parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import directly from new_gemi_client.py
try:
    from new_gemi_client import (
        GeminiLLM,
        load_tools,
        process_query,
        predict_tool_name,
        LOG_DIR,
        extract_info_and_create_tool,
        create_and_update_tool,
        process_tool_creation_flow # <--- NEW IMPORT
    )
    from google.genai import types
    from mcp import ClientSession
    from mcp.client.sse import sse_client
    import httpx
except ImportError as e:
    print(f"Error importing from new_gemi_client.py: {e}")
    print("Please ensure 'new_gemi_client.py' is in the parent directory of this API file.")
    print("Also ensure all dependencies (google-generativeai, python-dotenv, mcp, httpx) are installed.")
    sys.exit(1)


# --- Setup Logging ---
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "api.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AgentDRY",
    description="Backend API for the ChatAgent, managing LLM interactions, tools, and query processing.",
    version="1.0.0"
)

# Global instances for LLM and current_tools_declarations
llm: GeminiLLM = None
current_tools_declarations: types.Tool = None
MCP_SERVER_URL = "http://localhost:8000/sse"

@app.on_event("startup")
async def startup_event():
    logger.info("FastAPI application starting up.")
    global llm, current_tools_declarations

    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        logger.critical("GEMINI_API_KEY environment variable not found. Exiting.")
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY environment variable not found.")

    llm = GeminiLLM(api_key=gemini_api_key)

    try:
        # Create a temporary session for initial tool loading
        async with sse_client(url=MCP_SERVER_URL) as streams:
            async with ClientSession(*streams) as temp_session:
                await temp_session.initialize()
                tool_list_raw = await load_tools(temp_session)
                current_tools_declarations = types.Tool(function_declarations=tool_list_raw)
                logger.info(f"Loaded {len(tool_list_raw)} tools on startup.")
    except httpx.ConnectError as e:
        logger.error(f"Could not connect to MCP server at {MCP_SERVER_URL} during startup: {e}. Some features may be impacted.")
        # Allow startup, but features requiring MCP will fail
    except Exception as e:
        logger.error(f"Error during initial tool loading: {e}", exc_info=True)
        # Allow startup, but features requiring MCP will fail


@app.get("/health")
async def health_check():
    """
    Checks the health of the API.
    """
    return {"status": "healthy", "message": "ChatAgent API is running."}

@app.post("/process_query")
async def process_user_query(request: QueryRequest):
    """
    Processes a user query, interacts with the LLM, and potentially calls a tool.
    """
    logger.info(f"Received query: {request.query}")
    logger.info(f"Messages history (truncated for log): {request.messages_history[-1:]}")

    if not llm or not current_tools_declarations:
        raise HTTPException(status_code=503, detail="LLM or tools not initialized.")

    gemini_messages_history = [
        types.Content(parts=msg["parts"], role=msg["role"])
        for msg in request.messages_history
    ]

    try:
        # Open a new ClientSession for each request that needs to interact with MCP
        async with sse_client(url=MCP_SERVER_URL) as streams:
            async with ClientSession(*streams) as session:
                await session.initialize()
                tool_called, updated_messages_gemini_format, response_text = await process_query(
                    session,
                    llm,
                    current_tools_declarations,
                    request.query,
                    gemini_messages_history
                )

        updated_messages_serializable = [
            {"role": msg.role, "parts": [{"text": part.text} for part in msg.parts if hasattr(part, 'text')]}
            for msg in updated_messages_gemini_format
        ]

        return {
            "tool_called": tool_called,
            "messages_history": updated_messages_serializable,
            "response": response_text
        }
    except httpx.ConnectError as e:
        logger.error(f"Could not connect to MCP server at {MCP_SERVER_URL}: {e}")
        raise HTTPException(status_code=503, detail="Failed to connect to MCP server. Please ensure the MCP server is running.")
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing query: {e}")

# Removed determine_tool_necessity_endpoint and classify_question_endpoint
# as their logic is now encapsulated within process_tool_creation_flow


@app.post("/create_tool_and_update")
async def create_tool_and_update_endpoint(request: ToolCreationRequest):
    """
    Endpoint to trigger the tool creation and update process using the integrated flow.
    """
    logger.info(f"Request to create tool for topic: {request.tool_topic}")
    if not llm:
        raise HTTPException(status_code=503, detail="LLM not initialized.")

    try:
        tool_created, proposed_name = await process_tool_creation_flow(llm, request.tool_topic)

        if tool_created:
            logger.info("Tool creation initiated in backend. Waiting for MCP server to restart and re-load tools...")
            await asyncio.sleep(10) # Increased sleep for more robustness

            # Re-load tools after creation by establishing a new session
            global current_tools_declarations
            try:
                async with sse_client(url=MCP_SERVER_URL) as streams:
                    async with ClientSession(*streams) as session_for_reload:
                        await session_for_reload.initialize()
                        tool_list_raw = await load_tools(session_for_reload)
                        current_tools_declarations = types.Tool(function_declarations=tool_list_raw)
                        logger.info(f"Reloaded {len(tool_list_raw)} tools after creation.")
            except httpx.ConnectError as e:
                logger.error(f"Could not reconnect to MCP server after tool creation at {MCP_SERVER_URL}: {e}")
                # This is critical, but we might have partial success with the tool file creation
                return {"status": "warning", "message": f"Tool '{proposed_name}' created, but failed to reconnect to MCP server to confirm reload. Check MCP logs.", "proposed_tool_name": proposed_name}

            return {"status": "success", "message": f"Tool '{proposed_name}' created and reloaded.", "proposed_tool_name": proposed_name}
        else:
            return {"status": "failed", "message": "Tool creation not deemed necessary or failed in initial steps.", "proposed_tool_name": None}

    except Exception as e:
        logger.error(f"Error during tool creation flow: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error during tool creation flow: {e}")

@app.get("/list_available_tools")
async def list_available_tools_endpoint():
    """
    Lists the tools currently available on the server.
    """
    logger.info("Listing available tools.")
    try:
        # Open a new ClientSession for listing tools
        async with sse_client(url=MCP_SERVER_URL) as streams:
            async with ClientSession(*streams) as session:
                await session.initialize()
                tool_list_raw = await load_tools(session)
                return {"tools": tool_list_raw}
    except httpx.ConnectError as e:
        logger.error(f"Could not connect to MCP server at {MCP_SERVER_URL}: {e}")
        raise HTTPException(status_code=503, detail="Failed to connect to MCP server. Please ensure the MCP server is running.")
    except Exception as e:
        logger.error(f"Error listing tools: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error listing tools: {e}")

# To run this FastAPI app:
# 1. Create a project structure like:
#    your_project/
#    ├── new_gemi_client.py
#    ├── log_mcp_server_autorestart.py
#    ├── apis/
#    │   └── api_server.py  (this file)
#    └── .env (if using GEMINI_API_KEY from dotenv)
# 2. Ensure all required dependencies are installed in your environment:
#    pip install fastapi uvicorn pydantic google-generativeai python-dotenv httpx
#    (And your 'mcp' library and 'utils' module if they are not standard packages)
# 3. Set your GEMINI_API_KEY environment variable.
# 4. First, start the MCP server from 'your_project/' directory:
#    python log_mcp_server_autorestart.py
# 5. Then, start the FastAPI application from 'your_project/' directory:
#    uvicorn apis.api_server:app --reload --port 8001
#    (Note: --port 8001 is used to avoid conflict with MCP server's default port 8000)
# 6. Access the API documentation at `http://127.0.0.1:8001/docs`