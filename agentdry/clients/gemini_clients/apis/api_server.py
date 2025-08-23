import sys
import os
import asyncio
import logging
import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Tuple

# Adjust sys.path to find new_gemi_client.py in the parent directory
# If api_server.py is in 'your_project/apis/', this will add 'your_project/' to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import directly from new_gemi_client.py
try:
    from new_gemi_client import (
        GeminiLLM,
        load_tools,
        process_query,
        predict_tool_name,
        LOG_DIR, # Import LOG_DIR for consistent logging setup
        extract_info_and_create_tool, # For tool creation
        create_and_update_tool # For tool creation
    )
    # Also import types if needed for pydantic models or internal types
    from google.genai import types
    from mcp import ClientSession
    from mcp.client.sse import sse_client # We need this for the ClientSession initialization
    import httpx # For handling httpx.ReadError if needed in a global handler
except ImportError as e:
    print(f"Error importing from new_gemi_client.py: {e}")
    print("Please ensure 'new_gemi_client.py' is in the parent directory of this API file.")
    print("Also ensure all dependencies (google-generativeai, python-dotenv, mcp, httpx) are installed.")
    sys.exit(1)


# --- Setup Logging ---
# Use the LOG_DIR from new_gemi_client.py for consistency
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "api.log")), # Separate log for API
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Gemini LLM Client API",
    description="API for interacting with the Gemini LLM, managing tools, and processing queries.",
    version="1.0.0"
)

# Global instances for LLM and MCP Session (initialized on startup)
llm: GeminiLLM = None
# We will manage the ClientSession within the request scope or use a re-connect logic
# session: ClientSession = None # Not directly global anymore due to SSE context

current_tools_declarations: types.Tool = None # Stored in Gemini's Tool format
MCP_SERVER_URL = "http://localhost:8000/sse" # Default SSE URL for MCP server

# --- Request Models ---
class QueryRequest(BaseModel):
    query: str
    messages_history: List[Dict[str, Any]] = []

class ToolCreationRequest(BaseModel):
    tool_topic: str

# --- API Endpoints ---

@app.on_event("startup")
async def startup_event():
    logger.info("FastAPI application starting up.")
    global llm, current_tools_declarations

    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        logger.critical("GEMINI_API_KEY environment variable not found. Exiting.")
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY environment variable not found.")

    llm = GeminiLLM(api_key=gemini_api_key)

    # Attempt to load tools on startup to populate current_tools_declarations
    # This initial load is outside the request scope and needs its own session handling
    try:
        # Create a temporary session for initial tool loading
        # This closely mimics how new_gemi_client handles session creation for tool listing
        async with sse_client(url=MCP_SERVER_URL) as streams:
            async with ClientSession(*streams) as temp_session:
                await temp_session.initialize()
                tool_list_raw = await load_tools(temp_session)
                current_tools_declarations = types.Tool(function_declarations=tool_list_raw)
                logger.info(f"Loaded {len(tool_list_raw)} tools on startup.")
    except httpx.ConnectError as e:
        logger.error(f"Could not connect to MCP server at {MCP_SERVER_URL} during startup: {e}")
        # Allow startup, but features requiring MCP will fail
    except Exception as e:
        logger.error(f"Error during initial tool loading: {e}", exc_info=True)
        # Allow startup, but features requiring MCP will fail


@app.get("/health")
async def health_check():
    """
    Checks the health of the API.
    """
    return {"status": "healthy", "message": "Gemini LLM Client API is running."}

@app.post("/process_query")
async def process_user_query(request: QueryRequest):
    """
    Processes a user query, interacts with the LLM, and potentially calls a tool.
    """
    logger.info(f"Received query: {request.query}")
    logger.info(f"Messages history (truncated for log): {request.messages_history[-1:]}")

    if not llm or not current_tools_declarations:
        raise HTTPException(status_code=503, detail="LLM or tools not initialized.")

    # Convert incoming messages_history to the format expected by Gemini (List[types.Content])
    gemini_messages_history = [
        types.Content(parts=msg["parts"], role=msg["role"])
        for msg in request.messages_history
    ]

    try:
        # Open a new ClientSession for each request that needs to interact with MCP
        # This aligns with new_gemi_client's session management within its loop
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
        raise HTTPException(status_code=503, detail="Failed to connect to MCP server.")
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing query: {e}")


@app.post("/determine_tool_necessity")
async def determine_tool_necessity_endpoint(request: QueryRequest):
    """
    Determines if a given query requires a new tool or can be answered directly.
    """
    logger.info(f"Determining tool necessity for: {request.query}")
    if not llm:
        raise HTTPException(status_code=503, detail="LLM not initialized.")
    try:
        tool_needed = await llm.determine_tool_creation_necessity(request.query)
        return {"tool_needed": tool_needed}
    except Exception as e:
        logger.error(f"Error determining tool necessity: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error determining tool necessity: {e}")

@app.post("/classify_question")
async def classify_question_endpoint(request: QueryRequest):
    """
    Classifies a question as 'general' or 'direct' and extracts parts if direct.
    """
    logger.info(f"Classifying question: {request.query}")
    if not llm:
        raise HTTPException(status_code=503, detail="LLM not initialized.")
    try:
        qtype, general_part, specific_part = await llm.classify_and_split_question(request.query)
        return {
            "type": qtype,
            "general_part": general_part,
            "specific_part": specific_part
        }
    except Exception as e:
        logger.error(f"Error classifying question: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error classifying question: {e}")

@app.post("/create_tool_and_update")
async def create_tool_and_update_endpoint(request: ToolCreationRequest):
    """
    Endpoint to trigger the tool creation and update process.
    This calls the actual `extract_info_and_create_tool` and `create_and_update_tool` calls.
    """
    logger.info(f"Request to create tool for topic: {request.tool_topic}")
    if not llm:
        raise HTTPException(status_code=503, detail="LLM not initialized.")

    try:
        proposed_name = predict_tool_name(request.tool_topic)
        logger.info(f"Attempting tool creation for '{request.tool_topic}' with proposed name '{proposed_name}'.")

        # These operations might cause the MCP server to restart,
        # leading to a temporary disconnection.
        extract_info_and_create_tool(query=request.tool_topic)
        create_and_update_tool(query=f"Create a function for {request.tool_topic}")

        logger.info("Tool creation process initiated. Waiting for MCP server to restart and re-load tools...")
        # Give the MCP server time to restart and re-establish connection
        await asyncio.sleep(5) # This sleep is crucial for the server restart

        # Re-load tools after creation by establishing a new session
        global current_tools_declarations
        async with sse_client(url=MCP_SERVER_URL) as streams:
            async with ClientSession(*streams) as session_for_reload:
                await session_for_reload.initialize()
                tool_list_raw = await load_tools(session_for_reload)
                current_tools_declarations = types.Tool(function_declarations=tool_list_raw)
                logger.info(f"Reloaded {len(tool_list_raw)} tools after creation.")

        return {"status": "success", "message": f"Tool creation process initiated and tools reloaded for '{request.tool_topic}'.", "proposed_tool_name": proposed_name}
    except httpx.ConnectError as e:
        logger.error(f"Could not connect to MCP server after tool creation at {MCP_SERVER_URL}: {e}")
        raise HTTPException(status_code=503, detail="Failed to connect to MCP server after tool creation. It might be restarting.")
    except Exception as e:
        logger.error(f"Error during tool creation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error during tool creation: {e}")

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
        raise HTTPException(status_code=503, detail="Failed to connect to MCP server.")
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