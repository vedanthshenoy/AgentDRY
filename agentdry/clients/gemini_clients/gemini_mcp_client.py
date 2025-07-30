import sys
import os
import re
import asyncio
import logging
from google.genai import types
from google import genai
from dotenv import load_dotenv
from mcp import ClientSession
from mcp.client.sse import sse_client
import httpx

# Add parent directory to system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from utils.openai_format_check import transform_schema, parse_response
from utils.initiate_autotool_creation import extract_info_and_create_tool
from main import create_and_update_tool

# --- Setup Logging ---
LOG_DIR = "logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "client.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()

class GeminiLLM:
    """A wrapper class for Google Gemini API interactions using genai.Client."""
    def __init__(self, api_key):
        if not api_key:
            logger.critical("Gemini API key not found.")
            raise ValueError("GEMINI_API_KEY is not set.")
        self.client = genai.Client(api_key=api_key)
        self.model_name = "gemini-2.5-flash-lite"

    async def _run_sync_in_executor(self, sync_function):
        """Runs a synchronous function in a thread pool to avoid blocking asyncio."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, sync_function)

    async def _generate(self, prompt_parts, temperature=0.0):
        """Helper for making synchronous generate_content calls."""
        try:
            def sync_call():
                return self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt_parts,
                    config=types.GenerateContentConfig(temperature=temperature),
                )
            response = await self._run_sync_in_executor(sync_call)
            return response.candidates[0].content.parts[0].text.strip()
        except Exception as e:
            logger.error(f"Error during Gemini content generation: {e}", exc_info=True)
            return ""

    async def classify_and_split_question(self, question):
        """Classifies a question as 'general' or 'direct' using the LLM."""
        prompt = (
            "Classify the following user question as 'general' or 'direct'. "
            "If it's direct, also extract the general part and the specific query part.\n"
            "Examples:\n"
            "Input: Find factorial of a number\n"
            "Output: general | Find factorial of a number | \n"
            "Input: Find the factorial of 5\n"
            "Output: direct | Find factorial of a number | Find the factorial of 5\n"
            f"Input: {question}\n"
            "Output:"
        )
        text = await self._generate(prompt)
        match = re.match(r"(general|direct)\s*\|\s*(.*?)\s*\|\s*(.*)", text, re.IGNORECASE)
        if match:
            qtype, general, specific = match.groups()
            return qtype.lower(), general.strip(), specific.strip()
        logger.warning("Failed to classify question, defaulting to 'general'.")
        return "general", question, ""

    async def determine_tool_creation_necessity(self, query_text):
        """Determines if a query requires a new tool or can be answered directly."""
        prompt = (
            "Analyze the user query. Respond with 'TOOL_NEEDED' if a specialized function is "
            "required. Respond with 'NO_TOOL_NEEDED' for greetings or simple facts.\n"
            "Input: Create a script to calculate Fibonacci series\nOutput: TOOL_NEEDED\n"
            "Input: What is the capital of France?\nOutput: NO_TOOL_NEEDED\n"
            f"Input: {query_text}\nOutput:"
        )
        response_text = await self._generate(prompt)
        return response_text == "TOOL_NEEDED"

    async def get_model_response(self, messages_history, tools):
        """Generates a response from the LLM, potentially using tools."""
        try:
            def sync_call():
                # Correctly place `tools` inside GenerateContentConfig
                config = types.GenerateContentConfig(
                    temperature=0,
                    tools=[tools]
                )
                return self.client.models.generate_content(
                    model=self.model_name,
                    contents=messages_history,
                    config=config,
                )
            response = await self._run_sync_in_executor(sync_call)
            return response.candidates[0]
        except Exception as e:
            logger.error(f"Error getting model response with tools: {e}", exc_info=True)
            return None


async def load_tools(session):
    """Loads available tools from the MCP server."""
    tools_response = await session.list_tools()
    return [
        {"name": tool.name, "description": tool.description, "parameters": tool.inputSchema}
        for tool in tools_response.tools
    ]

def predict_tool_name(query_or_topic):
    """Predicts a snake_case tool name from a natural language query."""
    if not query_or_topic: return "default_tool"
    cleaned = re.sub(r'^(create|generate|write) a (tool|function) (for|to|that)\s+', '', query_or_topic, flags=re.IGNORECASE)
    cleaned = re.sub(r'[^a-zA-Z0-9\s]', '', cleaned).lower()
    snake_case = re.sub(r'\s+', '_', cleaned).strip('_')
    snake_case = re.sub(r'_{2,}', '_', snake_case)
    if not snake_case: return "default_action"
    return snake_case[:50].rsplit('_', 1)[0] if len(snake_case) > 50 else snake_case

async def process_query(session, llm, current_tools, query_text, messages_history):
    """Processes a single user query, deciding whether to call a tool or respond directly."""
    user_content = types.Content(parts=[{"text": query_text}], role="user")
    messages_history.append(user_content)

    candidate = await llm.get_model_response(messages_history, current_tools)
    if not candidate:
        return False, messages_history, "Sorry, I encountered an error trying to process that."

    parts = candidate.content.parts
    if hasattr(parts[0], "function_call") and parts[0].function_call:
        fc = parts[0].function_call
        logger.info(f"LLM wants to call function: {fc.name} with args: {fc.args}")
        try:
            result = await session.call_tool(fc.name, arguments=fc.args)
            tool_output = result.content[0].text if result and result.content else ""
            logger.info(f"Bot (tool output): {tool_output}")
            messages_history.append(types.Content(parts=[{"text": tool_output}], role="tool"))
            return True, messages_history, tool_output
        except Exception as e:
            logger.error(f"Error calling tool '{fc.name}': {e}", exc_info=True)
            return False, messages_history, ""
    else:
        bot_text = parts[0].text if hasattr(parts[0], "text") else ""
        return False, messages_history, bot_text

async def run():
    """The main application loop to handle user interaction, tool management, and conversation."""
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        logger.critical("GEMINI_API_KEY environment variable not found. Exiting.")
        sys.exit(1)
        
    llm = GeminiLLM(api_key=gemini_api_key)
    query_to_reprocess = None

    while True:
        try:
            async with sse_client(url="http://localhost:8000/sse") as streams:
                async with ClientSession(*streams) as session:
                    await session.initialize()
                    tool_list = await load_tools(session)
                    
                    logger.info("--- Available Tools on Server ---")
                    if tool_list:
                        for tool in tool_list:
                            logger.info(f"- Name: {tool['name']}, Desc: {tool['description']}")
                    else:
                        logger.info("No tools currently available.")
                    logger.info("---------------------------------")

                    existing_tool_names = {t['name'].lower() for t in tool_list}
                    tools = types.Tool(function_declarations=tool_list)
                    messages = []

                    if query_to_reprocess:
                        logger.info(f"Automatically re-processing query: {query_to_reprocess}")
                        user_query = query_to_reprocess
                        query_to_reprocess = None
                    else:
                        user_query = input("You: ")
                        logger.info(f"User: {user_query}")

                    while user_query.lower() != "quit":
                        tool_called, messages, response_text = await process_query(session, llm, tools, user_query, messages)

                        if tool_called:
                            user_query = input("You: ")
                            logger.info(f"User: {user_query}")
                            continue

                        if not await llm.determine_tool_creation_necessity(user_query):
                            logger.info(f"Bot: {response_text}")
                            messages = []
                            user_query = input("You: ")
                            logger.info(f"User: {user_query}")
                            continue

                        logger.info("No suitable function found. Evaluating for tool creation...")
                        qtype, general_part, _ = await llm.classify_and_split_question(user_query)
                        tool_topic = general_part if qtype == "direct" else user_query
                        proposed_name = predict_tool_name(tool_topic)

                        if proposed_name in existing_tool_names:
                            logger.warning(f"A tool named '{proposed_name}' already exists. LLM failed to call it.")
                            logger.info(f"Bot: {response_text or 'I see a tool that might work, but I had trouble using it. Could you rephrase your request?'}")
                        else:
                            logger.info(f"Bot: Creating a new tool for '{tool_topic}'. Please wait...")
                            extract_info_and_create_tool(query=tool_topic)
                            create_and_update_tool(query=f"Create a function for {tool_topic}")
                            logger.info("Tool created and added to server. Server will now restart.")
                            logger.info("Waiting for reconnection...")
                            await asyncio.sleep(5)

                            if qtype == "direct":
                                query_to_reprocess = user_query
                            break 
                        
                        messages = []
                        user_query = input("You: ")
                        logger.info(f"User: {user_query}")
                    
                    if user_query.lower() == 'quit': return

        except httpx.ReadError:
            logger.warning("Server connection lost (likely restart). Reconnecting in 5s...")
            await asyncio.sleep(5)
        except Exception as e:
            logger.error(f"A session error occurred: {e}", exc_info=True)
            logger.info("Retrying connection in 5 seconds...")
            await asyncio.sleep(5)

if __name__ == "__main__":
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        logger.info("\nClient shutdown requested. Exiting.")