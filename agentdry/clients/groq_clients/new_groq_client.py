import asyncio
import logging
import json
import os
import re
import sys
import httpx
from dotenv import load_dotenv
from groq import Groq, APIError

from mcp import ClientSession
from mcp.client.sse import sse_client

# Add parent directory to system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from utils.openai_format_check import transform_schema
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
        logging.FileHandler(os.path.join(LOG_DIR, "groq_client.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Load Environment and Check API Key ---
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    logger.critical("GROQ_API_KEY environment variable not found. Exiting.")
    sys.exit(1)


class GroqLLM:
    """A wrapper class for Groq API chat completion interactions."""
    def __init__(self, api_key):
        self.client = Groq(api_key=api_key)
        self.model_name = "llama3-8b-8192" # A reliable and fast model for these tasks

    async def _generate_non_tool(self, prompt, temperature=0.0):
        """Helper for making chat completion calls that DON'T require tools."""
        messages = [{"role": "user", "content": prompt}]
        try:
            response = self.client.chat.completions.create(
                messages=messages,
                model=self.model_name,
                temperature=temperature,
            )
            return response.choices[0].message.content.strip()
        except APIError as e:
            logger.error(f"Groq API error during non-tool generation: {e}")
            return "An internal error occurred while processing your request."

    async def get_model_response_with_tools(self, messages, tools):
        """Generates a response from the LLM, potentially using tools."""
        # This method should only be called if `determine_tool_creation_necessity`
        # has indicated that a tool might be needed.
        try:
            return self.client.chat.completions.create(
                messages=messages,
                model=self.model_name,
                tools=tools,
                tool_choice="auto",
                temperature=0.0
            )
        except APIError as e:
            # Re-raise the APIError so process_query can handle it specifically
            # when the LLM generates a malformed tool call
            raise e 

    async def determine_tool_creation_necessity(self, query_text):
        """Determines if a query requires a new tool or can be answered directly."""
        prompt = (
            "Analyze the user query. Respond with 'TOOL_NEEDED' if a specialized function is "
            "required to fulfill the request, such as calculations, data retrieval, or complex operations. "
            "Respond with 'NO_TOOL_NEEDED' if the query can be answered directly by a conversational AI, "
            "is a simple greeting, a question about general knowledge, or requires no specific tool functionality.\n"
            "Input: Create a script to calculate Fibonacci series\nOutput: TOOL_NEEDED\n"
            "Input: What is the capital of France?\nOutput: NO_TOOL_NEEDED\n"
            "Input: Hi there!\nOutput: NO_TOOL_NEEDED\n"
            "Input: How are you doing?\nOutput: NO_TOOL_NEEDED\n"
            "Input: Tell me a joke\nOutput: NO_TOOL_NEEDED\n"
            "Input: Calculate 5 plus 3\nOutput: TOOL_NEEDED\n"
            "Input: Search for Python tutorials on a specific topic\nOutput: TOOL_NEEDED\n"
            "Input: What is the circumference of a circle with radius 6?\nOutput: TOOL_NEEDED\n"
            f"Input: {query_text}\nOutput:"
        )
        response_text = await self._generate_non_tool(prompt)
        return "TOOL_NEEDED" in response_text


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
            "Input: what is the circumference of a circle with radius 6\n"
            "Output: direct | Calculate circumference of a circle | with radius 6\n"
            "Input: find the factorial of 3\n"
            "Output: direct | Find factorial of a number | find the factorial of 3\n"
            "Input: check if 7 is a prime number\n"
            "Output: direct | Check if a number is prime | check if 7 is a prime number\n"
            "Input: find if 6 is a prime or not\n"
            "Output: direct | Check if a number is prime | find if 6 is a prime or not\n"
            "Input: Is 10 an odd or even number?\n"
            "Output: direct | Check if a number is odd or even | Is 10 an odd or even number?\n"
            "Input: Find if 99 is an odd or even number\n"
            "Output: direct | Check if a number is odd or even | Find if 99 is an odd or even number\n"
            "Input: find the perimeter of a square\n"
            "Output: general | Calculate perimeter of a square | \n"
            "Input: find the perimeter of a square with side 9\n"
            "Output: direct | Calculate perimeter of a square | with side 9\n"
            "Input: what is the area of a square\n" # NEW
            "Output: general | Calculate area of a square | \n"
            "Input: what is the area of a square sided 3\n" # NEW
            "Output: direct | Calculate area of a square | what is the area of a square sided 3\n"
            f"Input: {question}\n"
            "Output:"
        )
        text = await self._generate_non_tool(prompt)
        match = re.match(r"(general|direct)\s*\|\s*(.*?)\s*\|\s*(.*)", text, re.IGNORECASE)
        if match:
            qtype, general, specific = match.groups()
            return qtype.lower(), general.strip(), specific.strip()
        logger.warning("Failed to classify question, defaulting to 'general'.")
        return "general", question, ""


async def load_tools(session):
    """Loads and formats tools from the MCP server for the Groq API."""
    tools_response = await session.list_tools()
    return [
        {
            'type': 'function',
            'function': {
                "name": tool.name,
                "description": tool.description,
                "parameters": transform_schema(tool.inputSchema),
            }
        }
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

async def process_query(session, llm, tools, query_text, messages_history):
    """
    Processes a single query, handling the API call and tool execution.
    Returns (tool_was_called_successfully, updated_messages_history, final_bot_response_text).
    """
    messages_history.append({"role": "user", "content": query_text})
    
    response = None
    try:
        response = await llm.get_model_response_with_tools(messages_history, tools)
    except APIError as e:
        logger.error(f"Groq API call failed during tool-enabled response: {e}")
        
        error_message_from_api = str(e) # Use str(e) for robust error message
        
        # Specific check for "tool was not request.tools"
        if "tool call validation failed: attempted to call tool" in error_message_from_api and "which was not request.tools" in error_message_from_api:
            # This means LLM tried to call a non-existent tool. We should proceed to create a new tool.
            error_msg = "The model attempted to use a tool that is not available."
            messages_history.append({"role": "assistant", "content": error_msg})
            # Return a message that does NOT contain "internal error" to trigger tool creation flow
            return False, messages_history, "The model attempted to use a tool that is not available, which may require creating a new tool."
        else:
            # For other API errors (e.g., malformed parameters for an *existing* tool)
            error_msg = f"I encountered an internal error: {error_message_from_api}. Could you please rephrase?"
            messages_history.append({"role": "assistant", "content": error_msg})
            return False, messages_history, error_msg


    if not response or not response.choices:
        return False, messages_history, "Sorry, I encountered an API error or no response from the model."

    response_message = response.choices[0].message
    messages_history.append(response_message)

    if response_message.tool_calls:
        # Assuming only one tool call per turn for simplicity, or iterate if multiple are supported
        tool_call = response_message.tool_calls[0] 
        function_name = tool_call.function.name
        tool_call_id = tool_call.id # Ensure tool_call_id is captured here

        try:
            function_args = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError:
            logger.error(f"Failed to decode JSON arguments for {function_name}: {tool_call.function.arguments}")
            tool_error_content = "Failed to parse tool arguments. Arguments were not valid JSON."
            messages_history.append({"role": "tool", "tool_call_id": tool_call_id, "content": tool_error_content})
            
            # After logging the tool error, get a natural language response from the LLM about it
            follow_up_response = await llm.get_model_response_with_tools(messages_history, tools)
            final_bot_text = follow_up_response.choices[0].message.content if follow_up_response and follow_up_response.choices else tool_error_content
            messages_history.append({"role": "assistant", "content": final_bot_text})
            return False, messages_history, final_bot_text
            
        logger.info(f"LLM wants to call function: {function_name} with args: {function_args}")
        try:
            tool_result = await session.call_tool(function_name, arguments=function_args)
            tool_output = tool_result.content[0].text if tool_result and tool_result.content else ""
            logger.info(f"Bot (tool output): {tool_output}")
            messages_history.append({"role": "tool", "tool_call_id": tool_call_id, "content": tool_output})
            
            # After a tool call, send the updated messages back to the LLM to get a natural language response
            follow_up_response = await llm.get_model_response_with_tools(messages_history, tools)
            if follow_up_response and follow_up_response.choices and follow_up_response.choices[0].message.content:
                final_bot_text = follow_up_response.choices[0].message.content
                messages_history.append({"role": "assistant", "content": final_bot_text})
                return True, messages_history, final_bot_text
            else:
                return True, messages_history, "Tool executed successfully, but I don't have a specific follow-up message."
        except Exception as e:
            logger.error(f"Error calling MCP tool '{function_name}': {e}", exc_info=True)
            tool_error_content = f"Error executing tool '{function_name}': {str(e)}"
            messages_history.append({"role": "tool", "tool_call_id": tool_call_id, "content": tool_error_content})
            # After logging the tool execution error, get a natural language response from the LLM about it
            follow_up_response = await llm.get_model_response_with_tools(messages_history, tools)
            final_bot_text = follow_up_response.choices[0].message.content if follow_up_response and follow_up_response.choices else tool_error_content
            messages_history.append({"role": "assistant", "content": final_bot_text})
            return False, messages_history, final_bot_text
    else:
        # If no tool call was made, the response is plain text
        bot_text = response_message.content
        return False, messages_history, bot_text


async def run():
    """The main application loop to handle user interaction and tool management."""
    llm = GroqLLM(api_key=GROQ_API_KEY)
    query_to_reprocess = None

    while True:
        try:
            async with sse_client(url="http://localhost:8000/sse") as streams:
                async with ClientSession(*streams) as session:
                    await session.initialize()
                    
                    tools = await load_tools(session)
                    logger.info("--- Available Tools on Server ---")
                    if tools:
                        for tool in tools:
                            func = tool['function']
                            logger.info(f"- Name: {func['name']}, Desc: {func['description']}")
                    else:
                        logger.info("No tools currently available.")
                    logger.info("---------------------------------")
                    
                    existing_tool_names = {t['function']['name'].lower() for t in tools}
                    messages = [] 

                    if query_to_reprocess:
                        logger.info(f"Automatically re-processing query: {query_to_reprocess}")
                        user_query = query_to_reprocess
                        query_to_reprocess = None
                    else:
                        user_query = input("You: ")
                        logger.info(f"User: {user_query}")

                    while user_query.lower() != "quit":
                        # **NEW: Hard-coded check for simple greetings/conversational inputs**
                        simple_greetings = ["hi", "hello", "hey", "how are you", "what's up", "good morning", "good afternoon", "good evening"]
                        if user_query.lower().strip() in simple_greetings:
                            bot_response = await llm._generate_non_tool(user_query)
                            logger.info(f"Bot: {bot_response}")
                            messages = [] # Clear messages as this was a simple conversational turn
                            user_query = input("You: ")
                            logger.info(f"User: {user_query}")
                            continue # Skip the rest of the loop and go to next input

                        # Original flow: Determine if a tool is needed for the query type
                        necessity = await llm.determine_tool_creation_necessity(user_query)

                        if not necessity: # If NO_TOOL_NEEDED (by LLM classification)
                            bot_response = await llm._generate_non_tool(user_query)
                            logger.info(f"Bot: {bot_response}")
                            messages = [] # Clear messages as this was a simple conversational turn
                        else: # If TOOL_NEEDED, then proceed with tool processing
                            tool_called_successfully, messages, response_text = await process_query(session, llm, tools, user_query, messages)

                            if tool_called_successfully:
                                logger.info(f"Bot: {response_text}")
                            else:
                                logger.info(f"Bot: {response_text}") 

                                # This condition now specifically checks for "internal error" or "Error executing tool"
                                # to distinguish from "tool not available" case, which should trigger creation.
                                if "I encountered an internal error" in response_text or "Error executing tool" in response_text:
                                    logger.warning("Tool execution error detected. Not attempting new tool creation for this query.")
                                    pass 
                                else:
                                    # This path is taken if no tool was called successfully AND it wasn't a recognized execution error.
                                    # This includes the "tool not available" scenario, triggering tool creation.
                                    logger.info("No suitable function found (or LLM chose not to call). Evaluating for tool creation...")
                                    qtype, general_part, _ = await llm.classify_and_split_question(user_query)
                                    tool_topic = general_part if qtype == "direct" else user_query
                                    proposed_name = predict_tool_name(tool_topic)

                                    if proposed_name in existing_tool_names:
                                        logger.warning(f"A tool named '{proposed_name}' already exists. LLM failed to call it or it was not applicable.")
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
                                            # Do NOT break for direct questions, allow the loop to re-process
                                            # This will lead to the "if query_to_reprocess:" block being hit.
                                        else: # qtype is "general"
                                            break # Only break if it's a general question requiring just tool creation
                        
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