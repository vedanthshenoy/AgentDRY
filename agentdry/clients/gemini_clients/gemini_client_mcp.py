# Create server parameters for stdio connection
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import sys
import os

# Add the 'agentdry' directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from utils.openai_format_check import transform_schema, parse_response
from utils.initiate_autotool_creation import extract_info_and_create_tool
from main import create_and_update_tool

import sys, asyncio
from google.genai import types
from google import genai
import os
from dotenv import load_dotenv

load_dotenv()

from mcp import ClientSession
from mcp.client.sse import sse_client

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

async def load_tools(session):
    tools_response = await session.list_tools()
    return [
        {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.inputSchema,
        }
        for tool in tools_response.tools
    ]

async def run():
    while True:
        try:
            async with sse_client(url="http://localhost:8000/sse") as streams:
                async with ClientSession(*streams) as session:
                    await session.initialize()
                    tool_list = await load_tools(session)
                    tools = types.Tool(function_declarations=tool_list)
                    messages = []
                    while True:
                        user_query = input("You: ")
                        if user_query.lower() == "quit":
                            return
                        # Gemini expects a list of Content objects for chat
                        user_content = types.Content(parts=[{"text": user_query}], role="user")
                        messages.append(user_content)

                        response = client.models.generate_content(
                            model="gemini-2.0-flash",
                            contents=messages,
                            config=types.GenerateContentConfig(
                                temperature=0.7,
                                tools=[tools],
                            ),
                        )

                        # Check for a function call
                        candidate = response.candidates[0]
                        parts = candidate.content.parts
                        if parts and hasattr(parts[0], "function_call") and parts[0].function_call:
                            function_call = parts[0].function_call
                            print(f"Function to call: {function_call.name}")
                            print(f"Arguments: {function_call.args}")
                            result = await session.call_tool(function_call.name, arguments=function_call.args)
                            tool_result_text = result.content[0].text if result and result.content and result.content[0].text else ""
                            print("Bot:", tool_result_text)
                            # Add tool response to messages for context
                            tool_content = types.Content(parts=[{"text": tool_result_text}], role="tool")
                            messages.append(tool_content)
                        else:
                            # Text response or no function call
                            bot_text = candidate.content.parts[0].text if parts and hasattr(parts[0], "text") else ""
                            print("Bot:", bot_text)
                            print("No function call found in the response.")
                            print("Creating tool for the query....")
                            topic = extract_info_and_create_tool(query=user_query)
                            create_and_update_tool(query=f"Create a function for {topic}")
                            print("Waiting for server to update tools...")
                            await asyncio.sleep(3)
                            try:
                                tool_list = await load_tools(session)
                                tools = types.Tool(function_declarations=tool_list)
                                print("Tools reloaded. Current tools:")
                                for tool in tool_list:
                                    print(f"- {tool['name']}: {tool['description']}")
                                print("You can now ask your question again.")
                            except Exception as e:
                                print(f"Message from server : I am ready now")
                                break  # Break out to outer loop to reconnect session
        except Exception as e:
            print(f"Session error or server not available: {e}")
            print("Retrying connection in 2 seconds...")
            await asyncio.sleep(2)

if __name__ == "__main__":
    asyncio.run(run())