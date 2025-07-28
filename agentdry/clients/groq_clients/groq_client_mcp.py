# from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()


# llm = ChatGroq(
#     model="llama-3.1-8b-instant",
#     temperature=0,
#     max_tokens=None,
#     timeout=None,
#     max_retries=2,
#     groq_api_key = os.getenv("GROK_API_KEY")
#     # other params...
# )

# print(llm.invoke("Hi how are you"))

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pydantic import BaseModel, Field
import asyncio
from mcp import ClientSession
from mcp.client.sse import sse_client
import sys
import os

# Add the 'agentdry' directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from utils.openai_format_check import transform_schema, parse_response
from utils.initiate_autotool_creation import extract_info_and_create_tool
from main import create_and_update_tool

from groq import Groq
from dotenv import load_dotenv
load_dotenv()

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)


# ...existing imports and setup...

async def load_tools(session):
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

async def run():
    while True:
        try:
            async with sse_client(url="http://localhost:8000/sse") as streams:
                async with ClientSession(*streams) as session:
                    await session.initialize()
                    tools = await load_tools(session)
                    messages = []
                    while True:
                        user_query = input("You: ")
                        if user_query.lower() == "quit":
                            return
                        messages.append({"role": "user", "content": user_query})

                        response = client.chat.completions.create(
                            messages=messages,
                            model="llama-3.3-70b-versatile",
                            temperature=0,
                            tools=tools
                        )

                        result = parse_response(response)
                        messages.append({"role": "assistant", "content": str(result)})

                        if type(result) == dict:
                            print("Calling MCP tool", result["function_name"])
                            tool_result = await session.call_tool(result['function_name'], arguments=result['function_args'])
                            print("Bot:", tool_result.content[0].text)
                            messages.append({"role": "tool", "tool_call_id": result.get("id", "tool_call_id"), "content": tool_result.content[0].text})
                        else:
                            print("Bot:", result)
                            print("No function call found in the response.")
                            print("Creating tool for the query....")
                            topic = extract_info_and_create_tool(query=user_query)
                            create_and_update_tool(query=f"Create a function for {topic}")
                            print("Waiting for server to update tools...")
                            await asyncio.sleep(3)  # <-- Delay to allow server to restart and register new tool
                            try:
                                tools = await load_tools(session)
                                print("Tools reloaded. Current tools:")
                                for tool in tools:
                                    func = tool['function']
                                    print(f"- {func['name']}: {func['description']}")
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