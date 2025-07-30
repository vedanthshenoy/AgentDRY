from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from google import genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

from dotenv import load_dotenv
import os

load_dotenv()

# client = genai.Client(
#     api_key=os.getenv("GEMINI_API_KEY")
# )
# response = client.models.generate_content(
#                 model="gemini-2.0-flash",  # Or your preferred model supporting function calling
#                 contents="Can you write a python function add 2 and 3")

# print(response)


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",#"gemini-1.5-pro-002",
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)
"""
Query = Create a tool for subtracting two numbers

Return : 
    
    
    @mcp.tool()
    def subtract(a,b):
    "Subtracts two numbers"
    return a-b
"""

class Response(BaseModel):
    code : str = Field("The actual python function from def to return")
    example : str = Field("Example input and output of the generated code")
    description : str = Field("Description of the code")
    
code_parser = PydanticOutputParser(pydantic_object = Response)

prompt = PromptTemplate(template = """
                        {format_instructions}
                        User Input : Write a python function for this query {query}. Start directly from def keyword. Never provide any heading or headers or source language. Add type hints and description.
                        """, input_variables = ["query"], partial_variables = {"format_instructions" : code_parser.get_format_instructions()})

obtain_code_chain = prompt | llm | code_parser



if __name__ == "__main__":
        
    response = obtain_code_chain.invoke("Generate a python function to subtract two numbers")
    # print(type(response.code))
    code_string = "@mcp.tool()\n" + response.code
    print(code_string)
    