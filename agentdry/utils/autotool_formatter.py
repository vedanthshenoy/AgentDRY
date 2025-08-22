from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from google import genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",#"gemini-1.5-pro-002",
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

class Response(BaseModel):
    code : str = Field("The actual python function from def to return")
    example : str = Field("Example input and output of the generated code")
    description : str = Field("Description of the code")
    
code_parser = PydanticOutputParser(pydantic_object = Response)

prompt = PromptTemplate(template = """
                        {format_instructions}
                        
                        User Input: {query}
                        
                        Create a GENERALIZED Python function that can handle this type of request. 
                        
                        Important guidelines:
                        1. If the query asks for a specific calculation (like "factorial of 5"), create a general function (like "calculate factorial of any number") 
                        2. If the query asks for specific data (like "weather in New York"), create a general function (like "get weather for any city")
                        3. Make the function reusable with parameters
                        4. Start directly from the 'def' keyword
                        5. Include type hints and a clear docstring
                        6. Never provide headings, headers, or language indicators
                        7. The function should be generic enough to handle similar requests
                        
                        Example transformations:
                        - "What is the factorial of 5?" → Create function: calculate_factorial(n: int)
                        - "Convert 100 USD to EUR" → Create function: convert_currency(amount: float, from_currency: str, to_currency: str)
                        - "What's the weather in Paris?" → Create function: get_weather(city: str)
                        
                        """, input_variables = ["query"], partial_variables = {"format_instructions" : code_parser.get_format_instructions()})

obtain_code_chain = prompt | llm | code_parser

if __name__ == "__main__":
        
    response = obtain_code_chain.invoke("What is the factorial of 5?")
    # print(type(response.code))
    code_string = "@mcp.tool()\n" + response.code
    print(code_string)
    print(f"\nDescription: {response.description}")
    print(f"Example: {response.example}")