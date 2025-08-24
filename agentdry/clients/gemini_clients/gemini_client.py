import asyncio
import logging
import os
import sys
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx
from dotenv import load_dotenv
from google import generativeai as genai

from mcp import ClientSession
from mcp.client.sse import sse_client

# Simple logging setup
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv()


class ConnectionManager:
    """Manages MCP server connections."""

    def __init__(self, server_url: str = "http://localhost:8000/sse"):
        self.server_url = server_url

    async def check_server_health(self) -> bool:
        """Check if the MCP server is responding."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.head(self.server_url)
                return response.status_code == 200
        except httpx.RequestError as e:
            logger.warning(f"Server health check failed: {e}")
            return False

    @asynccontextmanager
    async def get_session(self):
        """Get an MCP session."""
        try:
            streams = sse_client(url=self.server_url)
            async with streams as stream_pair:
                session = ClientSession(*stream_pair)
                async with session:
                    await asyncio.wait_for(session.initialize(), timeout=10.0)
                    yield session
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            raise


class GeminiClient:
    """Gemini API client."""

    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required")

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-1.5-flash")

    async def generate_response(self, prompt: str, temperature: float = 0.0) -> str:
        """Generate a response."""
        try:
            response = await asyncio.to_thread(
                self.model.generate_content,
                prompt,
                generation_config={"temperature": temperature},
            )
            return response.text.strip()
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return "I'm sorry, I couldn't generate a response. Please try again."


class ConversationManager:
    """Manages conversation history."""

    def __init__(self, max_history: int = 20):
        self.max_history = max_history
        self.history: List[Dict[str, str]] = []

    def add_message(self, role: str, content: str):
        """Add a message to history."""
        self.history.append({"role": role, "content": content})
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history :]

    def get_gemini_history(self) -> List[Dict[str, Any]]:
        """Convert history to Gemini API format."""
        return [
            {
                "role": "model" if msg["role"] == "assistant" else msg["role"],
                "parts": [{"text": msg["content"]}],
            }
            for msg in self.history
        ]

    def clear(self):
        """Clear conversation history."""
        self.history.clear()


class GeminiMCPClient:
    """Main client class for handling communication and conversation management."""

    def __init__(
        self,
        api_key: str,
        server_url: str = "http://localhost:8000/sse",
    ):
        self.llm = GeminiClient(api_key)
        self.connection_manager = ConnectionManager(server_url)
        self.conversation = ConversationManager()
        self.server_available = False

    async def initialize_connection(self) -> bool:
        """Initialize the client connection."""
        try:
            self.server_available = await self.connection_manager.check_server_health()
            return True
        except Exception as e:
            logger.error(f"Client initialization failed: {e}")
            return False

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get conversation history."""
        return self.conversation.history.copy()

    def clear_conversation(self):
        """Clear conversation history."""
        self.conversation.clear()

    async def generate_direct_response(self, query: str) -> str:
        """Generate a direct response without tools."""
        try:
            self.conversation.add_message("user", query)
            response = await self.llm.generate_response(f"Please answer this query: {query}")
            self.conversation.add_message("assistant", response)
            return response
        except Exception as e:
            logger.error(f"Error generating direct response: {e}")
            return "Sorry, I encountered an error generating a response."

    def is_server_available(self) -> bool:
        """Check if server is available."""
        return self.server_available


# Enhanced CLI interface
async def main():
    """Basic CLI interface for the client."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY not found")
        return

    client = GeminiMCPClient(api_key)

    if not await client.initialize_connection():
        print("Failed to initialize client")
        return

    print("🤖 Gemini MCP Client ready!")
    print("Type 'quit' to exit, 'clear' to clear history")

    while True:
        try:
            user_input = input("\nYou: ").strip()

            if user_input.lower() == "quit":
                break
            elif user_input.lower() == "clear":
                client.clear_conversation()
                print("Conversation history cleared.")
                continue

            if not user_input:
                continue

            # Generate direct response
            response = await client.generate_direct_response(user_input)
            print(f"Assistant: {response}")

        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Unexpected error: {e}")

    print("Goodbye!")


if __name__ == "__main__":
    asyncio.run(main())