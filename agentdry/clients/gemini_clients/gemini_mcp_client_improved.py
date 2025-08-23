import asyncio
import logging
import os
import sys
from typing import Optional
import json

from dotenv import load_dotenv
from mcp_bridge_service import MCPBridgeService

load_dotenv()

# Simple logging setup
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class GeminiMCPClient:
    """
    Enhanced Gemini MCP Client using the bridge service.
    Now supports dynamic tool creation and intelligent query classification.
    """
    
    def __init__(self, api_key: str, server_url: str = "http://localhost:8000/sse"):
        self.bridge = MCPBridgeService(api_key, server_url)
        self.conversation_history = []
        
    async def initialize(self) -> bool:
        """Initialize the client."""
        success = await self.bridge.initialize()
        if success:
            status = await self.bridge.get_server_status()
            logger.info(f"Client initialized - Server context: {status['server_context']}, Tools: {status['available_tools']}")
        return success
    
    async def process_query(self, query: str) -> Optional[dict]:
        """Process a query and return detailed response information."""
        if not query.strip():
            return None
            
        # Submit query to bridge service
        query_id = await self.bridge.submit_query(query)
        
        # Wait for result with progress indication
        dots = 0
        print("Processing", end="", flush=True)
        
        while True:
            result = await self.bridge.get_query_result(query_id)
            
            if result:
                print()  # New line after progress dots
                
                # Store in conversation history
                conversation_entry = {
                    "user": query,
                    "assistant": result.response,
                    "tools_used": result.tools_called if result.tool_used else [],
                    "tool_created": result.tool_created,
                    "success": result.success,
                    "processing_time": result.processing_time
                }
                self.conversation_history.append(conversation_entry)
                
                # Log interesting events
                if result.tool_used and result.tools_called:
                    logger.info(f"Tools used: {', '.join(result.tools_called)} with args: {getattr(result, 'tool_args', 'N/A')}")

                
                if result.tool_created:
                    logger.info("🎉 New tool was created for this query!")
                
                return conversation_entry
            
            # Show progress
            print("." * (dots % 4), end="\r", flush=True)
            dots += 1
            await asyncio.sleep(0.2)
    
    async def get_server_status(self):
        """Get detailed server status."""
        return await self.bridge.get_server_status()
    
    def clear_conversation(self):
        """Clear conversation history."""
        self.conversation_history.clear()
        
    def show_conversation_stats(self):
        """Show statistics about the conversation."""
        if not self.conversation_history:
            print("No conversation history yet.")
            return
            
        total_queries = len(self.conversation_history)
        tool_uses = sum(1 for entry in self.conversation_history if entry["tools_used"])
        tools_created = sum(1 for entry in self.conversation_history if entry["tool_created"])
        avg_time = sum(entry["processing_time"] or 0 for entry in self.conversation_history) / total_queries
        
        print(f"\n📊 Conversation Statistics:")
        print(f"   Total queries: {total_queries}")
        print(f"   Queries using tools: {tool_uses}")
        print(f"   New tools created: {tools_created}")
        print(f"   Average processing time: {avg_time:.2f}s")
        
        # Show unique tools used
        all_tools = set()
        for entry in self.conversation_history:
            all_tools.update(entry["tools_used"])
        
        if all_tools:
            print(f"   Tools used: {', '.join(sorted(all_tools))}")

    async def shutdown(self):
        """Shutdown the client."""
        await self.bridge.shutdown()


async def main():
    """Enhanced CLI interface."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ Error: GEMINI_API_KEY not found in environment")
        print("Please set your Gemini API key in the .env file")
        return

    client = GeminiMCPClient(api_key)

    if not await client.initialize():
        print("❌ Failed to initialize client")
        return

    print("🤖 Enhanced Gemini MCP Client ready!")
    print("✨ Now with intelligent query classification and dynamic tool creation!")
    
    status = await client.get_server_status()
    print(f"🔧 Server context: {status['server_context']}")
    print(f"🛠️  Available tools: {status['available_tools']}")
    
    if status['tool_names']:
        print(f"📋 Tool names: {', '.join(status['tool_names'])}")
    
    print("\nCommands:")
    print("  'quit' or 'exit' - Exit the client")
    print("  'status' - Show server status") 
    print("  'clear' - Clear conversation history")
    print("  'stats' - Show conversation statistics")
    print("  'help' - Show this help message")

    while True:
        try:
            user_input = input("\n💬 You: ").strip()

            if user_input.lower() in ["quit", "exit"]:
                break
            elif user_input.lower() == "clear":
                client.clear_conversation()
                print("🧹 Conversation history cleared.")
                continue
            elif user_input.lower() == "status":
                status = await client.get_server_status()
                print(f"\n📡 Server Status:")
                print(f"   Available: {'✅' if status['server_available'] else '❌'}")
                print(f"   Context: {status['server_context']}")
                print(f"   Tools: {status['available_tools']}")
                if status['tool_names']:
                    print(f"   Tool names: {', '.join(status['tool_names'])}")
                print(f"   Active queries: {status['active_queries']}")
                print(f"   Completed queries: {status['completed_queries']}")
                continue
            elif user_input.lower() == "stats":
                client.show_conversation_stats()
                continue
            elif user_input.lower() == "help":
                print("\nCommands:")
                print("  'quit' or 'exit' - Exit the client")
                print("  'status' - Show server status") 
                print("  'clear' - Clear conversation history")
                print("  'stats' - Show conversation statistics")
                print("  'help' - Show this help message")
                continue

            if not user_input:
                continue

            result = await client.process_query(user_input)
            if result:
                print(f"\n🤖 Assistant: {result['assistant']}")
                
                # Show additional info for interesting responses
                if result['tools_used']:
                    print(f"🔧 Tools used: {', '.join(result['tools_used'])}")
                
                if result['tool_created']:
                    print("🎉 A new tool was created to handle your request!")
                
                if result['processing_time']:
                    print(f"⏱️  Processing time: {result['processing_time']:.2f}s")

        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            print(f"❌ An error occurred: {e}")

    print("\n👋 Shutting down...")
    client.show_conversation_stats()  # Show final stats
    await client.shutdown()
    print("Goodbye!")


if __name__ == "__main__":
    asyncio.run(main())