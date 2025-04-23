# prass
Prompting As A Service 

🌟 Model Context Protocol (MCP) Server

📚 Overview
Welcome to our Model Context Protocol (MCP) implementation! This project provides a simple yet powerful MCP server that enables function calling capabilities across different LLM providers. By standardizing the interface between models and function calling, we make it easier to harness the power of AI agents regardless of your chosen model.

╭─────────────╮     ╭───────────────╮     ╭─────────────╮
│             │     │               │     │             │
│  LLM Client ├─────┤  MCP Server   ├─────┤  Functions  │
│             │     │               │     │             │
╰─────────────╯     ╰───────────────╯     ╰─────────────╯
   (Llama,            (Handles           (Add, Multiply,
    Gemini)         API requests)          and more!)


✨ Features

🔢 Built-in mathematical functions: Add and Multiply
🔄 OpenAI-compatible function calling format
🤖 Pre-configured clients for Llama (via Groq) and Gemini
🚀 Easy to extend with your own functions and clients
🛠️ Simple API for integrating with any LLM

agentdry/
|   __init__.py 
├── servers/
│   ├── mcp_trial_server.py  # The core MCP server implementation
│
├── clients/
│   ├── client_mcp.py  # Google's Gemini client
│   ├── groq_client_mcp.py # Groq-powered Llama client
│   └── ...
└── formatters/
    ├── openai_format_check.py  # OpenAI-compatible format converter
    └── ...

