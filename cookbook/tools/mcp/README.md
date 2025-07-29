# MCP Agents using Agno

Model Context Protocol (MCP) gives Agents the ability to interact with external systems through a standardized interface. Using Agno's MCP integration, you can build Agents that can connect to any MCP-compatible service.

## Examples in this Directory

1. Filesystem Agent (`filesystem.py`)

This example demonstrates how to create an agent that can explore, analyze, and provide insights about files and directories on your computer.

2. GitHub Agent (`github.py`)

This example shows how to create an agent that can explore GitHub repositories, analyze issues, pull requests, and more.

3. Groq with Llama using MCP (`groq_mcp.py`)

This example uses the file system MCP agent with Groq running the Llama 3.3-70b-versatile model.

4. Include/Exclude Tools (`include_exclude_tools.py`)

This example shows how to include and exclude tools from the MCP agent. This is useful for reducing the number of tools available to the agent, or for focusing on a specific set of tools.

5. Multiple MCP Servers (`multiple_servers.py`)

This example shows how to use multiple MCP servers in the same agent. 

6. Sequential Thinking (`sequential_thinking.py`)

This example shows how to use the MCP agent to perform sequential thinking.

7. Airbnb Agent (`airbnb.py`)

This example shows how to create an agent that uses MCP and Gemini 2.5 Pro to search for Airbnb listings.


## Getting Started

### Prerequisites

Install the required dependencies:

```bash
pip install agno mcp openai
```

Export your API keys:

```bash
export OPENAI_API_KEY="your_openai_api_key"
```

> For the GitHub example, create a Github PAT following [these steps](https://github.com/modelcontextprotocol/servers/tree/main/src/github#setup).

### Run the Examples

```bash
python filesystem.py
python github.py
```

## How It Works

These examples use Agno to create agents that leverage MCP servers. The MCP servers provide standardized access to different data sources (filesystem, GitHub), and the agents use these servers to answer questions and perform tasks.

The workflow is:
1. Agent receives a query from the user
2. Agent determines which MCP tools to use
3. Agent calls the appropriate MCP server to get information
4. Agent processes the information and provides a response

## Customizing

You can modify these examples to:
- Connect to different MCP servers
- Change the agent's instructions
- Add additional tools
- Customize the agent's behavior

## More Information

- Read more about [MCP](https://modelcontextprotocol.io/introduction)
- Read about [Agno's MCP integration](https://docs.agno.com/tools/mcp)

## New: Synchronous MCP Tools

We've added `SyncMCPTools` - a synchronous wrapper around the existing async `MCPTools` that allows you to use MCP servers without `async`/`await`.

### When to Use Each

**Use `SyncMCPTools` when:**
- You're working in a non-async codebase
- You want simpler, more straightforward code
- You're doing quick scripts or interactive sessions
- You don't need concurrent operations

**Use `MCPTools` when:**
- You're in an async environment (web apps, etc.)
- You need maximum performance
- You're doing concurrent operations
- You're already using async/await patterns

### Basic Usage

#### Synchronous (New)
```python
from agno.agent import Agent
from agno.tools.mcp import SyncMCPTools
from mcp import StdioServerParameters

# Set up server
server_params = StdioServerParameters(
    command="npx",
    args=["-y", "@modelcontextprotocol/server-filesystem", "/path/to/files"]
)

# Use sync context manager - no async/await needed!
with SyncMCPTools(server_params=server_params) as mcp_tools:
    agent = Agent(tools=[mcp_tools])
    agent.print_response("List files in current directory")
```

#### Asynchronous (Original)
```python
from agno.agent import Agent  
from agno.tools.mcp import MCPTools
from mcp import StdioServerParameters

async def main():
    server_params = StdioServerParameters(
        command="npx", 
        args=["-y", "@modelcontextprotocol/server-filesystem", "/path/to/files"]
    )
    
    # Use async context manager
    async with MCPTools(server_params=server_params) as mcp_tools:
        agent = Agent(tools=[mcp_tools])
        await agent.aprint_response("List files in current directory")

# Run with asyncio
import asyncio
asyncio.run(main())
```

### Sync Examples in this Directory

- `sync_mcp_example.py` - Simple example using SyncMCPTools
- `sync_vs_async_comparison.py` - Side-by-side comparison of both approaches

### API Compatibility

`SyncMCPTools` supports all the same parameters as `MCPTools`:

- `command` - Command to start MCP server
- `url` - URL for SSE/HTTP transport  
- `transport` - Protocol ("stdio", "sse", "streamable-http")
- `server_params` - Pre-configured server parameters
- `timeout_seconds` - Timeout for operations
- `include_tools`/`exclude_tools` - Filter available tools

The only difference is the context manager pattern:
- `SyncMCPTools`: `with SyncMCPTools() as tools:`
- `MCPTools`: `async with MCPTools() as tools:`
