"""
Comprehensive example comparing async MCPTools vs sync SyncMCPTools.

This demonstrates the difference between the two approaches and shows
when to use each one.
"""

import asyncio
import sys
from pathlib import Path

from agno.agent import Agent
from agno.tools.mcp import MCPTools, SyncMCPTools
from mcp import StdioServerParameters


def sync_example(prompt: str) -> None:
    """
    Example using SyncMCPTools - no async/await needed.
    
    Perfect for:
    - Scripts that don't use asyncio
    - Integration with synchronous codebases  
    - Simple one-off tasks
    - Interactive sessions
    """
    print("=== Synchronous Example ===")
    
    # Set up MCP server parameters
    server_params = StdioServerParameters(
        command="npx",
        args=[
            "-y", 
            "@modelcontextprotocol/server-filesystem",
            str(Path(__file__).parent.parent.parent),  # agno root
        ],
    )
    
    # Use sync MCP tools with context manager
    with SyncMCPTools(server_params=server_params) as mcp_tools:
        print(f"Initialized {len(mcp_tools.functions)} MCP tools")
        
        # Create agent and run synchronously
        agent = Agent(tools=[mcp_tools])
        agent.print_response(prompt, stream=False)


async def async_example(prompt: str) -> None:
    """
    Example using async MCPTools - requires async/await.
    
    Perfect for:
    - Async web applications 
    - Integration with async frameworks
    - Concurrent operations
    - Performance-critical applications
    """
    print("\n=== Asynchronous Example ===")
    
    server_params = StdioServerParameters(
        command="npx",
        args=[
            "-y",
            "@modelcontextprotocol/server-filesystem", 
            str(Path(__file__).parent.parent.parent),
        ],
    )
    
    # Use async MCP tools with async context manager
    async with MCPTools(server_params=server_params) as mcp_tools:
        print(f"Initialized {len(mcp_tools.functions)} MCP tools")
        
        # Create agent and run asynchronously
        agent = Agent(tools=[mcp_tools])
        await agent.aprint_response(prompt, stream=False)


def main():
    """Main function showing both approaches."""
    prompt = (
        sys.argv[1] if len(sys.argv) > 1 
        else "List the files in the current directory"
    )
    
    print("Comparing Sync vs Async MCP Tools\n")
    
    try:
        # Run synchronous example
        sync_example(prompt)
        
        # Run asynchronous example  
        asyncio.run(async_example(prompt))
        
        print("\n=== Summary ===")
        print("✓ SyncMCPTools: Use when you don't want async/await")
        print("✓ MCPTools: Use in async environments for better performance")
        print("Both provide the same MCP functionality!")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Note: This example requires an MCP server to be available.")


if __name__ == "__main__":
    main()