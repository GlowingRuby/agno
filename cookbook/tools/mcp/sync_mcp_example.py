"""
Example of using SyncMCPTools - the synchronous version of MCPTools.

This example demonstrates how to use MCP servers without async/await.
"""

import sys
from pathlib import Path

from agno.agent import Agent
from agno.tools.mcp import SyncMCPTools
from mcp import StdioServerParameters


def main(prompt: str) -> None:
    """
    Synchronous example using SyncMCPTools.
    
    This demonstrates how to use MCP tools in a non-async context.
    """
    # Initialize the MCP server parameters
    server_params = StdioServerParameters(
        command="npx",
        args=[
            "-y",
            "@modelcontextprotocol/server-filesystem",
            str(Path(__file__).parent.parent.parent),  # Root of the agno project
        ],
    )
    
    # Use the synchronous MCP toolkit
    with SyncMCPTools(server_params=server_params) as mcp_tools:
        # Create an agent with the sync MCP toolkit
        agent = Agent(tools=[mcp_tools])
        
        # Use the agent synchronously (no await needed)
        response = agent.print_response(prompt, stream=True)
        return response


if __name__ == "__main__":
    prompt = (
        sys.argv[1] if len(sys.argv) > 1 else "Read and summarize the file ./LICENSE"
    )
    main(prompt)