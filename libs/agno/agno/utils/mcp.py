from functools import partial
from uuid import uuid4
from datetime import timedelta

from agno.utils.log import log_debug, log_exception

try:
    from mcp import ClientSession
    from mcp.types import CallToolResult, EmbeddedResource, ImageContent, TextContent
    from mcp.types import Tool as MCPTool
except (ImportError, ModuleNotFoundError):
    raise ImportError("`mcp` not installed. Please install using `pip install mcp`")


from agno.media import ImageArtifact


def get_entrypoint_for_tool(tool: MCPTool, session: ClientSession, timeout_seconds: int = 30):
    """
    Return an entrypoint for an MCP tool.

    Args:
        tool: The MCP tool to create an entrypoint for
        session: The session to use
        timeout_seconds: Timeout in seconds for the tool call

    Returns:
        Callable: The entrypoint function for the tool
    """
    from agno.agent import Agent

    async def call_tool(agent: Agent, tool_name: str, **kwargs) -> str:
        try:
            log_debug(f"Calling MCP Tool '{tool_name}' with args: {kwargs}")
            result: CallToolResult = await session.call_tool(
                tool_name, 
                kwargs, 
                read_timeout_seconds=timedelta(seconds=timeout_seconds)
            )

            # Return an error if the tool call failed
            if result.isError:
                raise Exception(f"Error from MCP tool '{tool_name}': {result.content}")

            # Process the result content
            response_str = ""
            for content_item in result.content:
                if isinstance(content_item, TextContent):
                    response_str += content_item.text + "\n"
                elif isinstance(content_item, ImageContent):
                    # Handle image content if present
                    img_artifact = ImageArtifact(
                        id=str(uuid4()),
                        url=getattr(content_item, "url", None),
                        content=getattr(content_item, "data", None),
                        mime_type=getattr(content_item, "mimeType", "image/png"),
                    )
                    agent.add_image(img_artifact)
                    response_str += "Image has been generated and added to the response.\n"
                elif isinstance(content_item, EmbeddedResource):
                    # Handle embedded resources
                    response_str += f"[Embedded resource: {content_item.resource.model_dump_json()}]\n"
                else:
                    # Handle other content types
                    response_str += f"[Unsupported content type: {content_item.type}]\n"

            return response_str.strip()
        except Exception as e:
            log_exception(f"Failed to call MCP tool '{tool_name}': {e}")
            return f"Error: {e}"

    return partial(call_tool, tool_name=tool.name)


def get_sync_entrypoint_for_tool(tool: MCPTool, session: ClientSession, timeout_seconds: int = 30):
    """
    Return a synchronous entrypoint for an MCP tool.

    Args:
        tool: The MCP tool to create an entrypoint for
        session: The session to use
        timeout_seconds: Timeout in seconds for the tool call

    Returns:
        Callable: The synchronous entrypoint function for the tool
    """
    import asyncio
    from agno.agent import Agent

    def call_tool_sync(agent: Agent, tool_name: str, **kwargs) -> str:
        try:
            log_debug(f"Calling MCP Tool '{tool_name}' with args: {kwargs}")
            log_debug(f"Using timeout: {timeout_seconds} seconds")
            
            # Try to run the async call with better event loop handling
            result = _run_async_with_timeout(
                _async_call_tool(session, tool_name, kwargs, agent, timeout_seconds),
                timeout_seconds
            )
            
            return result
        except Exception as e:
            log_exception(f"Failed to call MCP tool '{tool_name}': {e}")
            return f"Error: {e}"

    return partial(call_tool_sync, tool_name=tool.name)


def _run_async_with_timeout(coro, timeout_seconds: int):
    """
    Run an async coroutine with timeout, handling various event loop scenarios.
    
    This function is designed to work in various environments including:
    - Normal synchronous contexts (no event loop)
    - Async contexts (with running event loop)
    - Debugger environments (PyCharm, etc.)
    - Multi-threaded environments
    
    Args:
        coro: The coroutine to run
        timeout_seconds: Timeout in seconds
        
    Returns:
        The result of the coroutine
        
    Raises:
        TimeoutError: If the operation times out
        Exception: Any other exception from the coroutine
    """
    import asyncio
    import concurrent.futures
    import threading
    
    log_debug(f"Running async operation with {timeout_seconds}s timeout")
    
    # Check if we're in an event loop
    try:
        current_loop = asyncio.get_running_loop()
        log_debug("Event loop detected, running in separate thread")
        # We're in an event loop, need to run in a separate thread
        
        def run_in_thread():
            """Run the coroutine in a new thread with its own event loop."""
            # Create a new event loop for this thread
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                return new_loop.run_until_complete(
                    asyncio.wait_for(coro, timeout=timeout_seconds)
                )
            except asyncio.TimeoutError:
                raise TimeoutError(f"MCP tool call timed out after {timeout_seconds} seconds")
            finally:
                new_loop.close()
                # Clean up the loop from the thread
                asyncio.set_event_loop(None)
        
        # Use ThreadPoolExecutor to run in a separate thread
        with concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="mcp-sync-") as executor:
            future = executor.submit(run_in_thread)
            try:
                # Add a small buffer to the thread timeout to account for overhead
                return future.result(timeout=timeout_seconds + 5)
            except concurrent.futures.TimeoutError:
                # Cancel the future and raise our own timeout error
                future.cancel()
                raise TimeoutError(f"MCP tool call timed out after {timeout_seconds} seconds")
                
    except RuntimeError:
        # No event loop running, we can use asyncio.run directly
        log_debug("No event loop detected, using asyncio.run")
        try:
            return asyncio.run(asyncio.wait_for(coro, timeout=timeout_seconds))
        except asyncio.TimeoutError:
            raise TimeoutError(f"MCP tool call timed out after {timeout_seconds} seconds")


async def _async_call_tool(session: ClientSession, tool_name: str, kwargs: dict, agent, timeout_seconds: int = 30) -> str:
    """Helper function to make the actual async MCP tool call."""
    from datetime import timedelta
    
    # Call the tool with explicit timeout
    result: CallToolResult = await session.call_tool(
        tool_name, 
        kwargs, 
        read_timeout_seconds=timedelta(seconds=timeout_seconds)
    )

    # Return an error if the tool call failed
    if result.isError:
        raise Exception(f"Error from MCP tool '{tool_name}': {result.content}")

    # Process the result content
    response_str = ""
    for content_item in result.content:
        if isinstance(content_item, TextContent):
            response_str += content_item.text + "\n"
        elif isinstance(content_item, ImageContent):
            # Handle image content if present
            img_artifact = ImageArtifact(
                id=str(uuid4()),
                url=getattr(content_item, "url", None),
                content=getattr(content_item, "data", None),
                mime_type=getattr(content_item, "mimeType", "image/png"),
            )
            agent.add_image(img_artifact)
            response_str += "Image has been generated and added to the response.\n"
        elif isinstance(content_item, EmbeddedResource):
            # Handle embedded resources
            response_str += f"[Embedded resource: {content_item.resource.model_dump_json()}]\n"
        else:
            # Handle other content types
            response_str += f"[Unsupported content type: {content_item.type}]\n"

    return response_str.strip()
