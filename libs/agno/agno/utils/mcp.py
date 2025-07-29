from functools import partial
from uuid import uuid4

from agno.utils.log import log_debug, log_exception

try:
    from mcp import ClientSession
    from mcp.types import CallToolResult, EmbeddedResource, ImageContent, TextContent
    from mcp.types import Tool as MCPTool
except (ImportError, ModuleNotFoundError):
    raise ImportError("`mcp` not installed. Please install using `pip install mcp`")


from agno.media import ImageArtifact


def get_entrypoint_for_tool(tool: MCPTool, session: ClientSession):
    """
    Return an entrypoint for an MCP tool.

    Args:
        tool: The MCP tool to create an entrypoint for
        session: The session to use

    Returns:
        Callable: The entrypoint function for the tool
    """
    from agno.agent import Agent

    async def call_tool(agent: Agent, tool_name: str, **kwargs) -> str:
        try:
            log_debug(f"Calling MCP Tool '{tool_name}' with args: {kwargs}")
            result: CallToolResult = await session.call_tool(tool_name, kwargs)  # type: ignore

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


def get_sync_entrypoint_for_tool(tool: MCPTool, session: ClientSession):
    """
    Return a synchronous entrypoint for an MCP tool.

    Args:
        tool: The MCP tool to create an entrypoint for
        session: The session to use

    Returns:
        Callable: The synchronous entrypoint function for the tool
    """
    import asyncio
    from agno.agent import Agent

    def call_tool_sync(agent: Agent, tool_name: str, **kwargs) -> str:
        try:
            log_debug(f"Calling MCP Tool '{tool_name}' with args: {kwargs}")
            
            # Run the async call in the current event loop or create a new one
            try:
                asyncio.get_running_loop()
                # If we're in an event loop, we need to use asyncio.run_coroutine_threadsafe
                # But for simplicity in this synchronous interface, we'll create a new thread
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, _async_call_tool(session, tool_name, kwargs, agent))
                    result = future.result()
            except RuntimeError:
                # No event loop running, we can use asyncio.run directly
                result = asyncio.run(_async_call_tool(session, tool_name, kwargs, agent))
            
            return result
        except Exception as e:
            log_exception(f"Failed to call MCP tool '{tool_name}': {e}")
            return f"Error: {e}"

    return partial(call_tool_sync, tool_name=tool.name)


async def _async_call_tool(session: ClientSession, tool_name: str, kwargs: dict, agent) -> str:
    """Helper function to make the actual async MCP tool call."""
    result: CallToolResult = await session.call_tool(tool_name, kwargs)  # type: ignore

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
