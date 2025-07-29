from unittest.mock import AsyncMock, patch

import pytest

from agno.tools.mcp import MCPTools, MultiMCPTools, SyncMCPTools


@pytest.mark.asyncio
async def test_sse_transport_without_url_nor_sse_client_params():
    """Test that ValueError is raised when transport is SSE but URL is not provided."""
    with pytest.raises(ValueError, match="One of 'url' or 'server_params' parameters must be provided"):
        async with MCPTools(transport="sse"):
            pass


@pytest.mark.asyncio
async def test_stdio_transport_without_command_nor_server_params():
    """Test that ValueError is raised when transport is stdio but server_params is None."""
    with pytest.raises(ValueError, match="One of 'command' or 'server_params' parameters must be provided"):
        async with MCPTools(transport="stdio"):
            pass


@pytest.mark.asyncio
async def test_streamable_http_transport_without_url_nor_server_params():
    """Test that ValueError is raised when transport is streamable_http but URL is not provided."""
    with pytest.raises(ValueError, match="One of 'url' or 'server_params' parameters must be provided"):
        async with MCPTools(transport="streamable-http"):
            pass


def test_empty_command_string():
    """Test that ValueError is raised when command string is empty."""
    with pytest.raises(ValueError, match="Empty command string"):
        # Mock shlex.split to return an empty list
        with patch("shlex.split", return_value=[]):
            MCPTools(command="")


@pytest.mark.asyncio
async def test_multimcp_without_endpoints():
    """Test that ValueError is raised when no endpoints are provided."""
    with pytest.raises(ValueError, match="Either server_params_list or commands or urls must be provided"):
        async with MultiMCPTools():
            pass


def test_multimcp_empty_command_string():
    """Test that ValueError is raised when a command string is empty."""
    with pytest.raises(ValueError, match="Empty command string"):
        # Mock shlex.split to return an empty list
        with patch("shlex.split", return_value=[]):
            MultiMCPTools(commands=[""])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mcp_tools,kwargs",
    (
        (MCPTools, {"command": "echo foo", "include_tools": ["foo"]}),
        (MCPTools, {"command": "echo foo", "exclude_tools": ["foo"]}),
    ),
)
async def test_mcp_include_exclude_tools_bad_values(mcp_tools, kwargs):
    """Test that _check_tools_filters raises ValueError during initialize"""
    session_mock = AsyncMock()
    tool_mock = AsyncMock()
    tool_mock.__name__ = "baz"
    tools = AsyncMock()
    tools.tools = [tool_mock]
    session_mock.list_tools.return_value = tools

    # _check_tools_filters should be bypassed during __init__
    tools = mcp_tools(**kwargs)
    with pytest.raises(ValueError, match="not present in the toolkit"):
        tools.session = session_mock
        await tools.initialize()


def test_sync_mcp_transport_without_command_nor_server_params():
    """Test that ValueError is raised when transport is stdio but server_params is None."""
    with pytest.raises(ValueError, match="One of 'command' or 'server_params' parameters must be provided"):
        with SyncMCPTools(transport="stdio"):
            pass


def test_sync_mcp_sse_transport_without_url_nor_sse_client_params():
    """Test that ValueError is raised when transport is SSE but URL is not provided."""
    with pytest.raises(ValueError, match="One of 'url' or 'server_params' parameters must be provided"):
        with SyncMCPTools(transport="sse"):
            pass


def test_sync_mcp_streamable_http_transport_without_url_nor_server_params():
    """Test that ValueError is raised when transport is streamable_http but URL is not provided.""" 
    with pytest.raises(ValueError, match="One of 'url' or 'server_params' parameters must be provided"):
        with SyncMCPTools(transport="streamable-http"):
            pass


def test_sync_mcp_empty_command_string():
    """Test that ValueError is raised when command string is empty."""
    with pytest.raises(ValueError, match="Empty command string"):
        # Mock shlex.split to return an empty list
        with patch("shlex.split", return_value=[]):
            SyncMCPTools(command="")


def test_sync_mcp_basic_instantiation():
    """Test that SyncMCPTools can be instantiated with valid parameters."""
    # Test with command
    tools1 = SyncMCPTools(command="echo test")
    assert tools1._command == "echo test"
    assert tools1._transport == "stdio"
    assert not tools1._initialized
    
    # Test with URL for streamable-http
    tools2 = SyncMCPTools(url="http://localhost:8080", transport="streamable-http") 
    assert tools2._url == "http://localhost:8080"
    assert tools2._transport == "streamable-http"
    
    # Test with server_params
    from mcp import StdioServerParameters
    server_params = StdioServerParameters(command="test", args=[])
    tools3 = SyncMCPTools(server_params=server_params)
    assert tools3._server_params == server_params


@patch("agno.tools.mcp.MCPTools")
def test_sync_mcp_initialization_mocking(mock_mcp_tools_class):
    """Test SyncMCPTools initialization with mocked async components."""
    # Mock the async MCPTools instance
    mock_mcp_instance = AsyncMock()
    mock_mcp_instance.functions = {
        "test_tool": AsyncMock(description="Test tool", parameters={"type": "object"})
    }
    mock_mcp_instance.session = AsyncMock()
    
    # Mock list_tools response
    mock_tool = AsyncMock()
    mock_tool.name = "test_tool"
    mock_tool.description = "Test tool"
    mock_tool.inputSchema = {"type": "object"}
    
    mock_tools_response = AsyncMock()
    mock_tools_response.tools = [mock_tool]
    mock_mcp_instance.session.list_tools.return_value = mock_tools_response
    
    # Set up the class mock
    mock_mcp_tools_class.return_value = mock_mcp_instance
    
    # Test the sync tools
    with patch("asyncio.run") as mock_asyncio_run:
        tools = SyncMCPTools(command="echo test")
        tools.initialize()
        
        # Verify asyncio.run was called (since no event loop is running in tests)
        assert mock_asyncio_run.called
        assert tools._initialized
