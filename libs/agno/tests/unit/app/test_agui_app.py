from unittest.mock import MagicMock

import pytest
from ag_ui.core import EventType

from agno.app.agui.utils import EventBuffer, async_stream_agno_response_as_agui_events
from agno.run.response import RunResponseContentEvent, ToolCallCompletedEvent, ToolCallStartedEvent


def test_event_buffer_initial_state():
    """Test EventBuffer initial state"""
    buffer = EventBuffer()

    assert not buffer.is_blocked()
    assert buffer.blocking_tool_call_id is None
    assert len(buffer.active_tool_call_ids) == 0
    assert len(buffer.ended_tool_call_ids) == 0
    assert len(buffer.buffer) == 0


def test_event_buffer_tool_call_lifecycle():
    """Test complete tool call lifecycle in EventBuffer"""
    buffer = EventBuffer()

    # Initial state
    assert not buffer.is_blocked()
    assert len(buffer.active_tool_call_ids) == 0

    # Start tool call
    buffer.start_tool_call("tool_1")
    assert buffer.is_blocked()
    assert buffer.blocking_tool_call_id == "tool_1"
    assert "tool_1" in buffer.active_tool_call_ids

    # End tool call
    unblocked = buffer.end_tool_call("tool_1")
    assert unblocked is True
    assert not buffer.is_blocked()
    assert "tool_1" in buffer.ended_tool_call_ids
    assert "tool_1" not in buffer.active_tool_call_ids


def test_event_buffer_multiple_tool_calls():
    """Test multiple concurrent tool calls"""
    buffer = EventBuffer()

    # Start first tool call (becomes blocking)
    buffer.start_tool_call("tool_1")
    assert buffer.blocking_tool_call_id == "tool_1"

    # Start second tool call (doesn't change blocking)
    buffer.start_tool_call("tool_2")
    assert buffer.blocking_tool_call_id == "tool_1"  # Still blocked by first
    assert len(buffer.active_tool_call_ids) == 2

    # End non-blocking tool call
    unblocked = buffer.end_tool_call("tool_2")
    assert unblocked is False
    assert buffer.is_blocked()  # Still blocked by tool_1
    assert buffer.blocking_tool_call_id == "tool_1"

    # End blocking tool call
    unblocked = buffer.end_tool_call("tool_1")
    assert unblocked is True
    assert not buffer.is_blocked()
    assert buffer.blocking_tool_call_id is None


def test_event_buffer_end_nonexistent_tool_call():
    """Test ending a tool call that was never started"""
    buffer = EventBuffer()

    # End tool call that was never started
    unblocked = buffer.end_tool_call("nonexistent_tool")
    assert unblocked is False
    assert not buffer.is_blocked()
    assert "nonexistent_tool" in buffer.ended_tool_call_ids


def test_event_buffer_duplicate_start_tool_call():
    """Test starting the same tool call multiple times"""
    buffer = EventBuffer()

    # Start same tool call twice
    buffer.start_tool_call("tool_1")
    buffer.start_tool_call("tool_1")  # Should not cause issues

    assert buffer.blocking_tool_call_id == "tool_1"
    assert len(buffer.active_tool_call_ids) == 1  # Should still be 1
    assert "tool_1" in buffer.active_tool_call_ids


def test_event_buffer_duplicate_end_tool_call():
    """Test ending the same tool call multiple times"""
    buffer = EventBuffer()

    buffer.start_tool_call("tool_1")

    # End same tool call twice
    unblocked_1 = buffer.end_tool_call("tool_1")
    unblocked_2 = buffer.end_tool_call("tool_1")

    assert unblocked_1 is True
    assert unblocked_2 is False  # Second end should not unblock
    assert not buffer.is_blocked()
    assert "tool_1" in buffer.ended_tool_call_ids


def test_event_buffer_complex_sequence():
    """Test complex sequence of tool call operations"""
    buffer = EventBuffer()

    # Start multiple tool calls
    buffer.start_tool_call("tool_1")  # This becomes blocking
    buffer.start_tool_call("tool_2")
    buffer.start_tool_call("tool_3")

    assert buffer.blocking_tool_call_id == "tool_1"
    assert len(buffer.active_tool_call_ids) == 3

    # End middle tool call (should not unblock)
    unblocked = buffer.end_tool_call("tool_2")
    assert unblocked is False
    assert buffer.is_blocked()
    assert buffer.blocking_tool_call_id == "tool_1"

    # End blocking tool call (should unblock)
    unblocked = buffer.end_tool_call("tool_1")
    assert unblocked is True
    assert not buffer.is_blocked()

    # End remaining tool call
    unblocked = buffer.end_tool_call("tool_3")
    assert unblocked is False  # Already unblocked

    # Check final state
    assert len(buffer.active_tool_call_ids) == 0
    assert len(buffer.ended_tool_call_ids) == 3


def test_event_buffer_blocking_behavior_edge_cases():
    """Test edge cases in blocking behavior"""
    buffer = EventBuffer()

    # Test that empty string tool_call_id is handled gracefully
    buffer.start_tool_call("")  # Empty string
    assert buffer.is_blocked()
    assert buffer.blocking_tool_call_id == ""

    # End with empty string
    unblocked = buffer.end_tool_call("")
    assert unblocked is True
    assert not buffer.is_blocked()


@pytest.mark.asyncio
async def test_stream_basic():
    """Test the async_stream_agno_response_as_agui_events function emits all expected events in a basic case."""
    from agno.run.response import RunEvent

    async def mock_stream():
        text_response = RunResponseContentEvent()
        text_response.event = RunEvent.run_response_content
        text_response.content = "Hello world"
        yield text_response
        completed_response = RunResponseContentEvent()
        completed_response.event = RunEvent.run_completed
        completed_response.content = ""
        yield completed_response

    events = []
    async for event in async_stream_agno_response_as_agui_events(mock_stream(), "thread_1", "run_1"):
        events.append(event)

    assert len(events) == 4
    assert events[0].type == EventType.TEXT_MESSAGE_START
    assert events[1].type == EventType.TEXT_MESSAGE_CONTENT
    assert events[1].delta == "Hello world"
    assert events[2].type == EventType.TEXT_MESSAGE_END
    assert events[3].type == EventType.RUN_FINISHED


@pytest.mark.asyncio
async def test_stream_with_tool_call_blocking():
    """Test that events are properly buffered during tool calls"""
    from agno.run.response import RunEvent

    async def mock_stream_with_tool_calls():
        # Start with a text response
        text_response = RunResponseContentEvent()
        text_response.event = RunEvent.run_response_content
        text_response.content = "I'll help you"
        yield text_response

        # Start a tool call
        tool_start_response = ToolCallStartedEvent()
        tool_start_response.event = RunEvent.tool_call_started
        tool_start_response.content = ""
        tool_call = MagicMock()
        tool_call.tool_call_id = "tool_1"
        tool_call.tool_name = "search"
        tool_call.tool_args = {"query": "test"}
        tool_start_response.tool = tool_call
        yield tool_start_response

        buffered_text_response = RunResponseContentEvent()
        buffered_text_response.event = RunEvent.run_response_content
        buffered_text_response.content = "Searching..."
        yield buffered_text_response
        tool_end_response = ToolCallCompletedEvent()
        tool_end_response.event = RunEvent.tool_call_completed
        tool_end_response.content = ""
        tool_end_response.tool = tool_call
        yield tool_end_response
        completed_response = RunResponseContentEvent()
        completed_response.event = RunEvent.run_completed
        completed_response.content = ""
        yield completed_response

    events = []
    async for event in async_stream_agno_response_as_agui_events(mock_stream_with_tool_calls(), "thread_1", "run_1"):
        events.append(event)

    # Asserting all expected events are present
    event_types = [event.type for event in events]
    assert EventType.TEXT_MESSAGE_START in event_types
    assert EventType.TEXT_MESSAGE_CONTENT in event_types
    assert EventType.TOOL_CALL_START in event_types
    assert EventType.TOOL_CALL_ARGS in event_types
    assert EventType.TOOL_CALL_END in event_types
    assert EventType.TEXT_MESSAGE_END in event_types
    assert EventType.RUN_FINISHED in event_types

    # Verify tool call ordering
    tool_start_idx = event_types.index(EventType.TOOL_CALL_START)
    tool_end_idx = event_types.index(EventType.TOOL_CALL_END)
    assert tool_start_idx < tool_end_idx


@pytest.mark.asyncio
async def test_stream_with_multiple_concurrent_tool_calls():
    """Test that events are properly handled with multiple concurrent tool calls"""
    from agno.run.response import RunEvent

    async def mock_stream_with_multiple_tool_calls():
        # Start with a text response
        text_response = RunResponseContentEvent()
        text_response.event = RunEvent.run_response_content
        text_response.content = "I'll check the stock prices for you"
        yield text_response

        # Start first tool call (TSLA)
        tool_call_1 = MagicMock()
        tool_call_1.tool_call_id = "call_Z2aSMKNTCoFuCIFaP8Eemgza"
        tool_call_1.tool_name = "get_current_stock_price"
        tool_call_1.tool_args = {"symbol": "TSLA"}
        
        tool_start_1 = ToolCallStartedEvent()
        tool_start_1.event = RunEvent.tool_call_started
        tool_start_1.content = ""
        tool_start_1.tool = tool_call_1
        yield tool_start_1

        # Start second tool call (AAPL) - concurrent
        tool_call_2 = MagicMock()
        tool_call_2.tool_call_id = "call_eD7OqR4WEstxXoNhjUeNhzIo"
        tool_call_2.tool_name = "get_current_stock_price"
        tool_call_2.tool_args = {"symbol": "AAPL"}
        
        tool_start_2 = ToolCallStartedEvent()
        tool_start_2.event = RunEvent.tool_call_started
        tool_start_2.content = ""
        tool_start_2.tool = tool_call_2
        yield tool_start_2

        # Start third tool call (MSFT) - concurrent
        tool_call_3 = MagicMock()
        tool_call_3.tool_call_id = "call_ZjpwHTxqOj4pEMZRbr1eu3dJ"
        tool_call_3.tool_name = "get_current_stock_price"
        tool_call_3.tool_args = {"symbol": "MSFT"}
        
        tool_start_3 = ToolCallStartedEvent()
        tool_start_3.event = RunEvent.tool_call_started
        tool_start_3.content = ""
        tool_start_3.tool = tool_call_3
        yield tool_start_3

        # Add some buffered text content during tool calls
        buffered_text_response = RunResponseContentEvent()
        buffered_text_response.event = RunEvent.run_response_content
        buffered_text_response.content = "Fetching stock data..."
        yield buffered_text_response

        # Complete first tool call (TSLA)
        tool_end_1 = ToolCallCompletedEvent()
        tool_end_1.event = RunEvent.tool_call_completed
        tool_end_1.content = ""
        tool_end_1.tool = tool_call_1
        tool_call_1.result = {"price": 250.50, "symbol": "TSLA"}
        yield tool_end_1

        # Complete second tool call (AAPL)
        tool_end_2 = ToolCallCompletedEvent()
        tool_end_2.event = RunEvent.tool_call_completed
        tool_end_2.content = ""
        tool_end_2.tool = tool_call_2
        tool_call_2.result = {"price": 175.25, "symbol": "AAPL"}
        yield tool_end_2

        # Complete third tool call (MSFT)
        tool_end_3 = ToolCallCompletedEvent()
        tool_end_3.event = RunEvent.tool_call_completed
        tool_end_3.content = ""
        tool_end_3.tool = tool_call_3
        tool_call_3.result = {"price": 320.75, "symbol": "MSFT"}
        yield tool_end_3

        # Final text response
        final_text_response = RunResponseContentEvent()
        final_text_response.event = RunEvent.run_response_content
        final_text_response.content = "Here are the current stock prices:"
        yield final_text_response

        # Complete the run
        completed_response = RunResponseContentEvent()
        completed_response.event = RunEvent.run_completed
        completed_response.content = ""
        yield completed_response

    events = []
    async for event in async_stream_agno_response_as_agui_events(mock_stream_with_multiple_tool_calls(), "thread_1", "run_1"):
        events.append(event)

    # Verify all expected event types are present
    event_types = [event.type for event in events]
    
    # Text message events
    assert EventType.TEXT_MESSAGE_START in event_types
    assert EventType.TEXT_MESSAGE_CONTENT in event_types
    assert EventType.TEXT_MESSAGE_END in event_types
    
    # Tool call events for all three tool calls
    assert EventType.TOOL_CALL_START in event_types
    assert EventType.TOOL_CALL_ARGS in event_types
    assert EventType.TOOL_CALL_END in event_types
    assert EventType.TOOL_CALL_RESULT in event_types
    
    # Run completion
    assert EventType.RUN_FINISHED in event_types

    # Verify tool call counts
    tool_start_indices = [i for i, event_type in enumerate(event_types) if event_type == EventType.TOOL_CALL_START]
    tool_end_indices = [i for i, event_type in enumerate(event_types) if event_type == EventType.TOOL_CALL_END]
    
    assert len(tool_start_indices) == 3  # Three tool calls started
    assert len(tool_end_indices) == 3    # Three tool calls ended

    # Verify specific tool call IDs are present
    tool_call_ids = []
    for event in events:
        if hasattr(event, 'tool_call_id'):
            tool_call_ids.append(event.tool_call_id)
    
    expected_tool_call_ids = [
        "call_Z2aSMKNTCoFuCIFaP8Eemgza",  # TSLA
        "call_eD7OqR4WEstxXoNhjUeNhzIo",  # AAPL
        "call_ZjpwHTxqOj4pEMZRbr1eu3dJ"   # MSFT
    ]
    
    for expected_id in expected_tool_call_ids:
        assert expected_id in tool_call_ids

    # Verify tool call arguments
    tool_call_args = []
    for event in events:
        if hasattr(event, 'delta') and event.type == EventType.TOOL_CALL_ARGS:
            tool_call_args.append(event.delta)
    
    expected_args = [
        '{"symbol": "TSLA"}',
        '{"symbol": "AAPL"}',
        '{"symbol": "MSFT"}'
    ]
    
    for expected_arg in expected_args:
        assert expected_arg in tool_call_args

    # Verify tool call results
    tool_call_results = []
    for event in events:
        if hasattr(event, 'content') and event.type == EventType.TOOL_CALL_RESULT:
            tool_call_results.append(event.content)
    
    expected_results = [
        "{'price': 250.5, 'symbol': 'TSLA'}",
        "{'price': 175.25, 'symbol': 'AAPL'}",
        "{'price': 320.75, 'symbol': 'MSFT'}"
    ]
    
    for expected_result in expected_results:
        assert expected_result in tool_call_results
