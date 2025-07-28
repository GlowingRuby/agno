"""Tests for parallel tool call handling in AGUI app to prevent deadlocks."""

import pytest
from unittest.mock import MagicMock
from ag_ui.core import EventType, ToolCallStartEvent, ToolCallEndEvent, ToolCallArgsEvent
from agno.app.agui.utils import EventBuffer, _emit_event_logic


def test_parallel_tool_calls_prevent_deadlock():
    """Test that parallel tool calls don't cause deadlocks when one hangs."""
    buffer = EventBuffer()
    
    # Simulate tool_1 starting and becoming blocking
    tool_1_start = ToolCallStartEvent(
        type=EventType.TOOL_CALL_START,
        tool_call_id="tool_1",
        tool_call_name="hanging_tool",
        parent_message_id="msg_1"
    )
    events_1_start = _emit_event_logic(tool_1_start, buffer)
    assert len(events_1_start) == 1
    assert buffer.blocking_tool_call_id == "tool_1"
    
    # Simulate tool_2 starting but getting buffered
    tool_2_start = ToolCallStartEvent(
        type=EventType.TOOL_CALL_START,
        tool_call_id="tool_2",
        tool_call_name="working_tool", 
        parent_message_id="msg_1"
    )
    events_2_start = _emit_event_logic(tool_2_start, buffer)
    assert len(events_2_start) == 0  # Gets buffered
    assert len(buffer.buffer) == 1
    
    # Simulate tool_2 being processed and becoming active
    # (this would happen when the buffer is flushed)
    buffer.active_tool_call_ids.add("tool_2")
    
    # Now tool_2 completes its work 
    tool_2_end = ToolCallEndEvent(
        type=EventType.TOOL_CALL_END,
        tool_call_id="tool_2"
    )
    events_2_end = _emit_event_logic(tool_2_end, buffer)
    
    # With the fix, tool_2 end should be emitted even though tool_1 is still blocking
    assert len(events_2_end) == 1
    assert "tool_2" in buffer.ended_tool_call_ids
    assert "tool_2" not in buffer.active_tool_call_ids
    
    # tool_1 is still the blocking tool call and hasn't completed
    assert buffer.blocking_tool_call_id == "tool_1"
    assert "tool_1" in buffer.active_tool_call_ids


def test_non_blocking_tool_call_args_still_buffered():
    """Test that args for non-active tool calls are still buffered properly.""" 
    buffer = EventBuffer()
    
    # Start blocking tool call
    buffer.start_tool_call("tool_1")
    
    # Try to send args for a tool call that hasn't started yet
    tool_2_args = ToolCallArgsEvent(
        type=EventType.TOOL_CALL_ARGS,
        tool_call_id="tool_2", 
        delta='{"query": "test"}'
    )
    events = _emit_event_logic(tool_2_args, buffer)
    
    # Args should be buffered since tool_2 is not active
    assert len(events) == 0
    assert len(buffer.buffer) == 1
    
    # Now make tool_2 active and try again
    buffer.active_tool_call_ids.add("tool_2")
    tool_2_args_2 = ToolCallArgsEvent(
        type=EventType.TOOL_CALL_ARGS,
        tool_call_id="tool_2",
        delta='{"param": "value"}'
    )
    events = _emit_event_logic(tool_2_args_2, buffer)
    
    # Args for active tool calls should be emitted even when blocked
    assert len(events) == 1


def test_multiple_parallel_tool_completions():
    """Test that multiple parallel tool calls can complete independently."""
    buffer = EventBuffer()
    
    # Start multiple tool calls
    buffer.start_tool_call("tool_1")  # Becomes blocking
    buffer.start_tool_call("tool_2")  # Active
    buffer.start_tool_call("tool_3")  # Active
    
    assert buffer.blocking_tool_call_id == "tool_1"
    assert len(buffer.active_tool_call_ids) == 3
    
    # End tool_2 (non-blocking)
    tool_2_end = ToolCallEndEvent(
        type=EventType.TOOL_CALL_END,
        tool_call_id="tool_2"
    )
    events_2 = _emit_event_logic(tool_2_end, buffer)
    assert len(events_2) == 1  # Should be emitted immediately
    
    # End tool_3 (non-blocking)  
    tool_3_end = ToolCallEndEvent(
        type=EventType.TOOL_CALL_END,
        tool_call_id="tool_3"
    )
    events_3 = _emit_event_logic(tool_3_end, buffer)
    assert len(events_3) == 1  # Should be emitted immediately
    
    # Verify both completed
    assert "tool_2" in buffer.ended_tool_call_ids
    assert "tool_3" in buffer.ended_tool_call_ids
    assert "tool_2" not in buffer.active_tool_call_ids
    assert "tool_3" not in buffer.active_tool_call_ids
    
    # tool_1 is still blocking
    assert buffer.blocking_tool_call_id == "tool_1"
    assert "tool_1" in buffer.active_tool_call_ids


def test_buffer_flush_still_works():
    """Test that buffer flushing still works when blocking tool call completes."""
    buffer = EventBuffer()
    
    # Start tool_1 (blocking)
    tool_1_start = ToolCallStartEvent(
        type=EventType.TOOL_CALL_START,
        tool_call_id="tool_1",
        tool_call_name="first_tool",
        parent_message_id="msg_1"
    )
    events_1 = _emit_event_logic(tool_1_start, buffer)
    assert len(events_1) == 1
    
    # Add some events to buffer
    tool_2_start = ToolCallStartEvent(
        type=EventType.TOOL_CALL_START,
        tool_call_id="tool_2",
        tool_call_name="second_tool",
        parent_message_id="msg_1"
    )
    events_2 = _emit_event_logic(tool_2_start, buffer)
    assert len(events_2) == 0  # Buffered
    assert len(buffer.buffer) == 1
    
    # End blocking tool call
    tool_1_end = ToolCallEndEvent(
        type=EventType.TOOL_CALL_END,
        tool_call_id="tool_1"
    )
    events_1_end = _emit_event_logic(tool_1_end, buffer)
    
    # Should emit tool_1_end and flush buffered tool_2_start
    assert len(events_1_end) == 2  # tool_1_end + tool_2_start from buffer
    # Note: tool_2 becomes the new blocking tool call when flushed from buffer
    assert buffer.blocking_tool_call_id == "tool_2"  # tool_2 now blocking
    assert len(buffer.buffer) == 0  # Buffer flushed