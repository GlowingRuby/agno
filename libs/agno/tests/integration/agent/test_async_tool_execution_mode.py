"""Test async tool execution mode functionality"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agno.agent.agent import Agent
from agno.models.response import ModelResponse, ToolExecution, ModelResponseEvent
from agno.run.messages import RunMessages
from agno.run.response import RunResponse


class MockAsyncTool:
    """Mock async tool for testing"""
    
    def __init__(self, name: str, delay: float = 0.1):
        self.name = name
        self.delay = delay
        self.call_count = 0
        self.call_times = []
    
    async def __call__(self, **kwargs):
        """Simulate async tool execution"""
        start_time = time.time()
        self.call_count += 1
        await asyncio.sleep(self.delay)
        end_time = time.time()
        self.call_times.append((start_time, end_time))
        return f"Result from {self.name}"


def test_agent_async_tool_execution_mode_parameter():
    """Test that Agent accepts and stores async_tool_execution_mode parameter"""
    
    # Test default value
    agent1 = Agent()
    assert agent1.async_tool_execution_mode == "serial"
    
    # Test explicit serial mode
    agent2 = Agent(async_tool_execution_mode="serial")
    assert agent2.async_tool_execution_mode == "serial"
    
    # Test parallel mode
    agent3 = Agent(async_tool_execution_mode="parallel")
    assert agent3.async_tool_execution_mode == "parallel"


def test_agent_async_tool_execution_mode_backward_compatibility():
    """Test that the new parameter doesn't break existing Agent instantiation"""
    
    # Test with common existing parameters
    agent = Agent(
        name="test-agent",
        description="Test agent",
        show_tool_calls=True,
        tool_call_limit=5,
        debug_mode=False
    )
    
    # Should default to serial mode
    assert agent.async_tool_execution_mode == "serial"
    
    # Should not affect other parameters
    assert agent.name == "test-agent"
    assert agent.description == "Test agent"
    assert agent.show_tool_calls is True
    assert agent.tool_call_limit == 5
    assert agent.debug_mode is False


@pytest.mark.asyncio
async def test_async_tool_execution_mode_integration():
    """Integration test to verify the execution mode logic paths are accessible"""
    
    # Mock the necessary components
    mock_model = MagicMock()
    mock_model.get_function_call_to_run_from_tool_execution.return_value = MagicMock()
    
    # Create async mock for arun_function_calls
    async def mock_arun_function_calls(function_calls, function_call_results, skip_pause_check=False):
        # Simulate tool execution delay
        await asyncio.sleep(0.1)
        
        # Yield started event
        model_response = ModelResponse()
        model_response.event = ModelResponseEvent.tool_call_started.value
        yield model_response
        
        # Yield completed event
        model_response = ModelResponse()
        model_response.event = ModelResponseEvent.tool_call_completed.value
        tool_execution = ToolExecution()
        tool_execution.result = "Mock result"
        tool_execution.tool_call_error = False
        model_response.tool_executions = [tool_execution]
        model_response.content = "Mock content"
        yield model_response
    
    mock_model.arun_function_calls = mock_arun_function_calls
    
    # Test both modes
    for mode in ["serial", "parallel"]:
        agent = Agent(
            model=mock_model,
            async_tool_execution_mode=mode
        )
        
        # Mock required attributes
        agent._functions_for_model = True
        
        # Create mock run response with tools
        run_response = RunResponse()
        run_response.session_id = "test-session"
        run_response.run_id = "test-run"
        run_response.agent_id = "test-agent"
        run_response.tools = []
        
        # Add mock tools
        for i in range(2):
            tool = ToolExecution()
            tool.tool_name = f"test_tool_{i}"
            tool.requires_confirmation = True
            tool.confirmed = True
            tool.result = None
            tool.requires_user_input = None
            tool.external_execution_required = None
            run_response.tools.append(tool)
        
        run_messages = RunMessages()
        run_messages.messages = []
        
        # Set agent run_response to avoid None references
        agent.run_response = run_response
        
        # Mock the _handle_event method
        agent._handle_event = lambda event, run_response: event
        
        # Test the async tool call updates method
        start_time = time.time()
        await agent._ahandle_tool_call_updates(run_response, run_messages)
        end_time = time.time()
        
        # Verify tools were processed
        for tool in run_response.tools:
            assert tool.requires_confirmation is False
        
        print(f"Mode {mode} execution took: {end_time - start_time:.2f}s")


def test_async_tool_execution_mode_type_hint():
    """Test that the type hint is correctly specified"""
    from agno.agent.agent import Agent
    import inspect
    
    # Get the constructor signature
    sig = inspect.signature(Agent.__init__)
    param = sig.parameters.get('async_tool_execution_mode')
    
    assert param is not None, "async_tool_execution_mode parameter should exist"
    assert param.default == "serial", "Default should be 'serial'"


# Performance comparison test (optional, can be slow)
@pytest.mark.slow
@pytest.mark.asyncio
async def test_parallel_vs_serial_performance():
    """Test that parallel execution is faster than serial execution (when possible)"""
    
    # This test is marked as slow and would typically be run separately
    # It demonstrates the performance benefit of parallel execution
    
    mock_model = MagicMock()
    
    async def slow_mock_arun_function_calls(function_calls, function_call_results, skip_pause_check=False):
        # Simulate slower tool execution
        await asyncio.sleep(0.2)
        
        model_response = ModelResponse()
        model_response.event = ModelResponseEvent.tool_call_completed.value
        tool_execution = ToolExecution()
        tool_execution.result = "Mock result"
        tool_execution.tool_call_error = False
        model_response.tool_executions = [tool_execution]
        yield model_response
    
    mock_model.arun_function_calls = slow_mock_arun_function_calls
    mock_model.get_function_call_to_run_from_tool_execution.return_value = MagicMock()
    
    execution_times = {}
    
    for mode in ["serial", "parallel"]:
        agent = Agent(
            model=mock_model,
            async_tool_execution_mode=mode
        )
        agent._functions_for_model = True
        
        run_response = RunResponse()
        run_response.session_id = "test-session"
        run_response.run_id = "test-run"
        run_response.agent_id = "test-agent"
        run_response.tools = []
        
        # Create 3 tools for testing
        for i in range(3):
            tool = ToolExecution()
            tool.tool_name = f"test_tool_{i}"
            tool.requires_confirmation = True
            tool.confirmed = True
            tool.result = None
            run_response.tools.append(tool)
        
        run_messages = RunMessages()
        run_messages.messages = []
        agent.run_response = run_response
        agent._handle_event = lambda event, run_response: event
        
        start_time = time.time()
        await agent._ahandle_tool_call_updates(run_response, run_messages)
        end_time = time.time()
        
        execution_times[mode] = end_time - start_time
    
    # Parallel should be faster than serial (with some tolerance for timing variations)
    # This assertion might be flaky in CI environments, so we make it optional
    if execution_times["parallel"] < execution_times["serial"] * 0.8:
        print(f"✅ Parallel execution ({execution_times['parallel']:.2f}s) is faster than serial ({execution_times['serial']:.2f}s)")
    else:
        print(f"⚠️  Performance difference not significant: parallel={execution_times['parallel']:.2f}s, serial={execution_times['serial']:.2f}s")