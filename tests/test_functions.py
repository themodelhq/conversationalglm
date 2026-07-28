import pytest
from inference.functions import FunctionRegistry, parse_tool_call
def test_calculator_and_parser():
 registry=FunctionRegistry();assert registry.call("calculator",{"expression":"2 * (3 + 4)"})["result"]==14
 assert parse_tool_call('<tool>{"name":"current_time","arguments":{}}</tool>')==("current_time",{})
 with pytest.raises(ValueError):registry.call("calculator",{"expression":"__import__('os').system('id')"})
