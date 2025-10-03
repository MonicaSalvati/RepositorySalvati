from typing import Type

from crewai.tools import BaseTool
from crewai.flow import Flow, listen, start
from crewai.tools import tool
from pydantic import BaseModel, Field

@tool("adder_tool")
def adder_tool(a: int, b: int) -> str:
    """
    Adds two numbers together.

    Parameters:
    - a (int): The first number.
    - b (int): The second number.
    Returns:
    - str: The sum of the two numbers as a string.

    """
    return str(a + b)
