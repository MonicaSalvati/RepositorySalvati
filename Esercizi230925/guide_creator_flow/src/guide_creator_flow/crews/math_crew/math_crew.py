from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from crewai.tools import tool
import math, json

from pydantic import BaseModel

class SingleParagraph(BaseModel):
    math_result: str

@tool("math_expression_evaluate")
def math_expression_evaluate(code: str):
    """
    This tool is used to evaluate math expression.
    It takes in input a string made of python code to be executed.

    params:
        - code: JSON Object like this {"code": "<code>"}  

    return:
        - A string representing the output produced by executing the provided Python code. 
    """
    if code[0] == "{":
        code = json.loads(code)
        code = code["code"]

    print("codice arrivato al tool, secondo step", code)
    local_vars = {}
    exec(code, {}, local_vars)
    return str(local_vars.get("result", None))

@CrewBase
class MathCrew():
    """MathCrew crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
    
    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools
    @agent
    def python_code_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['python_code_agent'], # type: ignore[index]
            verbose=True,
            tools=[math_expression_evaluate],
        )

    @task
    def python_code_task(self) -> Task:
        return Task(
            config=self.tasks_config['python_code_task'], # type: ignore[index]
        )

    @crew
    def crew(self) -> Crew:
        """Creates the MathCrew crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
