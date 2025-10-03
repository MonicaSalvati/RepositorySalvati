from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from guide_creator_flow.tools.custom_tool import adder_tool
from typing import List

@CrewBase
class SumCrew:
    """Poem Crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def sum_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["sum_agent"],  # type: ignore[index]
            tools=[adder_tool],
            verbose=True
        )

    @task
    def sum_task(self) -> Task:
        return Task(
            config=self.tasks_config["sum_task"],  # type: ignore[index]
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Research Crew"""
        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
        )
