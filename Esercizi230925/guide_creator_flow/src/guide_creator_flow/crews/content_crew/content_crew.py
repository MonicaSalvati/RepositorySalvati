from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from pydantic import BaseModel

class ResultModel(BaseModel):
    Final_Result: str

@CrewBase
class ContentCrew():
    """ContentCrew crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    @agent
    def evaluator(self) -> Agent:
        return Agent(
            config=self.agents_config['evaluator'], # type: ignore[index]
            verbose=True
        )
    
    @task
    def evaluation_task(self) -> Task:
        return Task(
            config=self.tasks_config['evaluation_task'], # type: ignore[index]
            output_json = ResultModel
        )

    @crew
    def crew(self) -> Crew:
        """Creates the ContentCrew crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
