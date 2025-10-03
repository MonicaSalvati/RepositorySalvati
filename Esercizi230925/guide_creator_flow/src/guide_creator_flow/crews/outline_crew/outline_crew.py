from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List

from pydantic import BaseModel

class TopicList(BaseModel):
    topics: list

@CrewBase
class OutlineCrew():
    """OutlineCrew crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    @agent
    def topic_extractor(self) -> Agent:
        return Agent(
            config=self.agents_config['topic_extractor'], # type: ignore[index]
            verbose=True
        )

    @agent
    def bias_reviewer(self) -> Agent:
        return Agent(
            config=self.agents_config['bias_reviewer'], # type: ignore[index]
            verbose=True
        )

    @task
    def topic_task(self) -> Task:
        return Task(
            config=self.tasks_config['topic_task'],
            output_json=TopicList
        )

    @task
    def bias_review_task(self) -> Task:
        return Task(
            config=self.tasks_config['bias_review_task'], # type: ignore[index]
            output_json=TopicList
        )

    @crew
    def crew(self) -> Crew:
        """Creates the OutlineCrew crew"""

        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
