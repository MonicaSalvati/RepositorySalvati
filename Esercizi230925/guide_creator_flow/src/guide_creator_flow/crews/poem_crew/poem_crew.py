from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from crewai_tools import SerperDevTool
# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

from pydantic import BaseModel

class SingleParagraph(BaseModel):
    paragraph: str

@CrewBase
class PoemCrew:
    """Poem Crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def topic_paragraph_writer(self) -> Agent:
        return Agent(
            config=self.agents_config["topic_paragraph_writer"],  # type: ignore[index]
            tools=[SerperDevTool()],
            verbose=True
        )
    
    
    @task
    def write_topic_paragraph(self) -> Task:
        return Task(
            config=self.tasks_config["write_topic_paragraph"],  # type: ignore[index]
            output_json=SingleParagraph
        )
    
    @crew
    def crew(self) -> Crew:
        """Creates the Research Crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
        )
