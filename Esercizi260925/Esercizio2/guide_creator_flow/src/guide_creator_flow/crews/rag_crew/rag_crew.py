from dataclasses import Field
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import Any, List
from pydantic import BaseModel
from pathlib import Path
from typing import Type    
from guide_creator_flow.tools.custom_tool import RAGSearch


class RagToolOutput(BaseModel):
    """Schema for outputs produced by the retrieval tool.

    Attributes
    ----------
    docs : List[Any]
        Retrieved documents or chunks, as returned by the RAG search tool.
    """
    docs: List[Any] = []

        

@CrewBase
class RagCrew():
    """Crew that performs retrieval using a single researcher agent.

    The crew exposes a researcher agent equipped with the ``RAGSearch`` tool
    and a single retrieval task that returns documents in the
    ``RagToolOutput`` format.
    """

    agents: List[BaseAgent]
    tasks: List[Task]

    @agent
    def researcher(self) -> Agent:
        """Create the retrieval agent configured with the RAGSearch tool."""
        return Agent(
            config=self.agents_config['researcher'], # type: ignore[index]
            tools=[RAGSearch],
            verbose=True
        )

    @task
    def research_task(self) -> Task:
        """Define the retrieval task that emits ``RagToolOutput`` JSON."""
        return Task(
            config=self.tasks_config['research_task'],
            output_json=RagToolOutput,            
        )

    @crew
    def crew(self) -> Crew:
        """Assemble the crew with the configured agents and tasks."""
        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
