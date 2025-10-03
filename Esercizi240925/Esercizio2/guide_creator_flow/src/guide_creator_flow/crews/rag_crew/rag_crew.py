"""RAG crew: retrieval using a custom RAGSearch tool.

This module contains a crew definition that wires a researcher agent equipped
with a custom ``RAGSearch`` tool. The crew is intended to be used by the
PoemFlow to perform vector-based retrieval over an internal vector store.
"""

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
    """Schema for documents retrieved by the RAG tool."""
    docs: List[Any] = []

        

@CrewBase
class RagCrew():
    """Crew that issues a search task using the ``RAGSearch`` tool."""

    agents: List[BaseAgent]
    tasks: List[Task]

    @agent
    def researcher(self) -> Agent:
        """Agent equipped with the ``RAGSearch`` tool to retrieve documents."""
        return Agent(
            config=self.agents_config['researcher'], # type: ignore[index]
            tools=[RAGSearch],
            verbose=True
        )

    @task
    def research_task(self) -> Task:
        """Task that produces ``RagToolOutput`` with retrieved docs."""
        return Task(
            config=self.tasks_config['research_task'],
            output_json=RagToolOutput,            
        )

    @crew
    def crew(self) -> Crew:

        """Create the crew with a sequential process for retrieval."""
        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
