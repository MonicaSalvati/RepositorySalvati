"""Crew that generates the RAG answer and composes a Markdown report."""

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List

@CrewBase
class OutputCrew():
    """Crew responsible for answering and authoring the final report."""

    agents: List[BaseAgent]
    tasks: List[Task]

    @agent
    def rag_responder(self) -> Agent:
        """Agent that formulates the RAG-based answer."""
        return Agent(
            config=self.agents_config['rag_responder'], # type: ignore[index]
            verbose=True
        )

    @agent
    def markdown_author(self) -> Agent:
        """Agent that structures the answer into a Markdown article."""
        return Agent(
            config=self.agents_config['markdown_author'], # type: ignore[index]
            verbose=True
        )

    @task
    def rag_answer_task(self) -> Task:
        """Task to synthesize an answer based on retrieved documents."""
        return Task(
            config=self.tasks_config['rag_answer_task'], # type: ignore[index]
        )

    @task
    def markdown_article_task(self) -> Task:
        """Task to write the final Markdown article to ``report.md``."""
        return Task(
            config=self.tasks_config['markdown_article_task'], # type: ignore[index]
            output_file='report.md'
        )

    @crew
    def crew(self) -> Crew:
        """Compose the crew with agents and tasks in a sequential process."""
        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
