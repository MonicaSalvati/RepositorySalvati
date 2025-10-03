"""Poem crew: evaluate the relevance of a user query.

This module provides a lightweight Crew definition that is used by the
``PoemFlow`` to determine whether a user-provided query is relevant to the
selected topic. The crew exposes a single public crew factory method
``crew()`` that returns a configured :class:`crewai.Crew` instance.

Notes
-----
The implementation relies on CrewAI decorators and expects to find agent
and task configuration YAML files under the ``config/`` directory. The
module only contains declarative wiring; the heavy-lifting is performed by
the configured agents at runtime.

Classes
-------
RelevanceOutput
    Pydantic model describing the expected task output schema with a single
    ``relevance`` boolean field.

PoemCrew
    Crew factory exposing the ``crew()`` method used by the flow.
"""

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from pydantic import BaseModel

class RelevanceOutput(BaseModel):
    """Schema for relevance evaluation results.

    Attributes
    ----------
    relevance : bool
        True when the user query is judged relevant to the topic, otherwise
        False.
    """
    relevance: bool = False

@CrewBase
class PoemCrew:
    """Crew that determines query relevance and returns a boolean flag.

    The class is a declarative CrewAI component: instantiate it and call
    ``PoemCrew().crew()`` to obtain an executable Crew object.
    """

    agents: List[BaseAgent]
    tasks: List[Task]

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def relevance_evaluator(self) -> Agent:
        """Create the relevance evaluator agent.

        Returns
        -------
        Agent
            Configured CrewAI Agent that performs the relevance judgment.
        """
        return Agent(
            config=self.agents_config["relevance_evaluator"],  # type: ignore[index]
        )

    @task
    def evaluate_relevance(self) -> Task:
        """Create the evaluation task that returns :class:`RelevanceOutput`.

        Returns
        -------
        Task
            CrewAI Task configured to produce a JSON matching ``RelevanceOutput``.
        """
        return Task(
            config=self.tasks_config["evaluate_relevance"],  # type: ignore[index]
            output_json=RelevanceOutput
        )

    @crew
    def crew(self) -> Crew:
        """Assemble and return the Crew instance.

        Returns
        -------
        Crew
            Executable CrewAI Crew object wired with agents and tasks.
        """
        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
        )
