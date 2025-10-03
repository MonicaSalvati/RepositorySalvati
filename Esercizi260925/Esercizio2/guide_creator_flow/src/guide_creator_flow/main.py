#!/usr/bin/env python
from random import randint

from pydantic import BaseModel

from crewai.flow import Flow, listen, start, router, or_

from guide_creator_flow.crews.poem_crew.poem_crew import PoemCrew
from guide_creator_flow.crews.rag_crew.rag_crew import RagCrew
from guide_creator_flow.crews.output_crew.output_crew import OutputCrew

# from utilis.ragas_prova import execute_ragas

import opik
opik.configure(use_local=True)
from opik.integrations.crewai import track_crewai
track_crewai(project_name="crewai-opik-demo")


class CustomState(BaseModel):
    topic: str = "Tourism/Holidays/Trips"
    user_query: str = ""
    relevance: bool = False
    documents: list = []

    opik_payload: dict = {}


class PoemFlow(Flow[CustomState]):

    @start()
    def simple_start(self):
        print("Start")

    @listen(or_(simple_start, "Not Relevant"))
    def get_user_query(self):
        print("Start")
        self.state.user_query = input(f"Inserisci la tua domanda rilevante al contesto {self.state.topic}: ")

    @listen(get_user_query)
    def relevance_evaluation(self):
        print("relevance_evaluation")
        crew = PoemCrew()
        result = (
            crew
            .crew()
            .kickoff(inputs={"user_query": self.state.user_query,
                             "topic":self.state.topic})
        )

        print("Risultato del controllo sulla rilevanza", result)

        if isinstance(result["relevance"], str):
            self.state.relevance = bool(result["relevance"])
        else:
            self.state.relevance = result["relevance"]

        self.state.opik_payload = {
            "user_query": self.state.user_query,
            "topic":self.state.topic,
            "relevance": self.state.relevance,
            "PoemCrew": crew
        }


    @router(relevance_evaluation)
    def save_poem(self):
        if self.state.relevance:
            return "Relevant"
        print("Domanda non rilevante")
        return "Not Relevant"
    
    @listen("Relevant")
    def vector_search(self):
        print("vector_search")
        crew = RagCrew()
        result = (
            crew
            .crew()
            .kickoff(inputs={"user_query": self.state.user_query})
        )
        self.state.documents = result["docs"]

        print("Documenti trovati:", self.state.documents)

        self.state.opik_payload["RagCrew"] = crew
        self.state.opik_payload["documents"] = "\n\n".join(self.state.documents) if self.state.documents else ""
        
    @listen(vector_search)
    def generate_output(self):
        print("generate_output")
        crew = OutputCrew()
        result = (
            crew
            .crew()
            .kickoff(inputs={"user_query": self.state.user_query,
                             "documents": self.state.documents,
                             "topic":self.state.topic})
        )
        print("Output generato con successo")

        with open(r"C:\Users\BK476KA\OneDrive - EY\Desktop\RepositorySalvati\Esercizi260925\Esercizio2\guide_creator_flow\report.md", "r", encoding="utf-8") as f:
            read_md = f.read()

        self.state.opik_payload["OutputCrew"] = crew
        self.state.opik_payload["output"] = read_md
        return self.state.opik_payload

    # @listen(generate_output)
    # def ragas(self):
    #     with open(r"C:\Users\JC235KM\OneDrive - EY\Desktop\Esercizi Accademy\CrewAIRAG\guide_creator_flow\report.md", "r", encoding="utf-8") as f:
    #         read_md = f.read()
    #         print("Contenuto del file report.md:", read_md[:5])  # Stampa i primi 500 caratteri per verifica
    #     execute_ragas(self.state.user_query, self.state.documents, read_md)


def kickoff():
    poem_flow = PoemFlow()
    poem_flow.kickoff()

if __name__ == "__main__":    
    kickoff()

 