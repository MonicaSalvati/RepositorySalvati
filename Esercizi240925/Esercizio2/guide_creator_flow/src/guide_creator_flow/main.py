#!/usr/bin/env python
from pydantic import BaseModel

from crewai.flow import Flow, listen, start, router, or_, and_
import json

from guide_creator_flow.crews.poem_crew.poem_crew import PoemCrew
from guide_creator_flow.crews.rag_crew.rag_crew import RagCrew
from guide_creator_flow.crews.output_crew.output_crew import OutputCrew
from guide_creator_flow.crews.research_crew.research_crew import ResearchCrew
from utilis.checker import Checker
from utilis.ragas_prova import execute_ragas

import opik
opik.configure(use_local=True)
from opik.integrations.crewai import track_crewai
track_crewai(project_name="crewai-opik-demo")

WHITE_LIST = [
    "lonelyplanet.com",       
    "tripadvisor.com",        
    "booking.com",            
    "expedia.com",            
    "airbnb.com",            
    "skyscanner.com",         
    "kayak.com",              
    "trivago.com",            
    "nationalgeographic.com/travel",  
    "unwto.org"               
]


class CustomState(BaseModel):
    topic: str = "Tourism/Holidays/Trips"
    user_query: str = ""
    relevance: bool = False
    documents: list = []
    web_documents: list = []

    opik_payload: dict = {}


class PoemFlow(Flow[CustomState]):

    @start()
    def start(self):
        print("Start")

    @router(start)
    def get_user_query(self):
        # Se la user_query è già stata impostata (es. da interfaccia Streamlit) non chiedere input da console
        checker = Checker("")
        # print(checker.prompt_shield(self.state.user_query))
        if checker.prompt_shield(self.state.user_query):
           return "Not Relevant" 
        return "OK"

    @listen(and_(get_user_query, "OK"))
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
    def search_router(self):
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
        self.state.documents = [json.dumps(d) for d in result["docs"]]
        self.state.opik_payload["RagCrew"] = crew
        self.state.opik_payload["documents"] = "\n\n".join(self.state.documents) if self.state.documents else ""

    @listen("Relevant")
    def web_search(self):
        print("web_search")
        crew = ResearchCrew()
        site_operators = " OR ".join([f"site:{domain}" for domain in WHITE_LIST])
        full_query = f"{self.state.user_query} ({site_operators})"
        result = (
            crew
            .crew()
            .kickoff(inputs={"user_query": full_query},)
        )
        self.state.web_documents = [json.dumps({**d, "trusted": True}) for d in result["results"]]
        self.state.opik_payload["WebCrew"] = crew
        self.state.opik_payload["web_documents"] = "\n\n".join(self.state.web_documents) if self.state.web_documents else ""

    @listen(and_(vector_search, web_search))
    def generate_output(self):
        print("generate_output")
        crew = OutputCrew()
        result = (
            crew
            .crew()
            .kickoff(inputs={"user_query": self.state.user_query,
                             "documents": self.state.documents,
                             "web_documents": self.state.web_documents,
                             "topic":self.state.topic})
        )
        print("Output generato con successo")

        with open(r"C:\Users\BK476KA\OneDrive - EY\Desktop\RepositorySalvati\Esercizi240925\Esercizio2\guide_creator_flow\report.md", "r", encoding="utf-8") as f:
            read_md = f.read()

        self.state.opik_payload["OutputCrew"] = crew
        self.state.opik_payload["output"] = read_md
        self.state.opik_payload["context"] = self.state.opik_payload["documents"]+ "\n\n" +self.state.opik_payload["web_documents"]
        return self.state.opik_payload
    
    @listen("Not Relevant")
    def generate_not_relevant_output(self):
        return "Individuato problema con l'input"

    @listen(generate_output)
    def ragas(self):
        with open(r"C:\Users\BK476KA\OneDrive - EY\Desktop\RepositorySalvati\Esercizi240925\Esercizio2\guide_creator_flow\report.md", "r", encoding="utf-8") as f:
            read_md = f.read()
            print("Contenuto del file report.md:", read_md[:5])  # Stampa i primi 500 caratteri per verifica
        execute_ragas(self.state.user_query, self.state.opik_payload["context"], read_md, self.state.topic)
        return self.state.opik_payload


# def kickoff():
#     poem_flow = PoemFlow()
#     poem_flow.plot()
#     poem_flow.kickoff()


# if __name__ == "__main__":   
#     kickoff()

 