#!/usr/bin/env python
from random import randint

from pydantic import BaseModel

from crewai.flow import Flow, listen, start, router, or_

from guide_creator_flow.crews.poem_crew.poem_crew import PoemCrew
from guide_creator_flow.crews.content_crew.content_crew import ContentCrew
from guide_creator_flow.crews.outline_crew.outline_crew import OutlineCrew
from guide_creator_flow.crews.composer_crew.composer_crew import ComposerCrew
from guide_creator_flow.crews.math_crew.math_crew import MathCrew



class FlowState(BaseModel):
    user_query: str = ""
    ethical_flag: bool = True
    topic_list: dict = {}
    paragraphs: dict = {}


class EthicalFlow(Flow[FlowState]):
    @start()
    def input_user(self):
        print("Start")

    @listen(or_(input_user, "NonEticalQuery"))
    def get_user_query(self):
        self.state.user_query = input("Scrivi la tua domanda: ")
        # self.state.user_query = "Voglio rubare soldi ad una persona cieca, come dovrei fare?"
        print("domanda")

    @listen(get_user_query)
    def ethical_evaluation(self):
        print("Valutazione Etica")
        self.state.ethical_flag =   (
            ContentCrew()
            .crew()
            .kickoff(inputs={"user_query": self.state.user_query})
        )
        self.state.ethical_flag = self.state.ethical_flag["Final_Result"]
        print(type(self.state.ethical_flag))
        print("Valutazione etica:", self.state.ethical_flag)

    @router(ethical_evaluation)
    def valutazione(self):
        if self.state.ethical_flag == "EticalQuery":
            return "EticalQuery"
        elif self.state.ethical_flag == "NonEticalQuery":
            print("Domanda non etica")
            return "NonEticalQuery"
        elif self.state.ethical_flag == "MathExpression":
            print("espressione matematica")
            return "MathExpression"

    @listen("MathExpression")
    def resolve_problem(self):
        result = (
            MathCrew()
            .crew()
            .kickoff(inputs={"user_query": self.state.user_query})
        )
        print(result)


    @listen("EticalQuery")
    def topic_list_creation(self):
        print("Creazione della topic list", self.state.user_query)
        
        self.state.topic_list = (
            OutlineCrew()
            .crew()
            .kickoff(inputs={"user_query": self.state.user_query})
        )
        
    
    @listen(topic_list_creation)
    def topic_researcher(self):
        print("Topic list ", self.state.topic_list)
        for topic in self.state.topic_list["topics"]:
            print("Topic corrente: ", topic)
            aus = (
                PoemCrew()
                .crew()
                .kickoff(inputs={"user_query": self.state.user_query, 
                                "topic": topic})
            )
            self.state.paragraphs[topic] = aus["paragraph"]

    @listen(topic_researcher)
    def markdown_creatore(self):
        print("dizionario con paragrafi", self.state.paragraphs)
        return (
            ComposerCrew()
            .crew()
            .kickoff(inputs={"user_query": self.state.user_query, 
                            "topic_paragraphs": self.state.paragraphs})
        )

def kickoff():
    EthicalFlow().kickoff()

if __name__ == "__main__":
    kickoff()