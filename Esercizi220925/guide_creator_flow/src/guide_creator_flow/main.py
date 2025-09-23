#!/usr/bin/env python
from random import randint

from pydantic import BaseModel

from crewai.flow import Flow, listen, start, router, or_

from guide_creator_flow.crews.poem_crew.poem_crew import PoemCrew
from guide_creator_flow.crews.content_crew.content_crew import ContentCrew



class FlowState(BaseModel):
    user_query: str = ""
    ethical_flag: bool = True


class EthicalFlow(Flow[FlowState]):
    @start()
    def input_user(self):
        print("Start")

    # @listen("NonEticalQuery")
    # def ethical_evaluation_2(self):
    #     print("Domanda non etica")
    #     # self.state.user_query = None
    #     # self.state.ethical_flag = None

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
        if self.state.ethical_flag:
            return "EticalQuery"
        else:
            print("Domanda non etica")
            return "NonEticalQuery"

    @listen("EticalQuery")
    def ethical_evaluation_1(self):
        print("Creazione risposta alla domanda etica", self.state.user_query)
        return (
            PoemCrew()
            .crew()
            .kickoff(inputs={"user_query": self.state.user_query})
        )





def kickoff():
    EthicalFlow().kickoff()


if __name__ == "__main__":
    kickoff()
