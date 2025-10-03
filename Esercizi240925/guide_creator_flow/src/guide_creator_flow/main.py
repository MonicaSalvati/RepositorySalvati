#!/usr/bin/env python
from random import randint

from pydantic import BaseModel

from crewai.flow import Flow, listen, start

from guide_creator_flow.crews.poem_crew.poem_crew import SumCrew



class FlowState(BaseModel):
    n1: int | None = None
    n2: int | None = None

class SumFlow(Flow[FlowState]):

    @start()
    def start(self):
        print("Start")

    @listen(start)
    def get_user_input(self):
        print("Getting user input")
        while True:
            user_input = input("Enter two numbers separated by space or comma: ").strip()
            
            parts = [p for p in user_input.replace(",", " ").split() if p != ""]
            if len(parts) != 2:
                print("Please enter exactly two numbers (e.g. `3 5` or `3,5`).")
                continue
            try:
                self.state.n1 = int(parts[0])
                self.state.n2 = int(parts[1])
                break
            except ValueError:
                print("Invalid input. Please enter numeric values for both entries.")

    @listen(get_user_input)
    def sum_two_numbers(self):
        result = (
            SumCrew()
            .crew()
            .kickoff(inputs={"n1": self.state.n1, "n2": self.state.n2})
        )
        print("The result is:", result)


def kickoff():
    poem_flow = SumFlow()
    poem_flow.kickoff()


def plot():
    poem_flow = SumFlow()
    poem_flow.plot()


if __name__ == "__main__":
    kickoff()
    plot()
