from crewai import Agent, Task
from crewai.project import CrewBase
from crewai.project.annotations import agent, task

@CrewBase
class ShoppingOrganizerCrew:
    def __init__(self, llm, output, output_file):
        self.llm = llm
        self.output = output
        self.output_file = output_file
    @agent
    def shopping_organizer_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["shopping_organizer_agent"],
            llm=self.llm
        )
    @task
    def shopping_task(self) -> Task:
        return Task(
            config=self.tasks_config['shopping_task'],
            agent=self.shopping_organizer_agent(),
            output_pydantic=self.output,
            output_file=self.output_file
        )