from crewai import Agent, Task
from crewai.project import CrewBase
from crewai.project.annotations import agent, task

@CrewBase
class SummaryCrew:
    def __init__(self, llm, output_file):
        self.llm = llm
        self.output_file = output_file
    @agent
    def summary_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["summary_agent"],
            llm=self.llm
        )
    @task
    def summary_task(self) -> Task:
        return Task(
            config=self.tasks_config['summary_task'],
            agent=self.summary_agent(),
            output_file=self.output_file
        )
