from crewai import Agent, Task
from crewai.project import CrewBase
from crewai.project.annotations import agent, task

@CrewBase
class BudgetAdvisorCrew:
    def __init__(self, llm, output_file, serper_dev_tool):
        self.llm = llm
        self.output_file = output_file
        self.serper_dev_tool = serper_dev_tool
    @agent
    def budget_advisor_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["budget_advisor_agent"],
            llm=self.llm,
            tools=[self.serper_dev_tool]
        )
    @task
    def budget_advisor_task(self) -> Task:
        return Task(
            config=self.tasks_config['budget_advisor_task'],
            agent=self.budget_advisor_agent(),
            output_file=self.output_file
        )