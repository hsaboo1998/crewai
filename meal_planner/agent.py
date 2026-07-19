from crewai import Agent, Task
from crewai.project import CrewBase
from crewai.project.annotations import agent, task

@CrewBase
class MealPlannerCrew:
    def __init__(self, llm, output, output_file, serper_dev_tool):
        self.llm = llm
        self.output = output
        self.output_file = output_file
        self.serper_dev_tool = serper_dev_tool
    @agent
    def meal_planner_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["meal_planner_agent"],
            llm=self.llm,
            tools=[self.serper_dev_tool]
        )
    @task
    def meal_planning_task(self) -> Task:
        return Task(
            config=self.tasks_config['meal_planning_task'],
            agent=self.meal_planner_agent(),
            output_pydantic=self.output,
            output_file=self.output_file
        )