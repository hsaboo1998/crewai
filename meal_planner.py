import certifi
import os
os.environ['SSL_CERT_FILE'] = certifi.where()
from dotenv import load_dotenv
load_dotenv()
from pydantic import BaseModel, Field
from crewai_tools import SerperDevTool
from crewai import Crew, Process
from crewai import LLM
from typing import List
from meal_planner.agent import MealPlannerCrew
from shopping_organizer.agent import ShoppingOrganizerCrew
from budget_advisor.agent import BudgetAdvisorCrew
from summary.agent import SummaryCrew

class GroceryItem(BaseModel):
    """Individual grocery item"""
    name: str = Field(description="Name of item")
    quantity: str = Field(description="Quantity needed for example 3kgs etc")
    estimated_price: str = Field(description="Estimated price for example Rs 3-5")
    category: str = Field(description="Store section (for example 'Produce', 'Dairy')")

class ShoppingCategory(BaseModel):
    """Store section with items"""
    section_name: str = Field(description="Store section for example 'Produce', 'Dairy")
    items: List[GroceryItem] = Field(description="Items in section")
    estimated_total: str = Field(description="Estimated cost for this section based on the cost of grocery items")

class MealPlan(BaseModel):
    """Simple Meal Plan"""
    meal_name: str = Field(description="Name of the meal")
    difficulty_level: str = Field(description="Easy or Medium or Hard")
    servings: str = Field(description="No. of people it serves")
    researched_ingredients: str = Field(description="Ingredients through research")

class GroceryShoppingPlan(BaseModel):
    total_budget: str = Field(description="total budget based on the shopping categories estimated cost")
    meal_plans: List[MealPlan] = Field(description="Planned Meals")
    shopping_sections: List[ShoppingCategory] = Field(description="Organized by store sections")
    shopping_tips: List[str] = Field(description="Money saving efficiency tips")

llm = LLM(
    model="gemini/gemini-3.5-flash",
    api_key=os.getenv('GEMINI_API_KEY')
)

serper_dev_tool = SerperDevTool()

meal_planner_crew = MealPlannerCrew(llm, MealPlan, 'meals.json', serper_dev_tool)
meal_planner_agent = meal_planner_crew.meal_planner_agent()
meal_planner_task = meal_planner_crew.meal_planning_task()

shopping_organizer_crew = ShoppingOrganizerCrew(llm, GroceryShoppingPlan, 'shopping_list.json')
shopping_organizer_agent = shopping_organizer_crew.shopping_organizer_agent()
shopping_organizer_task = shopping_organizer_crew.shopping_task()

budget_advisor_crew = BudgetAdvisorCrew(llm, 'budget_advise.json', serper_dev_tool)
budget_advisor_agent = budget_advisor_crew.budget_advisor_agent()
budget_advisor_task = budget_advisor_crew.budget_advisor_task()

summary_crew = SummaryCrew(llm, 'summary.json')
summary_agent = summary_crew.summary_agent()
summary_task = summary_crew.summary_task()

crew = Crew(
    agents=[meal_planner_agent, shopping_organizer_agent, budget_advisor_agent, summary_agent],
    tasks=[meal_planner_task, shopping_organizer_task, budget_advisor_task, summary_task],
    process=Process.sequential,
    verbose=True
)
result = crew.kickoff(
    inputs={
        "meal_name":"Plain Maggi",
        "servings": 4, 
        "budget": "Rs 100",
        "dietary_restrictions": "no ghee or oil",
        "cooking_skill": "beginner"
    }
)
print(result)
