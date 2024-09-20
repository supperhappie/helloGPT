from typing import Any, Type
from langchain.chat_models import ChatOpenAI
from langchain.tools import StructuredTool, Tool, BaseTool
from pydantic import BaseModel, Field
from langchain.agents import initialize_agent, AgentType

llm = ChatOpenAI(temperature=0.1) 
# this process use function call of open ai api

class CalculatorToolArgsSchema(BaseModel):
    a: float = Field(description="The first number")
    b: float = Field(description="The second number")
    
class CalculatorTool(BaseTool):
    name = "CalculatorTool"
    description = """
    Use this to perform sums of two numbers.
    The first and second arguments should be numbers.
    Only receives two arguments.
    """
    args_schema: Type[CalculatorToolArgsSchema] = CalculatorToolArgsSchema

    def _run(self, a, b):
        return a + b
    
agent = initialize_agent(
    llm=llm,
    verbose=True,
    agent=AgentType.OPENAI_FUNCTIONS,
    handle_parsing_errors=True,
    tools=[
        CalculatorTool()
    ],
)

prompt = "Total Cost of $355.39 + $924.87 + $721.2 + $1940.29 + $573.63 + $65.72 + $35.00 + $552.00 + $76.16 + $29.12"

out = agent.invoke(prompt)
print (out)