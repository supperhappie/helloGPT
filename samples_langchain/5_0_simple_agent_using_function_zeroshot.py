from langchain.chat_models import ChatOpenAI
from langchain.tools import StructuredTool, Tool
from langchain.agents import initialize_agent, AgentType

# I trired to use ollama mistral model, but it is not running expected. 
# I think for the 

llm = ChatOpenAI(temperature=0.1) 

def plus(a, b):  # StructuredTool
# def plus(inputs):     # Tool
    # print(inputs)
    # a, b = map(float, inputs.split(","))
    result = a + b
    print(f"a + b = {result}")
    return result 

agent = initialize_agent(
    llm=llm,
    verbose=True,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    tools=[
        StructuredTool.from_function(
        # Tool.from_function(
            func=plus,
            name="Sum Calculator",
            description="Use this to perform sums of two numbers. This tool take ONLY TWO ARGUMENTS, both  should be numbers.",
            # description="USE THIS TO PERFORM SUMS OF ONLY TWO NUMBERS. USE THIS TOOL BY SENDING A PAIR OF NUMBER SEPERATED BY A COMMA. \n EXAMPLE 1, 2.",
        ),
    ],
)

prompt = "Total Cost of $355.39 + $924.87 + $721.2 + $1940.29 + $573.63 + $65.72 + $35.00 + $552.00 + $76.16 + $29.12."

out = agent.invoke(prompt)
print (out)