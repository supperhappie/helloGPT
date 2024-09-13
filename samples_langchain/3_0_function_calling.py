from langchain.prompts import PromptTemplate
import sys, os, importlib

# [ import-guide using relative path ]
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

sys.path.append(parent_dir)
import langchain_utilities
importlib.reload(langchain_utilities)
from langchain_utilities import LangChainUtilities

def get_weather(lon, lat):
    print(f"lon : {lon}")
    print(f"lat : {lat}")
    print("call an api...")
LangChainUtilities.enable_cache()
# how llm recognize and use function ? It is json schema(function)! 

function = {
    "name": "get_weather", # Function name
    "description":"function that takes longitude and latitude to find the weather of a place", # Function description. Yes, it's for AI to understand and use.
    "parameters":{
        "type":"object",
        "properties":{
            "lon":{
                "type":"number",
                "description":"The longitude coordinate" 
            },
            "lat":{
                "type":"number",
                "description":"The latitude coordinate" 
            },
        },
    },
    "required":["lon", "lat"]
}

llm = LangChainUtilities.get_chat_gpt_3_5_turbo(temperature=0.1).bind(
    function_call={
        "name":"get_weather"
        # "auto" # llm choice automatically using binding functions
    },
    functions=[
        function
    ]
)

prompt = PromptTemplate.from_template("how is the weather in {city}")
chain = prompt|llm
response = chain.invoke({"city":"seoul"})
print(response)
print("-"*30)
response = response.additional_kwargs["function_call"]["arguments"]
print(response)
print("-"*30)
