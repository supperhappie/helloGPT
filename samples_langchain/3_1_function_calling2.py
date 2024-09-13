from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
import sys, os, importlib
# [ import-guide using relative path ]
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import langchain_utilities
importlib.reload(langchain_utilities)
from langchain_utilities import LangChainUtilities

LangChainUtilities.enable_cache()

function = {
    "name": "create_quiz",
    "description": "function that takes a list of questions and answers and returns a quiz",
    "parameters": {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                        },
                        "answers": {
                            "type": "array",
                            "items": {  # items 속성은 1개 이상의 input 을 의미함. 
                                "type": "object",
                                "properties": {
                                    "answer": {
                                        "type": "string",
                                    },
                                    "correct": {
                                        "type": "boolean",
                                    },
                                },
                                "required": ["answer", "correct"],
                            },
                            "minItems": 5,
                            "maxItems": 5
                        },
                    },
                    "required": ["question", "answers"],
                },
            }
        },
        "required": ["questions"],
    },
}


llm = LangChainUtilities.get_chat_gpt_3_5_turbo(temperature=0.1).bind(
        function_call={
            "name": "create_quiz",
        },
        functions=[
            function,
        ],
    )

prompt = PromptTemplate.from_template("Make a quiz about {city}")
chain = prompt | llm

response = chain.invoke({"city": "rome"})
response = response.additional_kwargs["function_call"]["arguments"]

import json

for question in json.loads(response)["questions"]:
    print(question["question"])
    for ans in question['answers']:
	    print("\t" + ("(o)" if ans["correct"] else "(x)") + "\t" + str(ans["answer"]))

