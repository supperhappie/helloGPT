from typing import Any, Dict, List
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.prompts.few_shot import FewShotPromptTemplate # give some example to the model 
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.prompts import ChatPromptTemplate
from langchain.prompts.example_selector import LengthBasedExampleSelector
from langchain.prompts.example_selector.base import BaseExampleSelector

class RandomExampleSelector(BaseExampleSelector):
    def __init__(self, examples):
        self.examples = examples
        
    def add_example(self, example):
        self.examples.append(example)
        
    def select_examples(self, input_variables):
        # return super().select_examples(input_variables)
        from random import choice
        return [choice(self.examples)]

chat = ChatOpenAI(model="gpt-4o-mini", 
                  temperature= 0.1,
                  streaming=True,
                  callbacks=[StreamingStdOutCallbackHandler(),])

examples = [
    {    
        "country" : "France",
        "answer": """
        Here is what I know : 
        - Capital : Paris
        - Language : French
        - Food : Wine and Cheese
        - Currency : Euro
        """,
    },
    {    
        "country" : "Korea",
        "answer": """
        Here is what I know : 
        - Capital : Seuol
        - Language : Korean
        - Food : Kimchi and boolbag
        - Currency : won
        """,
    },
    ]

# example_templit = ChatPromptTemplate.from_messages([
#     ("human", "What do you know about {country}?"),
#     ("ai", "{answer}")
# ])

example_templit = PromptTemplate.from_template("human:What do you know about {country}?\nai:{answer}")

# example_selector = LengthBasedExampleSelector(
#     examples=examples,
#     example_prompt=example_templit,
#     max_length=100
# )

example_selector = RandomExampleSelector(examples=examples)

prompt = FewShotPromptTemplate(
    example_prompt=example_templit,
    # examples=examples,
    example_selector=example_selector,
    suffix="human: What do you know about {country}?",
    input_variables=["country"]
)

out = prompt.format(country = "Brazil")

# chain  = final_prompt | chat
# out = chain.invoke({
#     "country":"Germany"
# })

print(out)