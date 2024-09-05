from typing import Any, Dict, List
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.prompts.example_selector.base import BaseExampleSelector
from random import choice

class RandomExampleSelector(BaseExampleSelector):
    def __init__(self, examples) -> None:
        super().__init__()
        self.examples = examples
        
    def add_example(self, example):
        self.examples.append(example)
        
    def select_examples(self, input_variables):
        # Return a list containing one random example
        return [choice(self.examples)]

# Define the OpenAI chat model
chat = ChatOpenAI(model="gpt-4o-mini", 
                  temperature=0.1,
                  streaming=True,
                  callbacks=[StreamingStdOutCallbackHandler(),])

# Define the examples
examples = [
    {    
        "country" : "France",
        "answer": """
        Here is what I know: 
        - Capital: Paris
        - Language: French
        - Food: Wine and Cheese
        - Currency: Euro
        """,
    },
    {    
        "country" : "Korea",
        "answer": """
        Here is what I know: 
        - Capital: Seoul
        - Language: Korean
        - Food: Kimchi and Bulgogi
        - Currency: Won
        """,
    },
]

# Define the template for the prompt
example_template = PromptTemplate.from_template("human: What do you know about {country}?\nai: {answer}")

# Create the example selector
example_selector = RandomExampleSelector(examples=examples)

# Define the few-shot prompt template
prompt = FewShotPromptTemplate(
    example_prompt=example_template,
    example_selector=example_selector,
    suffix="human: What do you know about {country}?",
    input_variables=["country"]
)

# Format the prompt for a specific input
out = prompt.format(country="Brazil")

# Print the result
print(out)
