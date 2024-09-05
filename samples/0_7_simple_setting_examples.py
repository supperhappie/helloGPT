from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.prompts.few_shot import FewShotPromptTemplate # give some example to the model 
from langchain.callbacks import StreamingStdOutCallbackHandler

chat = ChatOpenAI(model="gpt-4o-mini", 
                  temperature= 0.1,
                  streaming=True,
                  callbacks=[StreamingStdOutCallbackHandler(),])

examples = [
    {    
        "question" : "What do you know about France?",
        "answer": """
        Here is what I know : 
        - Capital : Paris
        - Language : French
        - Food : Wine and Cheese
        - Currency : Euro
        """,
    },
    {    
        "question" : "What do you know about Korea?",
        "answer": """
        Here is what I know : 
        - Capital : Seuol
        - Language : Korean
        - Food : Kimchi and boolbag
        - Currency : won
        """,
    },
    ]

example_template = """
    Human:{question}
    AI:{answer}
"""

example_prompt = PromptTemplate.from_template(example_template)
prompt = FewShotPromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
    suffix="Human: What do you know about {country}?",
    input_variables=["country"]
)

chain  = prompt | chat
out = chain.invoke({
    "country":"Germany"
})