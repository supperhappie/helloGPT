from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.prompts.few_shot import FewShotChatMessagePromptTemplate # give some example to the model 
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.prompts import ChatPromptTemplate


chat = ChatOpenAI(model="gpt-4o-mini", 
                  temperature= 0.1,
                  streaming=True,
                  callbacks=[StreamingStdOutCallbackHandler(),])

examples = [
    {    
        "country" : "France?",
        "answer": """
        Here is what I know : 
        - Capital : Paris
        - Language : French
        - Food : Wine and Cheese
        - Currency : Euro
        """,
    },
    {    
        "country" : "Korea?",
        "answer": """
        Here is what I know : 
        - Capital : Seuol
        - Language : Korean
        - Food : Kimchi and boolbag
        - Currency : won
        """,
    },
    ]

example_templit = ChatPromptTemplate.from_messages([
    ("human", "What do you know about {country}?"),
    ("ai", "{answer}")
])

example_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_templit,
    examples=examples,
)

final_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a geography expert. you give short answer. completely follow examples"),
    example_prompt,
    ("human", "What do you know about {country}?")
])

chain  = final_prompt | chat
out = chain.invoke({
    "country":"Germany"
})

print(out)