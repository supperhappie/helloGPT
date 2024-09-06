from langchain.memory import ConversationSummaryBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

llm = ChatOpenAI(temperature=0.1)

memory = ConversationSummaryBufferMemory(
    llm = llm,
    max_token_limit=80, 
    memory_key="chat_history"
)

template="""    
    {chat_history}
    human:{question}
    You:
"""

chain = LLMChain(
    llm=llm,
    memory=memory,
    prompt = PromptTemplate.from_template(template),
    verbose=True
)

print(chain.predict(question="I'm noah"))
print(chain.predict(question="What is my name?"))
print(memory.load_memory_variables({}))

