from langchain.memory import ConversationSummaryBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

llm = ChatOpenAI(temperature=0.1)

memory = ConversationSummaryBufferMemory(
    llm = llm,
    max_token_limit=80, 
    memory_key="chat_history",
    return_messages=True
)

prompt=ChatPromptTemplate.from_messages([
    ("system", "You are helpful AI talking to a human"),
    MessagesPlaceholder(variable_name="chat_history"),
    ('human', "{question}"),
])

chain = LLMChain(
    llm=llm,
    memory=memory,
    prompt = prompt,
    verbose=True
)

print(chain.predict(question="I'm noah"))
print(chain.predict(question="What is my name?"))

print(memory.load_memory_variables({}))
