from langchain.memory import ConversationSummaryBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnablePassthrough

llm = ChatOpenAI(temperature=0.1, model="gpt-3.5-turbo")

memory = ConversationSummaryBufferMemory(
    llm = llm,
    max_token_limit=80, 
    memory_key="chat_history",
    return_messages=True,
)

prompt=ChatPromptTemplate.from_messages([
    ("system", "You are helpful AI talking to a human"),
    MessagesPlaceholder(variable_name="chat_history"),
    ('human', "{question}"),
])

def load_memory(_):
    return memory.load_memory_variables({})['chat_history']

#RunnablePassthrough is part of the Langchain framework, used to forward data without modifying it in the chain of operations.
chain = RunnablePassthrough.assign(chat_history=load_memory)|prompt|llm

def invoke_chain(chain, question):
    result = chain.invoke({"question": question})
    memory.save_context({"input":question}, {"outputs":result.content},)
    print(result)
    
invoke_chain(chain=chain, question="My name is noah")
invoke_chain(chain=chain, question="What is my name?")

print(memory.load_memory_variables({}))

