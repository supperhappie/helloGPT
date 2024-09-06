from langchain.memory import ConversationSummaryBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback

llm = ChatOpenAI(temperature=0.1)

memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=100,
    return_messages=True,
)


def add_message(input, output):
    memory.save_context({"input": input}, {"output": output})


def get_history():
    return memory.load_memory_variables({})


add_message("Hi I'm Nicolas, I live in South Korea", "Wow that is so cool!")

with get_openai_callback() as usage:
    add_message("How far is Brazil from Argentina?", "I don't know! Super far!")
    print(get_history())
    print(usage)
        
with get_openai_callback() as usage:
    add_message("How far is Brazil from Argentina?", "I don't know! Super far!")
    print(get_history())
    print(usage)
    
with get_openai_callback() as usage: 
    # if the lmit is over, \summary is requested. and additional biling is occured. 
    add_message("How far is Brazil from Argentina?", "I don't know! Super far!")
    print(get_history())
    print(usage)
    
    