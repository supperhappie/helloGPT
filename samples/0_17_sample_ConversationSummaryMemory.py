from langchain.memory import ConversationSummaryMemory
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(temperature=0.1, model="gpt-4o-mini")

memory = ConversationSummaryMemory(llm=llm)


def add_message(input, output):
    memory.save_context({"input": input}, {"output": output})


def get_history():
    return memory.load_memory_variables({})


add_message("Hi I'm Nicolas, I live in South Korea", "Wow that is so cool!")
add_message("South Kddorea is so pretty", "I wish I could go!!!")

print(get_history())