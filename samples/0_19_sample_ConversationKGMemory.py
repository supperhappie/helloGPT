from langchain.memory import ConversationKGMemory
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback
from utils import Utils

utils = Utils()
llm = ChatOpenAI(model="gpt-4o-mini", temperature= 0.1)
# llm = ChatOpenAI(temperature=0.1)

memory = ConversationKGMemory(
    llm=llm,
    return_messages=True,
)


def add_message(input, output):
    memory.save_context({"input": input}, {"output": output})


utils.track_ai_command(add_message, "Hi I'm Nicolas, I live in South Korea", "Wow that is so cool!")
utils.track_ai_command(memory.load_memory_variables, {"input": "who is Nicolas"})
utils.track_ai_command(add_message, "Nicolas likes kimchi", "Wow that is so cool!")
utils.track_ai_command(memory.load_memory_variables, {"inputs": "what does nicolas like"})
print(memory)