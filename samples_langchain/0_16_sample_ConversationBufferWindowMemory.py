from langchain.memory import ConversationBufferWindowMemory

memory = ConversationBufferWindowMemory(
    return_messages = True,
    k=4
)

def add_message(input, output):
    memory.save_context({"input":input}, {"output":output})
    
for i in range(10):
    add_message(i,i)

memory.load_memory_variables({})