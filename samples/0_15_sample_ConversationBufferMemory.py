from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory()
memory.save_context({"input":"Hi!"}, {"output":"How are you?"})
memory.save_context({"input":"Hi!"}, {"output":"How are you?"})
memory.save_context({"input":"Hi!"}, {"output":"How are you?"})
print(memory.load_memory_variables({}))

# print(memory.load_memory_variables({}).get('history'))

memory = ConversationBufferMemory(return_messages=True) # for chat model 
memory.save_context({"input":"Hi!"}, {"output":"How are you?"})
memory.save_context({"input":"Hi!"}, {"output":"How are you?"})
memory.save_context({"input":"Hi!"}, {"output":"How are you?"})
memory.load_memory_variables({})

