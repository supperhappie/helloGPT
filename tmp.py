from langchain.memory import ConversationSummaryBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# Initialize LLM
llm = ChatOpenAI(temperature=0.1, model="gpt-4")

# Initialize memory buffer
memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=80,
    memory_key="chat_history",
    return_messages=True,
)

# Define prompt template
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI talking to a human."),
    ('human', "{question}")
])

# Load memory and pass it manually
def load_memory():
    return memory.load_memory_variables({})['chat_history']

# Chain execution
def invoke_chain(question):
    chat_history = load_memory()  # Load chat history
    prompt = prompt_template.format_messages(chat_history=chat_history, question=question)  # Apply template
    print(prompt)
    response = llm(prompt)  # Send to LLM
    
    memory.save_context({"input": question}, {"output": response.content})  # Save memory
    print(response.content)  # Output result

# Example usage
invoke_chain(question="My name is Noah")
invoke_chain(question="What is my name?")

print(memory.load_memory_variables({}))