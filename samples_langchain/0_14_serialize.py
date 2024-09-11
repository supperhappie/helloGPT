from langchain.llms.openai import OpenAI
from langchain.llms.loading import load_llm

chat = OpenAI(model="gpt-3.5-turbo", 
                  temperature= 0.1,                  
                  max_tokens=450
                )

chat.save("model.json")

chat = load_llm("model.json")
print(chat)