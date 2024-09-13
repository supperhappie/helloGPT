import os
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

# Load environment variables from .env file
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

llm = OpenAI(openai_api_key=openai_api_key, model="davinci-002")
chat = ChatOpenAI(openai_api_key=openai_api_key, model="gpt-4o-mini")

request = "How many planet are there?"

out_OpenAI = llm.predict(request)
# out_chatOpenAI = chat.predict(request)

print('request : ' + request)
print("-"*20)
print('llm : model. davinci-002')
print(out_OpenAI)
print("-"*20)
print('chat : model. gpt-4o-mini')
print(out_chatOpenAI)