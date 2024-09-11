from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate

chat = ChatOpenAI(model="gpt-4o-mini", temperature= 0.1)

messages = ChatPromptTemplate.from_messages([
    ("system", "You are a geography expert. And you only reply in {language}"),
    ("ai", "안녕, 난 {name}"),
    ("human", "What is the distance between {country_a} and {country_b}. Also, what is your name?")
])

prompt = messages.format_messages(language = "korean", name = "AI",country_a = "korea", country_b = "maxico")

response = chat.predict_messages(prompt)

print(response)