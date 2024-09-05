from langchain.chat_models import ChatOpenAI
chat = ChatOpenAI(model="gpt-4o-mini", temperature= 0.1)

from langchain.prompts import PromptTemplate, ChatPromptTemplate

template = ChatPromptTemplate.from_messages([
    ("system", "you are a list generating machine, Everything you are asked will be answered with a list of max {max_items}. Do not reply with anything else. your response must be seperated by comma "),
    # ('ai', "I'm ai"),
    ('human', "{question}")
])

prompt = template.format_messages(
    max_items = 10, 
    question = "country list in this world"
)

print(prompt)

response = chat.predict_messages(prompt)
print(response.content)

from langchain.schema import BaseOutputParser

class CommaOutputParser(BaseOutputParser):
    def parse(self, text):
        items = text.strip().split(",")
        return list(map(str.strip, items))
    
p = CommaOutputParser()
response_parsed = p.parse(response.content)

