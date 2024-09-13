from langchain.chat_models import ChatOpenAI
chat = ChatOpenAI(model="gpt-4o-mini", temperature= 0.1)

from langchain.prompts import PromptTemplate, ChatPromptTemplate

template = ChatPromptTemplate.from_messages([
    ("system", "you are a list generating machine, Everything you are asked will be answered with a list of max {max_items}. Do not reply with anything else. your response must be seperated by comma "),
    # ('ai', "I'm ai"),
    ('human', "{question}")
])


from langchain.schema import BaseOutputParser

class CommaOutputParser(BaseOutputParser):
    def parse(self, text):
        items = text.strip().split(",")
        return list(map(str.strip, items))


chain = template | chat | CommaOutputParser() # LLMChain 객체(chain)가 생성됨. or 연산을 overiding 하여 이런식으로 객체 생성하는 것이 요즘(?) python 트렌드라 함. 
out_chain = chain.invoke({
    "max_items":10, 
    "question":"poketmon list in this world"
})

print(out_chain)
print(type(out_chain))