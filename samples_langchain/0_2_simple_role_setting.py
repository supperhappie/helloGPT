from langchain.chat_models import ChatOpenAI
chat = ChatOpenAI(model="gpt-4o-mini")

from langchain.schema import HumanMessage, AIMessage, SystemMessage

messages = [
    SystemMessage(content="You are a geography expert. And you only reply in korean"), # SystemMessage : AI role 설정 
    AIMessage(content="안녕, 난 지리는 전문가야 "), # AIMessage : AI 가 초기에 보냈다고 가정하는 메시지 
    HumanMessage(content="What is the distance between Maxico and Thailand. Also, what is your name?") # 사용자 메시지 
]

response = chat.predict_messages(messages)

print(response)
