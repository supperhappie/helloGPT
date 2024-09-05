from langchain.chat_models import ChatOpenAI
from langchain.callbacks import StreamingStdOutCallbackHandler
chat = ChatOpenAI(model="gpt-4o-mini", 
                  temperature= 0.1,
                  streaming=True,
                  callbacks=[StreamingStdOutCallbackHandler(),])
# callbacks : StreamingStdOutCallbackHandler >> 실시간으로 결과 확인  

from langchain.prompts import PromptTemplate, ChatPromptTemplate

template_invest_analyzer = ChatPromptTemplate.from_messages([
    ("system", "You are a world-class international stock analysis expert. You create easy recipies to follow trading for beginner of investment"),
    # ('ai', "I'm ai"),
    ('human', "How about {stock_name}? summarize it. contain current value($), expected value(seperate it as next day, next week, next month, next year, next 5 years), nowdays issues, risk(describe as 100 percent and explain it) ")
])

template_invest_trader = ChatPromptTemplate.from_messages([
    ("system", "You are a world-class international stock traider expert. You check Affordability and financial stability to buy or sell target stocks. you should say like that I suggest sell(or buy) the stock(it will be changed as target stock name) [rate as percentage], after two enter. explain detail reasons. maybe human give you investment analyze report, refer it to demand"),
    # ('ai', "I'm ai"),
    ('human', "Refer to {analyze_report}, suggest me the result (sell or buy) and explain. Suggest me the result(sell or buy)")
])

chain_invest_analyzer = template_invest_analyzer | chat 
chain_invest_trader = template_invest_trader | chat
final_chain = {"analyze_report": chain_invest_analyzer} | chain_invest_trader
print( final_chain.invoke({
    "stock_name":"google"
}) )
