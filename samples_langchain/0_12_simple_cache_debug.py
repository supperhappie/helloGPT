from langchain.chat_models import ChatOpenAI
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.globals import set_llm_cache, set_debug
from langchain.cache import InMemoryCache, SQLiteCache

chat = ChatOpenAI(model="gpt-4o-mini", 
                  temperature= 0.1,
                #   streaming=True,
                #   callbacks=[StreamingStdOutCallbackHandler(),]
                )

# set_llm_cache(InMemoryCache()) 
set_llm_cache(SQLiteCache("cache.db")) 
set_debug(True)

chat.predict("How do you make Maxico pasta. short response.")
chat.predict("How do you make Maxico pasta. short response.")