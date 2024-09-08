from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.globals import set_llm_cache, set_debug
from langchain.cache import InMemoryCache, SQLiteCache

class Utils:
    def __init__(self):
        self.cache_enabled = False
        self.debug_enabled = False
        self.cache_type = None  # None, 'inmemory', or 'sqlite'
        self.cache_path = "cache.db"  # Default SQLite cache path

    def enable_cache(self, cache_type='sqlite', cache_path=None):
        """Enable caching with either InMemoryCache or SQLiteCache."""
        self.cache_enabled = True
        self.cache_type = cache_type

        if cache_type == 'inmemory':
            set_llm_cache(InMemoryCache())
        elif cache_type == 'sqlite':
            self.cache_path = cache_path or self.cache_path
            set_llm_cache(SQLiteCache(self.cache_path))
        else:
            raise ValueError("Invalid cache type. Choose 'inmemory' or 'sqlite'.")
        print(f"Cache enabled: {cache_type} (Path: {self.cache_path if cache_type == 'sqlite' else 'In-Memory'})")

    def disable_cache(self):
        """Disable cache."""
        self.cache_enabled = False
        set_llm_cache(None)
        print("Cache disabled.")

    def enable_debug(self):
        """Enable debug mode."""
        self.debug_enabled = True
        set_debug(True)
        print("Debugging enabled.")

    def disable_debug(self):
        """Disable debug mode."""
        self.debug_enabled = False
        set_debug(False)
        print("Debugging disabled.")
    
    def get_chat_gpt_4o_mini(self, streaming=False):
        """Get GPT-4o-mini model. Optionally enable streaming output."""
        chat = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.1,
            streaming=streaming,
            callbacks=[StreamingStdOutCallbackHandler()] if streaming else None,
        )
        return chat
    
    def get_chat_gpt_3_5_turbo(self, streaming=False):
        """Get GPT-3.5-turbo model. Optionally enable streaming output."""
        chat = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.1,
            streaming=streaming,
            callbacks=[StreamingStdOutCallbackHandler()] if streaming else None,
        )
        return chat

    def get_llm_davinci_002(self):
        """Get davinci-002 model."""
        llm = OpenAI(
            model="davinci-002"
        )
        return llm
        
    def track_ai_command(self, command, *args):
        """Track and print AI command usage with callbacks for token tracking."""
        with get_openai_callback() as usage:
            out = command(*args)
            print(out)
            print("-" * 100)
            print(usage)
        return out
            
    def get_current_path(self):
        """Return the current working directory."""
        import os
        return os.getcwd()

# Example usage
utils = Utils()

# Enable SQLite cache and debug
utils.enable_cache(cache_type='sqlite')
utils.enable_debug()

# Use the chat model
chat = utils.get_chat_gpt_4o_mini()
utils.track_ai_command(chat.predict, "How do you make italian pasta. short response.")
utils.track_ai_command(chat.predict, "How do you make italian pasta. short response.")

# Disable cache and debug after use
utils.disable_cache()
utils.disable_debug()
