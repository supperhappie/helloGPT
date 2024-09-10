from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.globals import set_llm_cache, set_debug
from langchain.cache import InMemoryCache, SQLiteCache
import os
import time

class Utils:
    cache_enabled = False
    debug_enabled = False
    cache_type = None  # None, 'inmemory', or 'sqlite'
    cache_path = "cache.db"  # Default SQLite cache path

    @staticmethod
    def enable_cache(cache_type='sqlite', cache_path=None):
        """Enable caching with either InMemoryCache or SQLiteCache."""
        Utils.cache_enabled = True
        Utils.cache_type = cache_type

        if cache_type == 'inmemory':
            set_llm_cache(InMemoryCache())
        elif cache_type == 'sqlite':
            Utils.cache_path = cache_path or Utils.cache_path
            set_llm_cache(SQLiteCache(Utils.cache_path))
        else:
            raise ValueError("Invalid cache type. Choose 'inmemory' or 'sqlite'.")
        print(f"Cache enabled: {cache_type} (Path: {Utils.cache_path if cache_type == 'sqlite' else 'In-Memory'})")

    @staticmethod
    def disable_cache():
        """Disable cache."""
        Utils.cache_enabled = False
        set_llm_cache(None)
        print("Cache disabled.")

    @staticmethod
    def enable_debug():
        """Enable debug mode."""
        Utils.debug_enabled = True
        set_debug(True)
        print("Debugging enabled.")

    @staticmethod
    def disable_debug():
        """Disable debug mode."""
        Utils.debug_enabled = False
        set_debug(False)
        print("Debugging disabled.")

    @staticmethod
    def get_chat_gpt_4o_mini(streaming=False):
        """Get GPT-4o-mini model. Optionally enable streaming output."""
        chat = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.1,
            streaming=streaming,
            callbacks=[StreamingStdOutCallbackHandler()] if streaming else None,
        )
        return chat

    @staticmethod
    def get_chat_gpt_3_5_turbo(streaming=False):
        """Get GPT-3.5-turbo model. Optionally enable streaming output."""
        chat = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.1,
            streaming=streaming,
            callbacks=[StreamingStdOutCallbackHandler()] if streaming else None,
        )
        return chat

    @staticmethod
    def get_llm_davinci_002():
        """Get davinci-002 model."""
        llm = OpenAI(
            model="davinci-002"
        )
        return llm

    @staticmethod    
    def track_ai_command(command, *args, verbose=False):
        """Track and print AI command usage with callbacks for token tracking.
           Prints the execution time if verbose is True.
        """
        if verbose:
            # Print the function name of the command
            print(f"Tracking AI command... {command.__name__}")
            start_time = time.time()  # Start tracking the time
        
        with get_openai_callback() as usage:
            out = command(*args)
        
        if verbose:
            print(out)
            print("-" * 100)
            print(usage)
            
            end_time = time.time()  # End tracking the time
            execution_time = end_time - start_time
            print(f"Execution Time: {execution_time:.2f} seconds")
        
        return out

    @staticmethod    
    def get_current_path():
        """Return the current working directory."""
        return os.getcwd()
