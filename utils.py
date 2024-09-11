from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.globals import set_llm_cache, set_debug
from langchain.cache import InMemoryCache, SQLiteCache
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
import streamlit as st

import os
import time

# [ import-guide using relative path ]
# import sys, os, importlib
# st.write(os.getcwd()) # check current path 
# parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '{relative path}'))
# sys.path.append(parent_dir)
# import utils
# importlib.reload(utils)
# from utils import Utils

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

    @st.cache_data(show_spinner="Embedding file...")
    @staticmethod
    def embed_file(file):
        print("run. embed_file")
        file_path = f"./.cache/files/{file.name}"
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "wb") as f:
            f.write(file.read())
        cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
        embeddings = OpenAIEmbeddings()
        cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
        
        splitter = CharacterTextSplitter.from_tiktoken_encoder(
            separator="\n",
            chunk_size=600,
            chunk_overlap=100,
        )        
        loader = UnstructuredFileLoader(file_path)
        # print(loader.load())
        docs = loader.load_and_split(text_splitter=splitter)
        
        vectorstore = FAISS.from_documents(docs, cached_embeddings)
        retriever = vectorstore.as_retriever()
        print("end. embed_file")
        return retriever
    
    def format_docs(docs):
        return "\n\n".join(document.page_content for document in docs)
    
    @st.cache_data()
    @staticmethod
    def get_prompt(header_system="", header_question=""):
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    header_system + "\n"
                    """                    
                    context : {context}
                    """,
                ),
                (
                    "human", 
                    header_question + "\n"
                    """
                    question : {question}
                    """
                ),
            ]
        )
        return prompt
    
    @staticmethod
    def chain_invoke_stuff(retriever, llm, message, prompt):
        print("chain_invoke_stuff")
        chain = chain = (
            {
                "context": retriever | RunnableLambda(Utils.format_docs),
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
        )
        return chain.invoke(message)
        
        