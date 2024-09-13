from langchain.chat_models import ChatOpenAI, ChatOllama
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.globals import set_llm_cache, set_debug
from langchain.cache import InMemoryCache, SQLiteCache
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings, OllamaEmbeddings
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
import streamlit as st

import os
import time

class LangChainUtilities:
    cache_enabled = False
    debug_enabled = False
    verbose_enabled = False
    cache_type = None  # None, 'inmemory', or 'sqlite'
    cache_path = "cache.db"  # Default SQLite cache path
    embedding_model = {"embedding":OllamaEmbeddings(model="mistral"), "model":"mistral"}  # Store the embedding model
    
    @staticmethod
    def enable_cache(cache_type='sqlite', cache_path=None):
        """캐시를 InMemoryCache 또는 SQLiteCache로 활성화합니다."""
        지원되는_캐시_유형 = ['inmemory', 'sqlite']
        
        LangChainUtilities.cache_enabled = True
        LangChainUtilities.cache_type = cache_type

        if cache_type.lower() == 'inmemory':
            set_llm_cache(InMemoryCache())
        elif cache_type.lower() == 'sqlite':
            LangChainUtilities.cache_path = cache_path or LangChainUtilities.cache_path
            set_llm_cache(SQLiteCache(LangChainUtilities.cache_path))
        else:
            if LangChainUtilities.verbose_enabled:
                print(f"{cache_type}은(는) 지원되지 않습니다. 지원되는 캐시 유형 목록: {지원되는_캐시_유형}")
            LangChainUtilities.cache_type = 'sqlite'  # 기본값으로 설정
            LangChainUtilities.cache_path = cache_path or LangChainUtilities.cache_path
            set_llm_cache(SQLiteCache(LangChainUtilities.cache_path))
        
        if LangChainUtilities.verbose_enabled:
            print(f"enable cache: {LangChainUtilities.cache_type} (path: {LangChainUtilities.cache_path if LangChainUtilities.cache_type == 'sqlite' else 'inmemory'})")

    @staticmethod
    def disable_cache():
        """Disable cache."""
        LangChainUtilities.cache_enabled = False
        set_llm_cache(None)
        if LangChainUtilities.verbose_enabled:
            print("Cache disabled.")

    @staticmethod
    def enable_debug():
        """Enable debug mode."""
        LangChainUtilities.debug_enabled = True
        set_debug(True)
        if LangChainUtilities.verbose_enabled:
            print("Debugging enabled.")

    @staticmethod
    def disable_debug():
        """Disable debug mode."""
        LangChainUtilities.debug_enabled = False
        set_debug(False)
        if LangChainUtilities.verbose_enabled:
            print("Debugging disabled.")

    @staticmethod
    def enable_verbose():
        """Enable verbose mode."""
        LangChainUtilities.verbose_enabled = True
        print("Verbose mode enabled.")

    @staticmethod
    def disable_verbose():
        """Disable verbose mode."""
        LangChainUtilities.verbose_enabled = False
        print("Verbose mode disabled.")

    @staticmethod
    def set_embedding_model(model_name: str = "mistral"):
        """Set the embedding model based on input string.
        Default is 'mistral'. Supported models are 'mistral' and 'chatopenai'.
        """
        supported_models = ['mistral', 'chatopenai']
        
        LangChainUtilities.embedding_model["model"] = model_name
        if model_name.lower() == 'mistral':
            LangChainUtilities.embedding_model["embedding"] = OllamaEmbeddings(model="mistral")
        elif model_name.lower() == 'chatopenai':
            LangChainUtilities.embedding_model["embedding"] = OpenAIEmbeddings()
        else:
            if LangChainUtilities.verbose_enabled:
                print(f"{model_name} is not supported. Supported list is {supported_models}")
            LangChainUtilities.embedding_model["embedding"] = OllamaEmbeddings(model="mistral")  # Default to mistral

        if LangChainUtilities.verbose_enabled:
            print(f"Embedding model set to: {LangChainUtilities.embedding_model['model']}")
        
    @staticmethod
    def get_chat_mistral(streaming=False, temperature=0.1):
        """Get mistral model using ollama api. Optionally enable streaming output."""
        chat = ChatOllama(
            model="mistral",
            temperature=temperature,
            streaming=streaming,
            callbacks=[StreamingStdOutCallbackHandler()] if streaming else None,
        )
        return chat
    
    @staticmethod
    def get_chat_gpt_4o_mini(streaming=False, temperature=0.1):
        """Get GPT-4o-mini model. Optionally enable streaming output."""
        chat = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=temperature,
            streaming=streaming,
            callbacks=[StreamingStdOutCallbackHandler()] if streaming else None,
        )
        return chat

    @staticmethod
    def get_chat_gpt_3_5_turbo(streaming=False, temperature=0.1):
        """Get GPT-3.5-turbo model. Optionally enable streaming output."""
        chat = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=temperature,
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
    def track_ai_command(command, *args):
        """Track and print AI command usage with callbacks for token tracking."""
        if LangChainUtilities.verbose_enabled:
            print(f"Tracking AI command... {command}")
            start_time = time.time()  # Start tracking the time
        
        with get_openai_callback() as usage:
            out = command(*args)
        
        if LangChainUtilities.verbose_enabled:
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
        if LangChainUtilities.verbose_enabled:
            print(f"run. embed_file (model : {LangChainUtilities.embedding_model['model']})")
        file_path = f"./.cache/{LangChainUtilities.embedding_model['model']}/files/{file.name}"
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "wb") as f:
            f.write(file.read())
        cache_dir = LocalFileStore(f"./.cache/{LangChainUtilities.embedding_model['model']}/embeddings/{file.name}")        
        
        cached_embeddings = CacheBackedEmbeddings.from_bytes_store(LangChainUtilities.embedding_model["embedding"], cache_dir)
        
        splitter = CharacterTextSplitter.from_tiktoken_encoder(
            separator="\n",
            chunk_size=600,
            chunk_overlap=100,
        )        
        loader = UnstructuredFileLoader(file_path)
        docs = loader.load_and_split(text_splitter=splitter)
        
        vectorstore = FAISS.from_documents(docs, cached_embeddings)
        retriever = vectorstore.as_retriever()
        if LangChainUtilities.verbose_enabled:
            print("end. embed_file")
        return retriever
    
    @staticmethod
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
        if LangChainUtilities.verbose_enabled:
            print("chain_invoke_stuff")
        chain = (
            {
                "context": retriever | RunnableLambda(LangChainUtilities.format_docs),
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
        )
        return chain.invoke(message)

    @classmethod
    def create_chat(cls, model_name: str = "mistral", streaming: bool = False, temperature: float = 0.1):
        if model_name == "gpt_4o_mini":
            llm = cls.get_chat_gpt_4o_mini(streaming, temperature)
        elif model_name == "gpt_3_5_turbo":
            llm = cls.get_chat_gpt_3_5_turbo(streaming, temperature)
        elif model_name == "mistral":
            llm = cls.get_chat_mistral(streaming, temperature)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        return llm

    @staticmethod
    @st.cache_data(show_spinner="Loading file...")
    def split_file(file):
        file_content = file.read()
        file_path = f"./.cache/files/{file.name}"
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            f.write(file_content)
        splitter = CharacterTextSplitter.from_tiktoken_encoder(
            separator="\n",
            chunk_size=600,
            chunk_overlap=100,
        )
        loader = UnstructuredFileLoader(file_path)
        docs = loader.load_and_split(text_splitter=splitter)
        return docs

def test_get_prompt():
    """get_prompt 메서드를 테스트합니다."""
    prompt = LangChainUtilities.get_prompt("시스템 헤더", "질문 헤더")
    assert isinstance(prompt, ChatPromptTemplate), "반환된 객체가 ChatPromptTemplate 인스턴스가 아닙니다."
    print("get_prompt 테스트 통과")

def test_chain_invoke_stuff():
    """chain_invoke_stuff 메서드를 테스트합니다."""
    assert hasattr(LangChainUtilities, 'chain_invoke_stuff'), "chain_invoke_stuff 메서드가 존재하지 않습니다."
    print("chain_invoke_stuff 테스트 통과")

def test_create_chat():
    """create_chat 메서드를 테스트합니다."""
    models = ["gpt_4o_mini", "gpt_3_5_turbo", "mistral"]
    for model in models:
        llm = LangChainUtilities.create_chat(model)
        assert llm is not None, f"{model} 모델에 대한 llm 생성 실패"
    
    try:
        LangChainUtilities.create_chat("unsupported_model")
    except ValueError:
        print("지원되지 않는 모델에 대해 적절히 예외를 발생시킵니다.")
    else:
        # 이 줄은 지원되지 않는 모델에 대해 예외가 발생하지 않았을 때 테스트를 실패시킵니다.
        # 'assert False'는 무조건 AssertionError를 발생시켜 테스트를 실패하게 만듭니다.
        # 메시지는 예외가 발생하지 않은 이유를 설명합니다.
        assert False, "지원되지 않는 모델에 대해 예외가 발생하지 않았습니다."
    
    print("create_chat 테스트 통과")

def run_all_tests():
    """모든 테스트 메서드를 실행합니다."""
    test_get_prompt()
    test_chain_invoke_stuff()
    test_create_chat()
    print("모든 테스트가 성공적으로 완료되었습니다.")

if __name__ == "__main__":
    run_all_tests()
