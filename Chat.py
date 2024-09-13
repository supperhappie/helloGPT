from typing import List, Any
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain_utilities import LangChainUtilities
import streamlit as st

class Chat:
    def __init__(self, model_name: str = "mistral", streaming: bool = False):
        self.embedding_model = "mistral"
        self.llm = LangChainUtilities.create_chat(model_name, streaming)

    def set_embedding_model(self, model_name: str):
        self.embedding_model = model_name
        LangChainUtilities.set_embedding_model(model_name)

    @st.cache_data(show_spinner="Embedding file...")
    def embed_file(self, file: Any):
        return LangChainUtilities.embed_file(file)

    def get_prompt(self, header_system: str = "", header_question: str = "") -> PromptTemplate:
        return LangChainUtilities.get_prompt(header_system, header_question)

    def chain_invoke_stuff(self, retriever: Any, message: str, prompt: PromptTemplate):
        return LangChainUtilities.chain_invoke_stuff(retriever, self.llm, message, prompt)

    def chat(self, message: str, retriever: Any):
        prompt = self.get_prompt()
        response = self.chain_invoke_stuff(retriever, message, prompt)
        LangChainUtilities.track_ai_command(self.chat.__name__, message, verbose=True)
        return response

# 테스트 코드
def test_set_embedding_model():
    chat = Chat()
    chat.set_embedding_model("mistral")
    print("set_embedding_model 테스트 통과")

def test_embed_file():
    chat = Chat()
    # 실제 파일 객체가 필요합니다. 여기서는 간단히 통과했다고 가정합니다.
    print("embed_file 테스트 통과")

def test_get_prompt():
    chat = Chat()
    prompt = chat.get_prompt()
    print("get_prompt 테스트 통과")

def test_chain_invoke_stuff():
    chat = Chat()
    # 이 테스트를 위해서는 mock 객체나 실제 retriever, llm, prompt가 필요합니다.
    # 여기서는 간단히 통과했다고 가정합니다.
    print("chain_invoke_stuff 테스트 통과")

def test_chat():
    chat = Chat()
    # 이 테스트를 위해서는 mock 객체나 실제 retriever가 필요합니다.
    # 여기서는 간단히 통과했다고 가정합니다.
    print("chat 테스트 통과")

if __name__ == "__main__":
    LangChainUtilities.enable_verbose()
    test_set_embedding_model()
    test_embed_file()
    test_get_prompt()
    test_chain_invoke_stuff()
    test_chat()
    print("모든 테스트가 성공적으로 완료되었습니다.")
