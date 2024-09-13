from typing import Dict, List
from uuid import UUID
from langchain.schema.output import ChatGenerationChunk, GenerationChunk
from langchain.chat_models import ChatOpenAI, ChatOllama
import streamlit as st
from header import set_header
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks.base import BaseCallbackHandler

import sys, os, importlib
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(parent_dir)
import utils
importlib.reload(utils)
from utils import Utils

# our local pc/server runs embedding and llm 
# How!

class ChatCallbackHandler(BaseCallbackHandler):
    
    def on_llm_start(self, *args, **kwargs):
        print("on_llm_start")
        self.message = ""
        self.message_box = st.empty()
        
    def on_llm_end(self, *args, **kwargs):
        print("on_llm_end")
        pass

    def on_llm_new_token(self, token, *args, **kwargs):
        print("on_llm_new_token")
        self.message += token
        self.message_box.markdown(self.message) 
        
        
def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})
def send_message(message, role, save=True):
    print("send_message")
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def paint_history():
    print("paint history")
    for message in st.session_state["messages"]:
        send_message(
            message["message"],
            message["role"],
            save=False,
        )
        
@st.cache_data(show_spinner=False)
def cached_chain_invoke(file, _retriever, _llm, message, _prompt):
    return Utils.chain_invoke_stuff(_retriever, _llm, message, _prompt) 
    

# Call the global configuration
set_header()
st.title("PrivateGPT")
Utils.enable_cache()
Utils.set_embedding_model('chatopenai')

# llm = Utils.get_chat_gpt_4o_mini()
# llm = Utils.get_chat_gpt_3_5_turbo()
llm = ChatOllama(
    temperature=0.1,
    model="mistral",
    streaming=True,
    callbacks=[
        ChatCallbackHandler(),
    ]
)
prompt = Utils.get_prompt(header_system="Answer the question using ONLY the following context. If you don't know the answer just say you don't know. DON'T make anything up.")

st.markdown(
    """
    Welcome!
                
    Use this chatbot to ask questions to an AI about your files!
    
    Upload your files on the sidebar.
    """
)
with st.sidebar:
    file = st.file_uploader(
        "Upload a .txt .pdf or .docx file",
        type=["pdf", "txt", "docx"],
    )

if file:
    retriever = Utils.embed_file(file)
    send_message("I'm ready! Ask away!", "ai", save=False)
    paint_history()
    message = st.chat_input("Ask anything about your file...")
    if message:
        send_message(message, "human")        
        with st.chat_message('ai'):
            response = cached_chain_invoke(file, retriever, llm, message, prompt) 
        save_message(response.content, "ai")
else:    
    st.session_state["messages"] = []
