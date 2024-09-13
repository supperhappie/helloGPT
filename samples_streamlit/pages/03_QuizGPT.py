from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
import streamlit as st
from langchain.retrievers import WikipediaRetriever
from header import set_header
import sys, os, importlib
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(parent_dir)
import LangChainUtilities
importlib.reload(LangChainUtilities)
from LangChainUtilities import LangChainUtilities
from Chat import Chat

set_header()
st.title("QuizGPT")

if 'docs' not in st.session_state:
    st.session_state.docs = None

with st.sidebar:
    choice = st.selectbox(
        "Choose what you want to use.",
        (
            "File",
            "Wikipedia Article",
        ),
    )
    if choice == "File":
        file = st.file_uploader(
            "Upload a .docx , .txt or .pdf file",
            type=["pdf", "txt", "docx"],
        )
        if file:
            st.session_state.docs = LangChainUtilities.split_file(file)
    else:
        topic = st.text_input("Search Wikipedia...")
        if topic:
            retriever = WikipediaRetriever(top_k_results=5)
            with st.status("Searching Wikipedia..."):
                st.session_state.docs = retriever.get_relevant_documents(topic)

if st.session_state.docs:
    st.write(f"Number of documents: {len(st.session_state.docs)}")
    st.write("---")  # Add separator
    for i, doc in enumerate(st.session_state.docs):
        st.write(f"Document {i+1}:")
        st.write(doc.page_content)
        if i < len(st.session_state.docs) - 1:
            st.write("---")  # Add separator