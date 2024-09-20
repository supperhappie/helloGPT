from langchain.schema.runnable import RunnablePassthrough
import streamlit as st
from langchain.retrievers import WikipediaRetriever
from header import set_header
import sys, os, importlib
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(parent_dir)
import langchain_utilities
importlib.reload(langchain_utilities)
from langchain_utilities import LangChainUtilities

set_header()
st.title("QuizGPT")
print("#"*10 + "QuizGPT" + "#"*10)
if 'docs' not in st.session_state:
    st.session_state.docs = None
if 'llm' not in st.session_state:
    st.session_state.llm = LangChainUtilities.create_chat(model_name="mistral", temperature=0.1, streaming=True)

header_system = """ 
    [INST]
    You are a helpful assistant that is role playing as a teacher.
            
    Based ONLY on the following context make 10 questions to test the user's knowledge about the text.

    Each question should have 4 answers, three of them must be incorrect and one should be correct.
            
    Use (o) to signal the correct answer.
            
    Question examples:
            
    Question: What is the color of the ocean?
    Answers: Red|Yellow|Green|Blue(o)
            
    Question: What is the capital or Georgia?
    Answers: Baku|Tbilisi(o)|Manila|Beirut
            
    Question: When was Avatar released?
    Answers: 2007|2001|2009(o)|1998
            
    Question: Who was Julius Caesar?
    Answers: A Roman Emperor(o)|Painter|Actor|Model
            
    Your turn!
            
    Context: {context}

    Question : {question}
    [/INST]
    """

question = """
    give me 10 quiz with following above example format
    """
    
header_system_json = "Context : {context}\n\n give me a json code that print above context"

@st.cache_data(show_spinner="Searching Wikipedia...")
def get_wiki(topic):
    retriever = WikipediaRetriever(top_k_results=5)
    return retriever.get_relevant_documents(topic)

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
            st.session_state.docs = get_wiki(topic)

if st.session_state.docs:
    # st.write(f"Number of documents: {len(st.session_state.docs)}")
    # st.write("---")  # Add separator
    # for i, doc in enumerate(st.session_state.docs):
    #     st.write(f"Document {i+1}:")
    #     st.write(doc.page_content)
    #     if i < len(st.session_state.docs) - 1:
    #         st.write("---")  # Add separator
    prompt = LangChainUtilities.get_prompt(header_system=header_system)
    chain = LangChainUtilities.get_chain(prompt=prompt, llm=st.session_state.llm)
    
    start = st.button("Generate Quiz")
    if start:
        quiz = chain.invoke({"context": st.session_state.docs,"question":question})
        st.write(quiz.content)
        st.write("---")
        prompt_json = LangChainUtilities.get_prompt(header_system=header_system_json)
        chain_json = prompt_json | st.session_state.llm
        json = chain_json.invoke({"context": quiz.content, "question":"give me json code"})
        st.write(json.content)
        
        
else : 
    st.markdown(
        """
        Welcome to QuizGPT.
                    
        I will make a quiz from Wikipedia articles or files you upload to test your knowledge and help you study.
                    
        Get started by uploading a file or searching on Wikipedia in the sidebar.
        """
    )
