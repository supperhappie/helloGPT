import os, sys, asyncio, importlib, requests
import xml.etree.ElementTree as ET
from urllib.parse import urlparse
import streamlit as st
from header import set_header
from langchain.document_loaders import SitemapLoader
from langchain.document_transformers import Html2TextTransformer
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
# 상대 경로를 사용하여 부모 디렉토리 추가
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(parent_dir)

import langchain_utilities
importlib.reload(langchain_utilities)
from langchain_utilities import LangChainUtilities

# Function to check and append sitemap if necessary
# def get_sitemap_url(user_input_url):
#     parsed_url = urlparse(user_input_url)
#     if not parsed_url.scheme:
#         user_input_url = "https://" + user_input_url  # Default to HTTPS if no scheme provided
#         parsed_url = urlparse(user_input_url)

#     if not parsed_url.path.endswith("sitemap.xml"):
#         sitemap_url = f"{parsed_url.scheme}://{parsed_url.netloc}/sitemap.xml"
#     else:
#         sitemap_url = user_input_url
    
#     return sitemap_url

# # Manually parse sitemap to extract URLs (if SitemapLoader is struggling)
# def parse_sitemap(sitemap_content):
#     print(f"[DEBUG] parse_sitemap : {sitemap_content}")
#     urls = []
#     try:
#         root = ET.fromstring(sitemap_content)
#         for url in root.findall("{http://www.sitemaps.org/schemas/sitemap/0.9}url"):
#             loc = url.find("{http://www.sitemaps.org/schemas/sitemap/0.9}loc").text
#             urls.append(loc)
#         return urls
#     except Exception as e:
#         print(f"Error parsing sitemap: {e}")
#         return []

asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

html2text_transformer = Html2TextTransformer()

# Call the global configuration
print("[DEBUG] start page")
set_header()

# # make a side bar 
# st.sidebar.title("Site Scraping")

# # input url >> script the url using playwright and chromium 
# with st.sidebar:
#     st.write("sample url : " + "https://warcraft.wiki.gg/wiki/Warcraft_Wiki")
#     url = st.text_input(
#         "Write down a URL",
#         placeholder="https://example.com", # good example : https://warcraft.wiki.gg/wiki/Warcraft_Wiki         
#     )
#     print(f"[DEBUG] url : {url}")

# if url:
#     print(f"[DEBUG] url : {url}")
#     sitemap_url = get_sitemap_url(url)
    
#     response = requests.get(sitemap_url)
#     print(f"[DEBUG] Sitemap Content: {response.text[:500]}")  # Print the first 500 characters for debugging

#     if response.status_code == 200:
#         try:
#             # Option 1: Manually parse the sitemap for URLs
#             urls = parse_sitemap(response.text)
#             if urls:
#                 st.write(f"Found {len(urls)} URLs in the sitemap.")
#                 for url in urls[:500]:  # Limit to first 10 URLs for display
#                     st.write(url)
#             else:
#                 st.error("No URLs found in the sitemap.")

#             # Option 2: If manual parsing works, use it for SitemapLoader
#             # loader = SitemapLoader(urls)  # Pass the list of URLs
#             # loader.requests_per_second = 1  # Adjust as needed
#             # contentdocs = loader.load()
#             # # transformed = html2text_transformer.transform_documents(contentdocs)
#             # st.write(contentdocs)
#             print("complete output sitemap")
#         except Exception as e:
#             st.error(f"Error loading sitemap: {e}")
#     else:
#         st.error(f"Failed to fetch the sitemap. HTTP Status: {response.status_code}")

# from langchain.document_loaders import SitemapLoader
# import streamlit as st


def parse_page(soup):
    header = soup.find("header")
    footer = soup.find("footer")
    if header:
        header.decompose()
    if footer:
        footer.decompose()
    return (
        str(soup.get_text())
        # .replace("\n", " ")
        # .replace("\xa0", " ")
        # .replace("CloseSearch Submit Blog", "")
    )


@st.cache_data(show_spinner="Loading website...")
def load_website(url):    
    # loader = SitemapLoader(url, filter_urls=["https://nomadcoders.co/tiktok-clone"])
    loader = SitemapLoader(
        url, 
        # filter_urls=[r"^(.*\/community\/).*"],
        # 이 줄은 URL이 "https://nomadcoders.co/community/thread/1020"에서 "https://nomadcoders.co/community/thread/1039" 사이에 있는 페이지들만 필터링하도록 정규 표현식을 사용합니다.
        # filter_urls=[r"https://nomadcoders.co/community/thread/10[2-3][0-9]{2}"],
        filter_urls=[r"https://nomadcoders.co/tiktok-clone"],
        
        parsing_function=parse_page,
        )
    loader.requests_per_second = 1
    
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200,
    )
    docs = loader.load_and_split(text_splitter=splitter)
    if not docs:
        raise ValueError("No documents found. Please check the URL and try again.")
    vector_store = FAISS.from_documents(docs, OpenAIEmbeddings())
    return vector_store.as_retriever()


llm = LangChainUtilities.get_chat_mistral()
answers_prompt = ChatPromptTemplate.from_template(
    """
    Using ONLY the following context answer the user's question. If you can't just say you don't know, don't make anything up.
                                                  
    Then, give a score to the answer between 0 and 5.

    If the answer answers the user question the score should be high, else it should be low.

    Make sure to always include the answer's score even if it's 0.

    Context: {context}
                                                  
    Examples:
                                                  
    Question: How far away is the moon?
    Answer: The moon is 384,400 km away.
    Score: 5
                                                  
    Question: How far away is the sun?
    Answer: I don't know
    Score: 0
                                                  
    Your turn!

    Question: {question}
"""
)

def get_answers(inputs):
    docs = inputs["docs"]
    question = inputs["question"]
    answers_chain = answers_prompt | llm
    # answers = []
    # for doc in docs:
    #     result = answers_chain.invoke(
    #         {"question": question, "context": doc.page_content}
    #     )
    #     answers.append(result.content)
    return {
        "question": question,
        "answers": [
            {
                "answer": answers_chain.invoke(
                    {"question": question, "context": doc.page_content}
                ).content,
                "source": doc.metadata["source"],
                "date": doc.metadata["lastmod"],
            }
            for doc in docs
        ],
    }

choose_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Use ONLY the following pre-existing answers to answer the user's question.

            Use the answers that have the highest score (more helpful) and favor the most recent ones.

            Cite sources and return the sources of the answers as they are, do not change them.

            Answers: {answers}
            """,
        ),
        ("human", "{question}"),
    ]
)

def choose_answer(inputs):
    answers = inputs["answers"]
    question = inputs["question"]
    choose_chain = choose_prompt | llm
    condensed = "\n\n".join(
        f"{answer['answer']}\nSource:{answer['source']}\nDate:{answer['date']}\n"
        for answer in answers
    )
    return choose_chain.invoke(
        {
            "question": question,
            "answers": condensed,
        }
    )



st.markdown(
    """
    # SiteGPT
            
    Ask questions about the content of a website.
            
    Start by writing the URL of the website on the sidebar.
"""
)


with st.sidebar:
    url = st.text_input(
        "Write down a URL",
        placeholder="https://example.com",
    )


if url:
    if ".xml" not in url:
        with st.sidebar:
            st.error("Please write down a Sitemap URL.")
    else:
        retriever = load_website(url)
        query = st.text_input("Ask a question to the website.")
        if query:
            chain = (
                {
                    "docs": retriever,
                    "question": RunnablePassthrough(),
                }
                | RunnableLambda(get_answers)
                | RunnableLambda(choose_answer)
            )
            result = chain.invoke(query)
            st.markdown(result.content.replace("$", "\$"))