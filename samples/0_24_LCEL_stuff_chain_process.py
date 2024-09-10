from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.vectorstores import FAISS
from langchain.storage import LocalFileStore
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from utils import Utils

llm = ChatOpenAI(temperature=0.1, model="gpt-3.5-turbo")

cache_dir = LocalFileStore("./.cache/")
embeddings = OpenAIEmbeddings()
cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)

splitter = CharacterTextSplitter.from_tiktoken_encoder(
    separator="\n",
    chunk_size=600,
    chunk_overlap=100,
)
loader = UnstructuredFileLoader("./files/sample_1.pdf")
docs = loader.load_and_split(text_splitter=splitter)

vectorstore = FAISS.from_documents(docs, cached_embeddings)
    
retriver = vectorstore.as_retriever()

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer questions using only the following context. If you don't know the answer just say you don't know, don't make it up:\n\n{context}",
        ),
        ("human", "{question}"),
    ]
)

chain = (
    {
        "context": retriver,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
)

out = Utils.track_ai_command(chain.invoke("Describe Tim Berners-Lee"), verbose=True)
print(out)