# About RAG Retrival Augmented generation. 
# RAG : if we search(retrival) some thing, that was collected from given user data (Augment) and request that with the collected bunch. 
# It can ge make the Ai model can check only the reference we choose. 
# Retrival process : Source >> Load(doc) > Split(Transform) >> Embed(doc to number) >> Store >> Retrieval (number search)
#   Langchain has many Loaders : Markdown, pdf, json, git, github, figma, power point, twitter ... they extract data from the formated docs 
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader 

loader = UnstructuredFileLoader("files/sample_1.pdf")

out = loader.load()
# 'out'이 Document 객체일 경우
for document in out:
    # Document 객체의 속성에 직접 접근
    print(document.page_content)

