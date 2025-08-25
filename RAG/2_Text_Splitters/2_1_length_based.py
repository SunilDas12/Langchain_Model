from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader=PyPDFLoader('RAG\\OfflineData\\SQL.pdf')
docs=loader.load()

splitter=CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
    separator=''
)

reesult=splitter.split_documents(docs)
print(reesult[0].page_content)
print("--------------------")
print(reesult[1].page_content)