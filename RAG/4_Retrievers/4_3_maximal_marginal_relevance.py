from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# Step 1: Your source documents
documents=[
    Document(page_content="Langchain makes it easy to work with LLM."),
    Document(page_content="Langchain is used to build LLM based application."),
    Document(page_content="Chroma is used to store and search document embeddings."),
    Document(page_content="Embeddings are vector representations of text."),
    Document(page_content="MMR helps you get diverse result when doing similarity search."),
    Document(page_content="Langchain supports Chroma, FAISS, Pinecode and more"),
]

# Initializing embedding model
embedding_model= HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# Step3: Create the FAISS vector store from documents
vectorstore=FAISS.from_documents(
    documents=documents,
    embedding=embedding_model
)

# Enable MMR in retriever
retriever=vectorstore.as_retriever(
    search_type="mmr",              #<---- This enable MMR                 ## Opt for test: similarity
    search_kwargs={"k":3, "lambda_mult":0}  # k=top results, lambda_mult= relevant-diversity balance[0--1]
)

query="What is langchain"
results=retriever.invoke(query)

# Print retrived content
for i, doc in enumerate(results):
    print(f"\n----Result {i+1}---")
    print(doc.page_content)


