from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# Step 1: Your source documents
documents=[
    Document(page_content="Langchain helps developers to build LLM applications easily."),
    Document(page_content="Chroma is a vector database optimized for LLM-based search."),
    Document(page_content="Embeddings Convert text into high-dimentional vector."),
    Document(page_content="OpenAI provides powerfull embedding models."),
]

# Step 2: Initializing embedding model
embedding_model= HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# Step3: Create Chroma vector store in memory
vectorstore=Chroma.from_documents(
    documents=documents,
    embedding=embedding_model,
    collection_name="my_collection"
)

# Step 4: Convert vectorstore into a retriever
retriever= vectorstore.as_retriever(
    search_kwargs={"k":2}
    )

query="What is Chroma used for?"
results=retriever.invoke(query)

# Print retrived content
for i, doc in enumerate(results):
    print(f"\n----Result {i+1}---")
    print(doc.page_content)

