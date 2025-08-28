from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline, HuggingFaceEndpoint
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from dotenv import load_dotenv

load_dotenv()


# Documents objects
docs=[
    Document(page_content="""The Grand Canyon is one of the most visited natural wonders in the world. Photosynthesis is the process by which green plants convert sunlight into energy. Millions of tourists travel to see it every year. The rocks date back millions of years.""", metadata={"source":"Doc1"}),

    Document(page_content=""""In medieval Europe, castles were built primarily for defense. The chlorophyll in plant cells captures sunlight during photosynthesis. Knights wore armor made of metal. Siege weapons were often used to breach castle walls.""", metadata={"source":"Doc2"}),

    Document(page_content="""Basketball was invented by Dr. James Naismith in the late 19th century. It was originally played with a soccer ball and peach baskets. NBA is now a global league.""", metadata={"source":"Doc3"}),

    Document(page_content="""The history of cinema began in the late 1800s. Silent films were the earliest form. Thomas Edison was among the pioneers. Photosynthesis does not occur in animal cells.""", metadata={"source":"Doc4"}),
]

# Initializing embedding model
embedding_model= HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# Step3: Create the FAISS vector store from documents
vectorstore=FAISS.from_documents(
    documents=docs,
    embedding=embedding_model
)

base_retriever=vectorstore.as_retriever(search_kwargs={"k":5})

# Set up the compressor using LLM
llm= HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)
model=ChatHuggingFace(llm=llm)
compressor=LLMChainExtractor.from_llm(model)


# Create the contextual compression retriever
compression_retriever=ContextualCompressionRetriever(
    base_retriever=base_retriever,
    base_compressor=compressor
)

# Query the retriever
query="What is photosynthesis?"
compressed_result=compression_retriever.invoke(query)

# Print retrived content
for i, doc in enumerate(compressed_result):
    print(f"\n----Result {i+1}---")
    print(doc.page_content)