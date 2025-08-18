from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()

embedding = OpenAIEmbeddings(
    model=os.getenv("OPENAI_AZURE_MODEL"),   # model name e.g., "text-embedding-3-large"
    deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),   
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    openai_api_version=os.getenv("OPENAI_API_VERSION"),
    openai_api_base=os.getenv("OPENAI_AZURE_ENDPOINT"),  
)

documents=[
    "Delhi is the capital of India",
    "Kolkata ius the capital of West bengal",
]
# Example query
result = embedding.embed_documents(documents)
print(result)
