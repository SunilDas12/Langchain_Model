import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI

# Load environment variables from .env file
load_dotenv()

http_proxy=os.environ['HTTP_PROXY'] 
https_proxy=os.environ['HTTPS_PROXY']  

def get_azure_openai():
    return AzureChatOpenAI(
        api_key=os.environ['OPENAI_API_KEY'],
        api_version=os.environ['OPENAI_API_VERSION'],
        azure_endpoint=os.environ['OPENAI_AZURE_ENDPOINT'],
        model_name=os.environ['OPENAI_AZURE_MODEL'],
        temperature=0.2,
        max_tokens=4000,
        seed=42,
    )
