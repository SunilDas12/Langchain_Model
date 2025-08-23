from langchain_community.document_loaders import TextLoader
from langchain_openai import AzureChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

model=AzureChatOpenAI(
        api_key=os.environ['OPENAI_API_KEY'],
        api_version=os.environ['OPENAI_API_VERSION'],
        azure_endpoint=os.environ['OPENAI_AZURE_ENDPOINT'],
        model_name=os.environ['OPENAI_AZURE_MODEL'],
        temperature=0.4,
        max_tokens=1000,
        seed=42,
)

prompt= PromptTemplate(
    template='Write a summary for the following text \n {text}',
    input_variables=['text']
)

parser= StrOutputParser()

loader=TextLoader('RAG\\OfflineData\\wipro.txt', encoding='utf-8')

docs=loader.load()
#print(len(docs))   ###List of items
#print(docs[0])
print(docs[0].page_content)
print(docs[0].metadata)

chain = prompt | model | parser

result=chain.invoke({'text': docs[0].page_content})
print(result)
