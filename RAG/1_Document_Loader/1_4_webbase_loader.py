from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import AzureChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()


os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"  # Set User-Agent

headers = {
    "User-Agent": os.environ["USER_AGENT"],
    "Accept-Language": "en-US,en;q=0.9",
}

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
    template='Answer the following question \n {question} from the following text-\n {text}',
    input_variables=['question','text']
)

parser= StrOutputParser()

url='https://www.mckinsey.com/featured-insights/mckinsey-explainers/what-is-generative-ai'

loader=WebBaseLoader(url, requests_kwargs={"headers": headers, "timeout": 15})

docs=loader.load()

print(len(docs))

#print(docs[0].page_content)
print(docs[0].metadata)

chain=prompt | model | parser

result=chain.invoke({'question':'What is generative AI in 10 lines', 'text':docs[0].page_content})
print(result)
