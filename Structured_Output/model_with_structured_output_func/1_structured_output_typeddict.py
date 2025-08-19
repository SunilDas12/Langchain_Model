from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
from typing import TypedDict
import os

load_dotenv()
# model=ChatOpenAI()
model=AzureChatOpenAI(
        api_key=os.environ['OPENAI_API_KEY'],
        api_version=os.environ['OPENAI_API_VERSION'],
        azure_endpoint=os.environ['OPENAI_AZURE_ENDPOINT'],
        model_name=os.environ['OPENAI_AZURE_MODEL'],
        temperature=0.4,
        max_tokens=1000,
        seed=42,
)

class Review(TypedDict):
    summary: str
    sentiment: str

structured_model=model.with_structured_output(Review)

result=structured_model.invoke("""This hardware is greate, but the software feels bloated. There are too many pre-installed apps that I can't remove. Also, the UI looks outdated compared to other brand. Hoping for a software update to fix this""")

print(result)
print(result['summary'])
print(result['sentiment'])