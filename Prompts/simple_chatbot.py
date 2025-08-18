from langchain_openai import AzureChatOpenAI
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

while True:
    user_input=input('You:')
    if user_input == 'exit':
        break
    result=model.invoke(user_input)
    print('AI:', result.content)