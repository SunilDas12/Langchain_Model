
from azure_openai import get_azure_openai
from dotenv import load_dotenv

load_dotenv()

model=get_azure_openai()

result=model.invoke("write a 5 line poem on cricket")
print(result.content)