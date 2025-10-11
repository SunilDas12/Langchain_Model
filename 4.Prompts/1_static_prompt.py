from langchain_openai import AzureChatOpenAI
import os
from dotenv import load_dotenv
import streamlit as st

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

st.header('Research Tool')

user_input=st.text_input("Enter your prompt")

if st.button('Summerize'):
    result=model.invoke(user_input)
    st.write(result.content)

