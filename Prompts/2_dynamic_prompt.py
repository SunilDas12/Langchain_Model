from langchain_openai import AzureChatOpenAI
import streamlit as st
import os
from langchain_core.prompts import PromptTemplate, load_prompt
from dotenv import load_dotenv

st.header('Research Tool')

load_dotenv()
#model=ChatOpenAI()

model=AzureChatOpenAI(
        api_key=os.environ['OPENAI_API_KEY'],
        api_version=os.environ['OPENAI_API_VERSION'],
        azure_endpoint=os.environ['OPENAI_AZURE_ENDPOINT'],
        model_name=os.environ['OPENAI_AZURE_MODEL'],
        temperature=0.4,
        max_tokens=1000,
        seed=42,
)
# Dropdowns
paper_input = st.selectbox("Select Paper Type", ["Research", "Whitepaper", "Article", "Technical Report"])
style_input = st.selectbox("Select Writing Style", ["Formal", "Informal", "Narrative", "Explanatory"])
length_input = st.selectbox("Select Length", ["Short", "Medium", "Long"])

# Load template  [prompt_generator---->template.json---->load]
template= load_prompt('template.json')

prompt=template.invoke({
    'paper_input':paper_input,
    'style_input':style_input,
    'length_input':length_input

})


# Button click handler
if st.button("Response"):
    result=model.invoke(prompt)
    st.write(result.content)
 
