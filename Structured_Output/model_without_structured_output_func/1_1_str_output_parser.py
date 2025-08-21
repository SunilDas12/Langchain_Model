from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()


llm= HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)
model=ChatHuggingFace(llm=llm)

templet1=PromptTemplate(
    template='write a detailed report on {topic}',
    input_variables=['topic']
)
templet2=PromptTemplate(
    template='write a 5 lines summary on the following text. \n {text}',
    input_variables=['text']
)

# --- step 1: format first prompt ---
prompt1 = templet1.format(topic="The impact of climate change on global agriculture")
result = model.invoke(prompt1)

# --- step 2: format second prompt using result of step 1 ---
prompt2 = templet2.format(text=result.content)
result1 = model.invoke(prompt2)

print(result1.content)
