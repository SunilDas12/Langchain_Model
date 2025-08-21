from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
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

parser=StrOutputParser()

chain=templet1 | model | parser | templet2 | model | parser

result=chain.invoke({'topic': 'The impact of climate change on global agriculture'})
print(f"Report: {result}") 