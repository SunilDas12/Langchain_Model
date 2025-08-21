from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm= HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)
model=ChatHuggingFace(llm=llm)

parser=JsonOutputParser()

templet=PromptTemplate(
    template='Give me the age, name and city of a fictional person  \n {format_instructions}',
    input_variables=[],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

prompt=templet.format()
print(f"Prompt: {prompt}")

result=model.invoke(prompt)
#print(f"Result: {result}")

final_result=parser.parse(result.content)

print(f"Final Result: {final_result}")
print(final_result['name'])

