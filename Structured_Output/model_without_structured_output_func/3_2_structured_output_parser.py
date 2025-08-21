from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
 
load_dotenv()

llm= HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)
model=ChatHuggingFace(llm=llm)

scehama = [
    ResponseSchema(name="fact_1",description='fact_1 about the topic'),
    ResponseSchema(name="fact_2",description='fact_2 about the topic'),
    ResponseSchema(name="fact_3",description='fact_3 about the topic')
]

parser= StructuredOutputParser.from_response_schemas(scehama)

templet=PromptTemplate(
    template='write 3 facts about {topic} \n {format_instructions}',
    input_variables=['topic'],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)


chain = templet | model | parser
result=chain.invoke({'topic': 'The impact of climate change on global agriculture'})
print(f"Final Result: {result}")