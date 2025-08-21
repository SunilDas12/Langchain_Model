from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate

load_dotenv()

llm= HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)
model=ChatHuggingFace(llm=llm)

# pydantic object
class Person(BaseModel):
    name: str = Field(description="Name of the person")
    age: int = Field(gt=18, description="Age of the person")
    email: Optional[str] = Field(default=None, description="Email address of the person")
    is_active: bool = Field(default=True, description="Whether the person is currently active")

parser=PydanticOutputParser(pydantic_object=Person)

template= PromptTemplate(
    template="Extract the person's details from the following text: {input}\n\n{format_instructions}",
    input_variables=["input"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

chain= template | model | parser

final_result = chain.invoke(
    {'input': "Alice Smith is a 25-year-old data scientist with the email john@gmail.com'. She is currently active in her role."}
)
print(final_result)
 
