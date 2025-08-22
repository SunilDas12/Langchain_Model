from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal
import os

load_dotenv()

model = AzureChatOpenAI(
        api_key=os.environ['OPENAI_API_KEY'],
        api_version=os.environ['OPENAI_API_VERSION'],
        azure_endpoint=os.environ['OPENAI_AZURE_ENDPOINT'],
        model_name='Gpt4mini',
        temperature=0.4,
        max_tokens=4000,
        seed=42,
)

parser=StrOutputParser()

# Output of LLM may vary, so we define a Pydantic model to parse the output as consistency is important.
class Feedback(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(description='The sentiment of the feedback text')

parser2 = PydanticOutputParser(pydantic_object=Feedback)

Prompt1 = PromptTemplate(
    template='Classify the sentiment of the following feedback text into positive or negative \n {feedback} \n {format_instructions}',
    input_variables=['feedback'],
    partial_variables={'format_instructions': parser2.get_format_instructions()}
)

classifier_chain= Prompt1 | model | parser2

#print(classifier_chain.invoke({'feedback': 'This is a terrible smartphone.'}).sentiment)

prompt2 = PromptTemplate(
    template='Write an appropriate resonse to this positive feedback \n {feedback}',
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template='Write an appropriate resonse to this negative feedback \n {feedback}',
    input_variables=['feedback']
)


brach_chain = RunnableBranch(
    (lambda x: x.sentiment == 'positive', prompt2 | model | parser),   #(condition, chain)
    (lambda x: x.sentiment == 'negative', prompt3 | model | parser),   #(condition, chain)
    RunnableLambda(lambda x: "could not find sentiment")  # Fallback if no condition matches   #default
)

final_chain=  classifier_chain | brach_chain

result = final_chain.invoke({
    'feedback': 'This is a terrible smartphone.'
})
print(result)

final_chain.get_graph().print_ascii()