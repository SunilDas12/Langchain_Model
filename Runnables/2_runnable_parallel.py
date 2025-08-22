from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
from langchain.schema.runnable import RunnableSequence,RunnableParallel

load_dotenv()

model=AzureChatOpenAI(
        api_key=os.environ['OPENAI_API_KEY'],
        api_version=os.environ['OPENAI_API_VERSION'],
        azure_endpoint=os.environ['OPENAI_AZURE_ENDPOINT'],
        model_name='Gpt4mini',
        temperature=0.4,
        max_tokens=4000,
        seed=42,
)

prompt1=PromptTemplate(
    template='Generate a tweet about {topic}',
    input_variables=['topic']
)

prompt2=PromptTemplate(
    template='Generate a Linkedin post about {topic}',
    input_variables=['topic']
)

parser=StrOutputParser()

parallel_chain=RunnableParallel(
    {
        'tweet': RunnableSequence(prompt1, model, parser),
        'linkedin': RunnableSequence(prompt2, model, parser)
    }
)

result=parallel_chain.invoke({'topic':'AI'})
print('Tweet:', result['tweet'])
print('LinkedIn:', result['linkedin'])
