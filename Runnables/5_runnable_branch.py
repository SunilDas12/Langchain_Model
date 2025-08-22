from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
from langchain.schema.runnable import RunnableSequence, RunnablePassthrough, RunnableParallel, RunnableLambda, RunnableBranch

load_dotenv()

prompt1=PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)
prompt2=PromptTemplate(
    template='Summerize the following \n {text}',
    input_variables=['text']
)

model=AzureChatOpenAI(
        api_key=os.environ['OPENAI_API_KEY'],
        api_version=os.environ['OPENAI_API_VERSION'],
        azure_endpoint=os.environ['OPENAI_AZURE_ENDPOINT'],
        model_name='Gpt4mini',
        temperature=0.4,
        max_tokens=400,
        seed=42,
)

parser=StrOutputParser()

report_gen_chain=RunnableSequence(prompt1, model, parser)

branch_chain=RunnableBranch(
    (lambda x: len(x.split())>100, RunnableSequence(prompt2, model, parser)),
    RunnablePassthrough()
)

final_chain=RunnableSequence(report_gen_chain, branch_chain)

result=final_chain.invoke({'topic':'Russia vs Ukrain'})

print(result)