from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
from langchain.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough

load_dotenv()

#passthrough=RunnablePassthrough()

### RunnablePassthrough - Return the same output as input 
#print(passthrough.invoke({'name':'sunil'}))


prompt1=PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)

model=AzureChatOpenAI(
        api_key=os.environ['OPENAI_API_KEY'],
        api_version=os.environ['OPENAI_API_VERSION'],
        azure_endpoint=os.environ['OPENAI_AZURE_ENDPOINT'],
        model_name='Gpt4mini',
        temperature=0.4,
        max_tokens=4000,
        seed=42,
)

parser=StrOutputParser()

prompt2=PromptTemplate(
    template='Explain the following joke {text}',
    input_variables=['text']
)

joke_gen_chain=RunnableSequence(prompt1, model, parser)

parallel_chain=RunnableParallel(
    {
        'joke': RunnablePassthrough(),
        'explanation': RunnableSequence(prompt2, model, parser)
    }
)

final_chain=RunnableSequence(joke_gen_chain, parallel_chain)

result=final_chain.invoke({'topic':'AI'})
print(result)

