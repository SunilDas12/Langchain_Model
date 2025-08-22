from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
from langchain.schema.runnable import RunnableSequence, RunnablePassthrough, RunnableParallel, RunnableLambda

load_dotenv()

def word_count(text):
    return len(text.split())

prompt=PromptTemplate(
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

joke_gen_chain=RunnableSequence(prompt, model, parser)

parallel_chain=RunnableParallel(
    {
        'joke': RunnablePassthrough(),
        'word_count': RunnableLambda(word_count)
        #or 'word_count': RunnableLambda(lambda x: len(x.split()))
    }
)

final_chain=RunnableSequence(joke_gen_chain, parallel_chain)

result=final_chain.invoke({'topic': 'AI'})
final_result=""""{} \n word count - {}""". format(result['joke'], result['word_count'])
print(final_result)