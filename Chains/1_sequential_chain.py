from langchain_openai import AzureChatOpenAI
import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

 # Define the prompts here
prompt1=PromptTemplate(
    template='Genarate a detailed report on {topic}',
    input_variables=['topic']
)

prompt2=PromptTemplate(
    template='Generate a 5 pointer summary from the following text \n {text}',
    input_variables=['text']
)

# Define LLM model
model=AzureChatOpenAI(
        api_key=os.environ['OPENAI_API_KEY'],
        api_version=os.environ['OPENAI_API_VERSION'],
        azure_endpoint=os.environ['OPENAI_AZURE_ENDPOINT'],
        model_name=os.environ['OPENAI_AZURE_MODEL'],
        temperature=0.4,
        max_tokens=1000,
        seed=42,
)

# Define the output parser
parser=StrOutputParser()

# chain the components together
chain= prompt1 | model | parser | prompt2 | model | parser

result=chain.invoke({
    'topic': 'Unemployement in India',
})
print(result)

# TO VISUALIZE THE CHAIN
chain.get_graph().print_ascii()