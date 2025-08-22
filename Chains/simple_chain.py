from langchain_openai import AzureChatOpenAI
import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Define the prompt here
prompt = PromptTemplate(
    template ='Generate 5 interacting facts about {topic}.',
    input_variables=['topic']
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
chain = prompt | model | parser

result=chain.invoke({
    'topic': 'Python programming language'
})

print(result)

# TO VISUALIZE THE CHAIN
chain.get_graph().print_ascii()