from langchain_openai import AzureChatOpenAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel
import os

load_dotenv()

# model 1 
model1 = AzureChatOpenAI(
        api_key=os.environ['OPENAI_API_KEY'],
        api_version=os.environ['OPENAI_API_VERSION'],
        azure_endpoint=os.environ['OPENAI_AZURE_ENDPOINT'],
        model_name='Gpt4mini',
        temperature=0.4,
        max_tokens=4000,
        seed=42,
)

# model 2
llm= HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)
model2=ChatHuggingFace(llm=llm)


# Define the prompts here
prompt1 = PromptTemplate(
    template='Generate short and simple notes from the following notes \n {text}',
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template='Generate 5 short question answers from the following text \n {text}',
    input_variables=['text']
)

prompt3 = PromptTemplate(
    template='Merge the provided notes and quiz into a single document. \n Notes: {notes} \n Quiz: {quiz}',
    input_variables=['notes', 'quiz']
)

# Define the output parser
parser = StrOutputParser()

# parallel chain 
parallel_chain=RunnableParallel({
    'notes':prompt1 | model1 | parser,
    'quiz':prompt2 | model2 | parser
})

# Merging chain
merged_chain = prompt3 | model1 | parser

# Combine the parallel chain with the merging chain
final_chain = parallel_chain | merged_chain

text= """ Deforestation is the large-scale removal of forested land, typically to make space for agriculture, urban development, or mining. It is one of the most pressing environmental issues of our time, affecting biodiversity, climate, and human well-being. Forests cover about 31% of the Earth’s land surface and are home to more than 80% of the terrestrial species of animals, plants, and fungi. When forests are cut down, not only is this rich biodiversity lost, but the balance of ecosystems is severely disrupted. In many tropical regions, deforestation is driven by the demand for commodities like soy, palm oil, and beef. These products require vast amounts of land, leading to the clearing of rainforests, particularly in the Amazon and Southeast Asia. Illegal logging also contributes significantly, with valuable hardwoods being extracted and sold on international markets. Additionally, infrastructure projects like roads and dams open up previously inaccessible forest areas to further exploitation.

Climate change and deforestation are closely linked. Trees absorb carbon dioxide (CO₂) from the atmosphere and store it. When forests are destroyed, not only is this carbon storage capacity lost, but the carbon stored in trees is also released back into the atmosphere, contributing to global warming. Deforestation is responsible for about 10% of global greenhouse gas emissions. It also affects the water cycle. Forests play a vital role in maintaining regional rainfall patterns, and their removal can lead to changes in precipitation, causing droughts in some areas and floods in others. The loss of tree cover leads to soil erosion, reducing the fertility of the land and making it more difficult to grow crops in the long term.

Human communities, especially indigenous populations, are often the first to suffer from the consequences of deforestation. Many depend directly on forests for food, medicine, and shelter. When forests are cleared, these communities lose their livelihoods and are sometimes forcibly displaced. Moreover, the cultural significance of forests to indigenous peoples is often overlooked in policymaking. Deforestation also increases the risk of zoonotic diseases — those that jump from animals to humans — by bringing people into closer contact with wildlife habitats. This has led scientists to warn that continued deforestation could lead to more pandemics in the future."""

# Invoke the final chain
result = final_chain.invoke({'text': text})
print(result)

# TO VISUALIZE THE CHAIN
final_chain.get_graph().print_ascii()