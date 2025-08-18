from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

embedding= HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')


facts = [
    "Delhi is the capital of India.",
    "Mount Everest is the highest mountain in the world.",
    "The Pacific Ocean is the largest ocean on Earth.",
    "The Sahara is the largest hot desert in the world.",
    "The Amazon River is the second longest river in the world.",
    "The Great Wall of China is the longest wall in the world.",
    "The Taj Mahal is located in Agra, India.",
    "Water freezes at 0 degrees Celsius.",
    "The Earth orbits the Sun once every 365 days.",
    "The human heart has four chambers."
]

query='which river is the second longest river in the world'

doc_embedding=embedding.embed_documents(facts)
query_embedding=embedding.embed_query(query)

scores=cosine_similarity([query_embedding],doc_embedding)[0]

# To calculate the highest matching score vector
enum_list=list(enumerate(scores))
print(enum_list)
index, score = sorted((enum_list), key=lambda x:x[1])[-1]

print(query)
print(facts[index])
print('Similarity score is', score)
 