from langchain_huggingface import HuggingFaceEmbeddings

embedding= HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

#text='Delhi is the capital of India'
#vector=embedding.embed_query(text)

documents=[
    "Delhi is the capital of India",
    "Kolkata ius the capital of West bengal",
    "Lion is the king of the forest"
]
vector=embedding.embed_documents(documents)
print(str(vector))