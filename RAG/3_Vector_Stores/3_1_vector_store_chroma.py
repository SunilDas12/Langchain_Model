from langchain_huggingface import HuggingFaceEmbeddings
#from langchain_community.vectorstores import Chroma    ## Deprecated 0.2.9 and will remove in 1.0
from langchain_chroma import Chroma
from langchain.schema import Document

embedding= HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')


# Create langchain documents for IPL players

doc1=Document(
    page_content="Virat Kohli is one of the greatest batsmen in modern cricket, known for his aggressive playing style and remarkable consistency across all formats. He has led India to many memorable victories and holds numerous batting records worldwide.",
    metadata={"team": "Royal Challengers Bangalore"}
)
doc2 = Document(
    page_content="MS Dhoni is celebrated as one of the most successful captains in cricket history, admired for his calm demeanor and sharp decision-making skills. He led India to historic victories, including the 2007 T20 World Cup and the 2011 ICC Cricket World Cup.",
    metadata={"team": "Chennai Super Kings"}
)

doc3 = Document(
    page_content="Rohit Sharma is renowned for his elegant batting style and record-breaking performances as an opener in international cricket. Known as the 'Hitman,' he has the unique feat of scoring multiple double centuries in ODIs.",
    metadata={"team": "Mumbai Indians"}
)

doc4 = Document(
    page_content="Ravindra Jadeja is a dynamic all-rounder, known for his sharp fielding, accurate left-arm spin, and crucial batting performances. He has played a key role in many of India’s victories with his versatility and consistency.",
    metadata={"team": "Chennai Super Kings"}
)

doc5 = Document(
    page_content="Jasprit Bumrah is one of the world’s premier fast bowlers, recognized for his deadly yorkers and ability to perform under pressure. He has been instrumental in India’s success across formats with his match-winning spells.",
    metadata={"team": "Mumbai Indians"}
)


docs=[doc1, doc2, doc3, doc4, doc5]

vector_store=Chroma(
    embedding_function=embedding,
    persist_directory='RAG\\chroma_db',
    collection_name='sample'
)

## add documents ##
#store=vector_store.add_documents(docs)

## view documents ##
get_store=vector_store.get(include=['embeddings','documents','metadatas'])
print(get_store, "\n\n")

## search documents ##
doc_search=vector_store.similarity_search(
    query='who among these are a bowler ?',
    k=2
)
print(doc_search, "\n\n")

## search with similarity score ##
doc_search_score=vector_store.similarity_search_with_score(
    query='who among these are a bowler ?',
    k=2
)
print(doc_search_score, "\n\n")

## meta-data filtering ##
meta_data_filter=vector_store.similarity_search_with_score(
    query="",
    filter={"team":"Chennai Super Kings"}
)
print(meta_data_filter, "\n\n")

## update documents ##
updated_doc1=Document(
    page_content="Virat Kohli, the former captain of Royal Challengers Bangalore(RCB), is renowed for his aggressive leadership and consistency",
    metadata={"team": "Royal Challengers Bangalore"}
)
vector_store.update_document(document_id='b6884e80-817e-4eee-930d-1a213ab97d0c', document=updated_doc1)
get_store_updated=vector_store.get(include=['embeddings','documents','metadatas'])
print("After Doc1 updated \n")
print(get_store_updated, "\n\n")


## delete document ##
vector_store.delete(ids=['6dc229ad-28fe-4140-9e07-c32dbdbf49db'])
get_store_deleted=vector_store.get(include=['embeddings','documents','metadatas'])
print("After Doc3 deleted \n")
print(get_store_deleted, "\n\n")







