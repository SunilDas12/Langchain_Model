from langchain_community.document_loaders import CSVLoader

loader=CSVLoader('RAG\\OfflineData\\company_data.csv')

docs=loader.load()
print(len(docs))
print(docs[0])