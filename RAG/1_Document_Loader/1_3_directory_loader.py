from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader=DirectoryLoader(
    path='RAG\\OfflineData',
    glob='*.pdf',
    loader_cls=PyPDFLoader
)

## Load function

docs=loader.load()
print(len(docs))
print(docs[0].page_content)
print(docs[0].metadata)

### Lazy_load function (when you have large number of documents instead of loading all in memory use lazy_load for faster execution) ###

#docs=loader.lazy_load()
for document in docs:
    print(document.metadata)
