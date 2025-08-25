from langchain.text_splitter import RecursiveCharacterTextSplitter

text="""
One of the most important things I didn't understand about the world when I was a child is the degree to which the returns for performance are superlinear.

Teachers and coaches implicitly told us the returns were linear. You get out, I heard a thousand times, what you put in. They meant well, but this is rarely true. If your product is only half as good as your competitor's, you don't get half as many customers. You get no customers, and you go out of business.
"""

# Initialize the splitter

splitter=RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=0
)

chunks=splitter.split_text(text)

print(len(chunks))
print(chunks)


### 1. First it will check paragraph wise chuncking '\n\n'
### 2. Second it will check line wise chunking '\n'
### 3. Third it will check word wise chunking ' '
### 4. Fourth it will check character wise chunking ''