from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder

# chart templet
chat_template= ChatPromptTemplate([
        ("system", "You are a helpful helpful customer support agent."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{query}"),     
    ])

chat_history=[]
# load chart history
with open("Prompts//chat_history_placeholder.txt") as file:
    chat_history.extend(file.readlines())

prompt=chat_template.invoke({"chat_history": chat_history, "query": "Where is my refund"})
print(prompt)