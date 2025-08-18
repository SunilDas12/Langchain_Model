from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from azure_openai import get_azure_openai
from openai_response import get_chat_response
from dotenv import load_dotenv

load_dotenv()
model = get_azure_openai()  # Initialize the Azure OpenAI client: OPTIONAL FOR UNDERSTANDING
chat_history = [
    SystemMessage(content="You are a helpful AI assistant."),
]

while True:
    user_input = input("You: ")
    chat_history.append(HumanMessage(content=user_input))
    if user_input.lower() == "exit":
        print("Exiting the chat. Goodbye!")
        break
    response =  model.invoke(chat_history)#get_chat_response(chat_history)
    chat_history.append(AIMessage(content=response.content))
    print(f"AI: {response.content}")

print(f"Chat history: {chat_history}")