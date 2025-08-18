from dotenv import load_dotenv
from openai import AzureOpenAI
import os

# Load environment variables from .env file
load_dotenv()

# Initialize Azure OpenAI client
model = AzureOpenAI(
    api_key=os.environ['OPENAI_API_KEY'],
    api_version=os.environ['OPENAI_API_VERSION'],
    azure_endpoint=os.environ['OPENAI_AZURE_ENDPOINT'],
)

def get_chat_response(messages: list) -> str:
    # Convert messages to the required format for OpenAI API
    formatted_messages = []
    for idx, msg in enumerate(messages):
        if isinstance(msg, dict) and 'role' in msg and 'content' in msg:
            formatted_messages.append(msg)
        elif hasattr(msg, 'role') and hasattr(msg, 'content'):
            formatted_messages.append({"role": msg.role, "content": msg.content})
        elif msg.__class__.__name__ == "SystemMessage" and hasattr(msg, 'content'):
            formatted_messages.append({"role": "system", "content": msg.content})
        elif msg.__class__.__name__ == "HumanMessage" and hasattr(msg, 'content'):
            formatted_messages.append({"role": "user", "content": msg.content})
        elif msg.__class__.__name__ == "AIMessage" and hasattr(msg, 'content'):
            formatted_messages.append({"role": "assistant", "content": msg.content})
        elif hasattr(msg, 'role') and hasattr(msg, 'text'):
            formatted_messages.append({"role": msg.role, "content": msg.text})
        else:
            print(f"Invalid message at index {idx}: {msg} (type: {type(msg)})")
            raise ValueError("Each message must be a dictionary with 'role' and 'content' keys, or a supported LangChain message object.")
    
    # Call the Azure OpenAI API with the formatted messages
    response = model.chat.completions.create(
        model=" ",  # Replace with your actual deployment name 
        messages=formatted_messages,
        max_tokens=4000,
        temperature=0.4,
        seed=42,
        n=1,
    )
    return response.choices[0].message.content

# For prompt_ai.py, you would use the function like this:

# def get_chat_response(user_input: str) -> str:
#     response = model.chat.completions.create(
#         model="Gpt4mini",  # Replace with your actual deployment name
#         messages=[{"role": "user", "content": user_input}],
#         max_tokens=4000,
#         temperature=0.4,
#         seed=42,
#         n=1,
#     )
#     return response.choices[0].message.content