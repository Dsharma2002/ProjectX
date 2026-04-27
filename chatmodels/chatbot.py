from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI(model="gpt-5.4-mini")

messages = []

while True:
    print("Welcome! Type 'exit' to quit.")
    prompt = input("You: ")
    if prompt.lower() == "exit":
        break
    messages.append({"role": "user", "content": prompt})
    response = model.invoke(messages)
    messages.append({"role": "assistant", "content": response.content})
    print("Bot: ", response.content)