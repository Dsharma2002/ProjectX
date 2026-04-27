from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

model = ChatOpenAI(model="gpt-5.4-mini")

messages = [
    SystemMessage(
        content="You are a helpful assistant that answers questions based on the context provided simply using analogies or examples."
    ),
]

print("Welcome! Type 'exit' to quit.")
while True:
    prompt = input("You: ")
    if prompt.lower() == "exit":
        break
    messages.append(HumanMessage(content=prompt))
    response = model.invoke(messages)
    messages.append(AIMessage(content=response.content))
    print("Bot: ", response.content)