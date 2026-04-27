from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI(model="gpt-5.4-mini")
response = model.invoke("Hello, how are you?")

print(response)