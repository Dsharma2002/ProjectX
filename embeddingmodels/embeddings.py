from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv()

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    dimensions=64
)

texts = ["Hello, how are you?", 
        "What is your name?"]

vectors = embeddings.embed_documents(texts)
print(vectors)