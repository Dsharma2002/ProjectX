from langchain_core.documents import Document 
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_mistralai import ChatMistralAI 
from dotenv import load_dotenv

load_dotenv()

docs = [
    Document(page_content="Gradient descent is an optimization algorithm used in machine learning."),
    Document(page_content="Gradient descent minimizes the loss function."),
    Document(page_content="Gradient descent is an optimization that minimizes the loss function."),
    Document(page_content="Neural networks use gradient descent for training."),
    Document(page_content="Support Vector Machines are supervised learning algorithms.")
]

embeddings = HuggingFaceEmbeddings()

vector_store = Chroma.from_documents(
    documents=docs, 
    embedding=embeddings,
)

retriever = vector_store.as_retriever()

llm = ChatMistralAI(model="mistral-small-latest")

multi_query_retriever = MultiQueryRetriever.from_llm(retriever=retriever, llm=llm)

print("Multi Query Retriever Results:")
results = multi_query_retriever.invoke("What is gradient descent?")
for i, doc in enumerate(results):
    print(f"Document {i+1}: {doc.page_content}")
