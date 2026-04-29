from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

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

similarity_retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

print("Similarity Search Results:")
similarity_results = similarity_retriever.invoke("What is gradient descent?")
for i, doc in enumerate(similarity_results):
    print(f"Document {i+1}: {doc.page_content}")
    
mmr_retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 3})

print("\nMMR Search Results:")
mmr_results = mmr_retriever.invoke("What is gradient descent?")
for i, doc in enumerate(mmr_results):
    print(f"Document {i+1}: {doc.page_content}")