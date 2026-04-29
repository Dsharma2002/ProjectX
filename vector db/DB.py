from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

docs = [
    Document(page_content="Python is widely used in Artificial Intelligence.", metadata={"source": "AI_book"}),
    Document(page_content="Pandas is used for data analysis.", metadata={"source": "Data_analysis_book"}),
    Document(page_content="Neural Networks are used in Deep Learning.", metadata={"source": "DeepLearning_book"}),
]

embedding_model = OpenAIEmbeddings()

vector_store = Chroma.from_documents(
    documents=docs, 
    embedding=embedding_model,
    persist_directory="chroma_db"
)

# Vector Stores are used to store and retrieve documents. 
# Vectore stores are not responsible for answering questions. That's the job of a language model.
result = vector_store.similarity_search("What is used for Data Analysis?", k=2)

print(result)
print("-------") 

retriever = vector_store.as_retriever(search_kwargs={"k": 2})

docs_result = retriever.invoke("What is used for Data Analysis?")

print(docs_result)