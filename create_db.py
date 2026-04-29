# load the pdf
# split the pdf into chunks
# create the embeddings
# create the vector store and store (in Chroma)


from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

loader = PyPDFLoader("documentLoaders/deeplearning.pdf")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

chunks = splitter.split_documents(documents)

embedding_model = OpenAIEmbeddings()

vectore_store = Chroma.from_documents(
    documents=chunks, 
    embedding=embedding_model,
    persist_directory="chroma_db"
)