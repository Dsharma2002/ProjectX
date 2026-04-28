from dotenv import load_dotenv
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader 

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import TokenTextSplitter

load_dotenv()

# splitter = CharacterTextSplitter(separator="", chunk_size=10, chunk_overlap=2)
splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
data = TextLoader("notes.txt").load()
chunks = splitter.split_documents(data)

# for chunk in chunks:
#     print(chunk.page_content)

med_data = PyPDFLoader("GRU.pdf")
docs = med_data.load()
token_splitter = TokenTextSplitter(
    chunk_size=1000, 
    chunk_overlap=50
)
token_chunks = token_splitter.split_documents(docs)
