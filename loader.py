from dotenv import load_dotenv
# from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
# from langchain_openai import ChatOpenAI
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

## ChatOpenAI is paid.
# model = ChatOpenAI(model="gpt-5.4-mini")

# prepare the model
model = ChatMistralAI(model="mistral-small-2506")

# prompt template for summarization
template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant that summarizes text."),
        ("human", "{text}"),
    ]
)