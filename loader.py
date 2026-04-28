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
# load the document
data = PyPDFLoader("documentLoaders/deeplearning.pdf").load()
# prepare the text splitter
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
# split the document into chunks. This is important for large documents, as most language models have a context window limit.
chunks = splitter.split_documents(data)

# prompt template for summarization
template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant that summarizes text."),
        ("human", "{text}"),
    ]
)

# This sends the entire document to the model, which may not be ideal for large documents. 
# In practice, you might want to chunk the document and summarize each chunk separately.
prompt = template.format_prompt(text=data)

response = model.invoke(prompt)

print(response.content)