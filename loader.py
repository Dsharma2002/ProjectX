from dotenv import load_dotenv
# from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

model = ChatOpenAI(model="gpt-5.4-mini")

data = PyPDFLoader("documentLoaders/GRU.pdf").load()

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