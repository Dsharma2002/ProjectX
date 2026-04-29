import streamlit as st
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_mistralai import ChatMistralAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tempfile
import os

# Load env
load_dotenv()

# Initialize model and embeddings
model = ChatMistralAI(model="mistral-small-2506")
embedding_model = OpenAIEmbeddings()

st.title("RAG PDF Chatbot")

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# Process PDF
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    loader = PyPDFLoader(tmp_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model
    )

    st.session_state.vectorstore = vectorstore
    st.success("PDF processed successfully!")

# Ask question
question = st.text_input("Ask a question")

if question and st.session_state.vectorstore:
    retriever = st.session_state.vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5},
    )

    docs = retriever.invoke(question)
    context = "\n".join([doc.page_content for doc in docs])

    template = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Use ONLY provided context to answer the question. If you DON'T know the answer, say you don't know. DO NOT make up an answer."),
        ("human", """
        Context: {context}
        Question: {question}
        """),
    ])

    final_prompt = template.invoke({"context": context, "question": question})
    response = model.invoke(final_prompt)

    st.write("### Answer")
    st.write(response.content)

elif question and not st.session_state.vectorstore:
    st.warning("Please upload and process a PDF first.")
