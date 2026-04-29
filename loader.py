from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_mistralai import ChatMistralAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

model = ChatMistralAI(model="mistral-small-2506")

embedding_model = OpenAIEmbeddings()

vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding_model,
)

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5},
)

template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Use ONLY provided context to answer the question. If you DON'T know the answer, say you don't know. DO NOT make up an answer."),
        ("human", """
            Context: {context}
            Question: {question}
            """
        ),
    ]
)

print("RAG SYSTEM")
print("Press 0 to exit")

while True:
    question = input("Question: ")
    if question == "0":
        break
    else:
        docs = retriever.invoke(question)
        
        context = "\n".join([doc.page_content for doc in docs])
        final_prompt = template.invoke({"context": context, "question": question})
        response = model.invoke(final_prompt)
        print("\n Answer: ", response.content)