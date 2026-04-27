import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# Load environment variables
load_dotenv()

# Initialize model
model = ChatOpenAI(model="gpt-5.4-mini")

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(
            content="You are a helpful assistant that answers questions based on the context provided simply using analogies or examples."
        )
    ]

st.title("Simple AI Chatbot")

# Display chat history (skip system message)
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    elif isinstance(msg, AIMessage):
        st.chat_message("assistant").write(msg.content)

# Input box
user_input = st.chat_input("Type your message...")

if user_input:
    # Add user message
    st.session_state.messages.append(HumanMessage(content=user_input))
    st.chat_message("user").write(user_input)

    # Get response
    response = model.invoke(st.session_state.messages)

    # Add AI response
    st.session_state.messages.append(AIMessage(content=response.content))
    st.chat_message("assistant").write(response.content)
