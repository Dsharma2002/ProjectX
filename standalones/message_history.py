from dotenv import load_dotenv
from langchain.tools import tool
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage
from rich import print

load_dotenv()

@tool
def get_text_length(text: str) -> int:
    """Returns the length of the given text."""
    return len(text)

tools = {
    "get_text_length": get_text_length
}

llm = ChatMistralAI(model="mistral-small-2506")

llm_with_tools = llm.bind_tools([get_text_length])

#local storage
message_history = []

query = HumanMessage(content="Return the length of the following text: 'Hello, world!'")
message_history.append(query)

result = llm_with_tools.invoke(message_history)
message_history.append(result)

if result.tool_calls:
    tool_name = result.tool_calls[0]['name']
    llm_result = tools[tool_name].invoke(result.tool_calls[0])
    message_history.append(llm_result)
    print(llm_with_tools.invoke(message_history).content)