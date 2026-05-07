from dotenv import load_dotenv
from langchain.tools import tool
from langchain_mistralai import ChatMistralAI
from rich import print

load_dotenv()

@tool
def get_text_length(text: str) -> int:
    """Returns the length of the given text."""
    return len(text)

llm = ChatMistralAI(model="mistral-small-2506")

# tool binding
# tool binding means that the tool is now available to the model for use
# tool calling means the LLM chooses to use the tools provided
# LLM doesn't execute the tool. It only suggest the tool call
llm_with_tools = llm.bind_tools([get_text_length])

# tool invocation
# result = llm.invoke("Return the length of the following text: 'Hello'")

# the following model uses additional tokens
result = llm_with_tools.invoke("Return the length of the following text: 'Hello'")
print(result) # now the model suggests the tool call but doesn't return 5 (the answer)

# Tool execution is when we actually run the tool after LLM decides to use it
# LLM selects the tool -> we execute it -> get the result -> send back to LLM

# tool execution by self
if result.tool_calls:
    tool_call = result.tool_calls[0]
    tool_args = tool_call['args']
    
    tool_result = get_text_length.invoke(tool_args)
    final_result = llm_with_tools.invoke(f"The length of the text is '{tool_result}'")
    print(final_result.content) # now the model returns the length of the text