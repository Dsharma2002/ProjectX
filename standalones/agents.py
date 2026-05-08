from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from tavily import TavilyClient
from rich import print

import os
import requests

load_dotenv()

# create tools 
@tool
def get_weather(location: str) -> str:
    """Get the current weather for the given location."""
    url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={os.getenv('OPENWEATHER_API_KEY')}&units=metric"
    response = requests.get(url)
    data = response.json()
    
    if str(data['cod']) != "200":
        return f"Error: {data['message']}"
    
    temp = data['main']['temp']
    description = data['weather'][0]['description']
    
    return f"The current temperature in {location} is {temp} degrees celsius. Desc: {description}."

# tavily news aggregator tool
@tool
def get_news(location: str) -> str:
    """Get the latest news for the given location."""
    client = TavilyClient(os.getenv('TAVILY_API_KEY'))
    
    query = f"Latest news in {location}"
    
    response = client.search(query=query, search_depth="basic", max_results=3)
    
    results = response.get('results', [])
    if not results:
        return f"No news found for {location}."
    
    news_list = []
    for result in results:
        title = result.get('title', 'No Title')
        url = result.get('url', 'No URL')
        snippet = result.get('content', '')
        news_list.append(f"- {title}\n  {url}\n  {snippet[:100]}...")
    
    return f"Latest news in {location}:\n" + "\n".join(news_list)

# model
llm = ChatMistralAI(model="mistral-small-2506")

# tools = {
#     "get_weather": get_weather,
#     "get_news": get_news
# }

# llm_with_tools = llm.bind_tools([get_weather, get_news])

# # Build lookup dict for tool dispatch
# tools_map = {t.name: t for t in tools}
tools = [get_weather, get_news]
tools_map = {t.name: t for t in tools}
llm_with_tools = llm.bind_tools(tools)
# Agent Loop
messages = []

print("City Intelligence Agent")
print("====================")
print("Type 'exit' to quit.")

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        break

    messages.append(HumanMessage(content=user_input))

    while True:
        result = llm_with_tools.invoke(messages)
        messages.append(result)

        if result.tool_calls:
            rejected = False
            for tool_call in result.tool_calls:
                tool_name = tool_call['name']
                confirmation = input(f"Agent wants to call tool '{tool_name}'. Do you approve? (yes/no): ")

                if confirmation.lower() == "no":
                    print(f"Tool '{tool_name}' call rejected by user.")
                    rejected = True
                    break
                elif confirmation.lower() == "yes":
                    tool_result = tools_map[tool_name].invoke(tool_call['args'])
                    messages.append(ToolMessage(content=str(tool_result), tool_call_id=tool_call['id']))

            if rejected:
                break  # exit inner while, go back to user input
            # else: continue inner while so LLM can reason with tool results
        else:
            print("Agent:", result.content)
            break  # final answer reached, exit inner while