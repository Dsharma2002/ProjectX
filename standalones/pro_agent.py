from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from tavily import TavilyClient
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
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

@wrap_tool_call
def human_approval(request, handler):
    """Ask for human approval before calling the tool."""
    tool_name = request.tool_call["name"]
    confirm = input(f"Agent wants to call tool '{tool_name}'. Do you approve? (yes/no): ")
    if confirm.lower() == "no":
        return ToolMessage(
            content="Tool call rejected by user.",
            tool_call_id=request.tool_call["id"],
        )
    return handler(request)

# model
llm = ChatMistralAI(model="mistral-small-2506")

agent = create_agent(
    llm,
    tools=[get_weather, get_news],
    system_prompt="You are a helpful city assistant.",
    middleware=[human_approval],
)

print("City Intelligence Agent")
print("====================")
print("Type 'exit' to quit.")

while True:
    user_input = input("You: ")
    
    if user_input.lower() == "exit":
        break
    
    result = agent.invoke(
        {"messages": [{"role": "user", "content": user_input}]}
    )
    print(f"Agent: {result['messages'][-1].content}")