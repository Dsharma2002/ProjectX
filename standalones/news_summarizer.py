from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

search_tool = TavilySearchResults(max_results=5)

llm = ChatMistralAI(model="mistral-small-2506")

prompt = ChatPromptTemplate.from_template(
    """
    You are a helpful assistant that summarizes the following news articles: {news}
    """
)

chain = prompt | llm | StrOutputParser()

result = search_tool.run("What is the latest on the FIFA 2026 World Cup?")
response = chain.invoke({"news": result})

print(response)