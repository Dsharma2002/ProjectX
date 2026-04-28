# web based loader
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader

load_dotenv()

# This example demonstrates how to use WebBaseLoader to load content from web page.
url = "https://www.apple.com/uk/watch/"

loader = WebBaseLoader(url)

documents = loader.load() 

print(documents[0].page_content)