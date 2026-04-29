from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt = ChatPromptTemplate.from_template(
    "Explain {topic} to me in simple terms."
)

model = ChatMistralAI(model="mistral-small-2506")

parser = StrOutputParser()

# chain
LLM_chain = prompt | model | parser 

# run
response = LLM_chain.invoke({"topic": "What does this mean: There is never a perfect moment"})
print(response)