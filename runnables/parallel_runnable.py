from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()

short_prompt = ChatPromptTemplate.from_template(
    "Explain {topic} to me in 20 words."
)

long_prompt = ChatPromptTemplate.from_template(
    "Explain {topic} to me in in detail."
)

model = ChatMistralAI(model="mistral-small-2506")

parser = StrOutputParser()

topic = "LangChain vs LangGraph"

# Two tasks in parallel
chain = RunnableParallel({
    "short": short_prompt | model | parser,
    "long": long_prompt | model | parser,
})

# run in parallel
response = chain.invoke({"topic": topic})
print(response)