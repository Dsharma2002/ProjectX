from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing import List, Optional
from langchain_core.output_parsers import PydanticOutputParser

load_dotenv()

model = ChatOpenAI(model="gpt-5.4-mini")

# Define a structured data model for movies
class Movie(BaseModel):
    title: str
    director: str
    release_year: Optional[int]
    cast: List[str]
    plot_summary: str
    imdb_rating: Optional[float]
    
# parse the response using the PydanticOutputParser
# checks if the response is valid and returns a Movie object
output_parser = PydanticOutputParser(pydantic_object=Movie)

paragraph_1 = "Interstellar is a visually stunning science fiction epic directed by Christopher Nolan. "
"Released in 2014, the film stars Matthew McConaughey, Anne Hathaway, Jessica Chastain, and Michael Caine. " 
"The story revolves around a group of astronauts who travel through a wormhole near Saturn in search " 
"of a new home for humanity as Earth faces environmental collapse. "
"The movie was widely appreciated for its emotional depth, scientific accuracy, and Hans Zimmer's powerful soundtrack. "
"It holds a rating of 8.6 on IMDb and is often considered one of the greatest sci-fi films of the 21st century."

paragraph_2 = "The Shawshank Redemption is a critically acclaimed film directed by Frank Darabont. " 
"Released in 1994, the film stars Tim Robbins, Morgan Freeman. The story follows a banker " 
"who is sentenced to life in Shawshank prison and forms an unlikely friendship with a fellow inmate "
"while maintaining hope for freedom. The movie was widely praised for its storytelling, performances, and impact on cinema. "
"It holds a rating of 9.3 on IMDb and is considered one of the greatest films of all time."

paragraph_3 = "The Dark Knight is a critically acclaimed film directed by Christopher Nolan. "
"Released in 2008, the film stars Christian Bale, Heath Ledger, Aaron Eckhart. Batman faces "
"his greatest psychological and physical test against the chaotic Joker who seeks to plunge "
"Gotham City into anarchy. The movie was widely praised for its storytelling, performances, and impact on cinema. "
"It holds a rating of 9.0 on IMDb and is considered one of the greatest films of all time."

paragraph_4 = "The Matrix is a critically acclaimed film directed by The Wachowskis. "
"Released in 1999, the film stars Keanu Reeves, Laurence Fishburne, Carrie-Anne Moss. "
"A hacker discovers that reality is a simulated world and joins a rebellion against the machines "
"controlling humanity. The movie was widely praised for its storytelling, performances, and impact on cinema. "
"It holds a rating of 8.7 on IMDb and is considered one of the greatest films of all time."

paragraph_5 = "The Lord of the Rings trilogy is a series of epic fantasy films directed by Peter Jackson. "
"Released in 2001, the films star Elijah Wood, Ian McKellen, Viggo Mortensen, and Orlando Bloom. "
"The story follows the journey of Frodo Baggins, a hobbit, and his quest to destroy the One Ring "
"and save Middle-earth from the evil Sauron. The movies were widely praised for their storytelling, "
"performances, and impact on cinema. They hold a rating of 9.0 on IMDb and are considered one of the greatest films of all time."

chat_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are an information extraction system.

Your job is to extract structured data from a movie description.

Return the output ONLY as valid JSON that matches the schema exactly.

Rules:
1. Do NOT return explanations, only JSON.
2. Fill ALL fields:
   - If information is missing, make a reasonable inference.
   - Never leave lists empty (infer from context if needed).
3. Extract multiple cast members as a list of strings.
4. Ensure types are correct:
   - release_year → integer
   - imdb_rating → float
5. Keep plot_summary concise (1–2 sentences).
6. Do NOT output null unless absolutely impossible.

{format_instructions}
"""
        ),
        ("human", "{input}")
    ]
)

response = model.invoke(chat_prompt.format(format_instructions=output_parser.get_format_instructions(), input=paragraph_2))

print(response.content)