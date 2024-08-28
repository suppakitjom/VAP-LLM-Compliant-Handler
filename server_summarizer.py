from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langserve import add_routes
from fastapi.middleware.cors import CORSMiddleware
import json
import os
from config import LANGCHAIN_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY, GOOGLE_API_KEY
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            '''You are a helpful assistant that digests a complaint in Thai given into JSON format, in Thai as follow:
    ### format
     Summary: '...', AllegedParty: '...', Accusation: '...', Location: '...', Amount: '...'
    
    ### field description
     where Summary is the concise summary of the complaint, and just explain what the alleged party is accused of, 
     AllegedParty is the person(people) or/and organization(s) being accused (NOT THE ACCUSER), if there are multiple, separate them with comma, 
     Accusation is the action that the perpetrator is accused of, 
     Location is the province, district, or subdistrict, 
     Amount is the amount of money, if there is any involved, in the format of (number) บาท for example 1,000,000 บาท
    
    ### additional information
     the Summary is the only field that is required. if information is not explicitly available for any other field, just leave it as empty string
     EVERY FIELD MUST BE FILLED IN FORMAL THAI LANGUAGE''',
        ),
        ("human", "{complaint}"),
    ]
)
ON_PREM = True
if ON_PREM:
    json_llm = ChatOllama(model="gemma2",format="json",temperature=0)
else:
    llm = ChatOpenAI(model="gpt-4o-mini")
    json_llm = llm.bind(response_format={"type": "json_object"})
    

parser = JsonOutputParser()

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple api server using Langchain's Runnable interfaces",
)

# Set all CORS enabled origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

add_routes(
    app,
    prompt | json_llm | parser,
    path="/summarizer",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8001)