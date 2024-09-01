import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
import os

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

ON_PREM = False

def SummarizerHandler(event, _):
    body = json.loads(event['body'])
    complaint = body['message']

    summarizer_prompt = ChatPromptTemplate.from_messages(
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

    if ON_PREM:
        json_llm = ChatOllama(model="gemma2", format="json", temperature=0)
    else:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        json_llm = llm.bind(response_format={"type": "json_object"})

    json_parser = JsonOutputParser()

    summarizer_chain = summarizer_prompt | json_llm | json_parser

    response = summarizer_chain.invoke({
        "complaint": complaint
    })
    
    return {
        'statusCode': 200,
        'body': json.dumps({"result": response})
    }
