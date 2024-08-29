from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langserve import add_routes
from fastapi.middleware.cors import CORSMiddleware
import json
import os
from config import LANGCHAIN_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY, GOOGLE_API_KEY
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

ON_PREM = True

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
    json_llm = ChatOllama(model="gemma2",format="json",temperature=0)
else:
    llm = ChatOpenAI(model="gpt-4o-mini")
    json_llm = llm.bind(response_format={"type": "json_object"})

json_parser = JsonOutputParser()

summarizer_chain = summarizer_prompt | json_llm | json_parser

categories = '''เรียกรับสินบน ให้/ขอให้ หรือรับว่าจะให้ทรัพย์สินหรือประโยชน์อื่นใดแก่เจ้าหน้าที่ของรัฐ
จัดซื้อจัดจ้าง
ยักยอก/เบียดบังเงินหรือทรัพย์สินของทางราชการ 
ออกเอกสารสิทธิ
การบริหารงานบุคคล (การบรรจุ/แต่งตั้ง/เลื่อนตำแหน่ง/โยกย้าย/ลงโทษวินัย) 
ทุจริตในการจัดทำงบประมาณ/โครงการ/เบิกจ่ายเงินในโครงการอันเป็นเท็จ
ปฏิบัติหรือละเว้นการปฏิบัติหน้าที่โดยมิชอบหรือโดยทุจริต
การขัดกันระหว่างประโยชน์ส่วนบุคคลและประโยชน์ส่วนรวม
ร่ำรวยผิดปกติ
ฝ่าฝืนหรือไม่ปฏิบัติตามมาตรฐานทางจริยธรรมอย่างร้ายแรง'''
cat = []
for x in categories.split('\n'):
    cat.append(x.strip())

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            '''You are given a complaint in Thai and you must categorize it into only one of the following categories: {categories}. You must only provide the category name in full as the output. Do not make up category names'''.format(categories=', '.join(cat)),
        ),
        ("human", "{complaint}"),
    ]
)

if ON_PREM:
    llm = ChatOllama(model="gemma2")
else:
    llm = ChatOpenAI(model="gpt-4o-mini")

class StrOutputParserWithStrip(StrOutputParser):
    def parse(self, text):
        return super().parse(text).strip()
parser = StrOutputParserWithStrip()

categorizer_chain = prompt | llm | parser

app = FastAPI(
    title="NACC LLM Server",
    version="1.0",
    description="This is a server for NACC LLM",
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
    summarizer_chain,
    path="/summarizer",
)

add_routes(
    app,
    categorizer_chain,
    path="/categorizer",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)