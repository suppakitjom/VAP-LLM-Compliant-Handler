import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
import os

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

ON_PREM = False

def CategorizerHandler(event, _):
    body = json.loads(event['body'])
    complaint = body['message']

    categories = [
        "เรียกรับสินบน ให้/ขอให้ หรือรับว่าจะให้ทรัพย์สินหรือประโยชน์อื่นใดแก่เจ้าหน้าที่ของรัฐ",
        "จัดซื้อจัดจ้าง",
        "ยักยอก/เบียดบังเงินหรือทรัพย์สินของทางราชการ",
        "ออกเอกสารสิทธิ",
        "การบริหารงานบุคคล (การบรรจุ/แต่งตั้ง/เลื่อนตำแหน่ง/โยกย้าย/ลงโทษวินัย)",
        "ทุจริตในการจัดทำงบประมาณ/โครงการ/เบิกจ่ายเงินในโครงการอันเป็นเท็จ",
        "ปฏิบัติหรือละเว้นการปฏิบัติหน้าที่โดยมิชอบหรือโดยทุจริต",
        "การขัดกันระหว่างประโยชน์ส่วนบุคคลและประโยชน์ส่วนรวม",
        "ร่ำรวยผิดปกติ",
        "ฝ่าฝืนหรือไม่ปฏิบัติตามมาตรฐานทางจริยธรรมอย่างร้ายแรง"
    ]

    prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            '''You are given a complaint in Thai and you must categorize it into only one of the following categories: {categories}. You must only provide the category name in full as the output. Do not make up category names'''.format(categories=', '.join(categories)),
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

    response = categorizer_chain.invoke({
        "complaint": complaint
    })
    
    return {
        'statusCode': 200,
        'body': json.dumps({"result": response})
    }
