from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langserve import add_routes
from fastapi.middleware.cors import CORSMiddleware

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

ON_PREM = True
if ON_PREM:
    llm = ChatOllama(model="gemma2")
else:
    llm = ChatOpenAI(model="gpt-4o-mini")


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
    prompt | llm,
    path="/categorizer",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)