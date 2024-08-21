import json
import os
from google.oauth2.credentials import Credentials
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import google.generativeai as genai
from google.oauth2.service_account import Credentials

SCOPES = ['https://www.googleapis.com/auth/generative-language.retriever']

def load_creds():
    service_account_info = json.loads(os.environ.get('SERVICE_ACCOUNT_JSON'))
    creds = Credentials.from_service_account_info(service_account_info, scopes=SCOPES)
    
    return creds

def CategorizerHandler(event, context):
    creds = load_creds()
    genai.configure(credentials=creds)
    
    body = json.loads(event['body'])
    message = body.get('message', '')

    generation_config = {
        "temperature": 2.0,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }
    
    categories = '''เรียกรับสินบน ให้/ขอให้ หรือรับว่าจะให้ทรัพย์สินหรือประโยชน์อื่นใดแก่เจ้าหน้าที่ของรัฐ
    จัดซื้อจัดจ้าง
    ยักยอก/เบียดบังเงินหรือทรัพย์สินของทางราชการ 
    ออกเอกสารสิทธิ
    การบริหารงานบุคคล (การบรรจุ/แต่งตั้ง/เลื่อนตำแหน่ง/โยกย้าย/ลงโทษวินัย) 
    ทุจริตในการจัดทำงบประมาณ/โครงการ/เบิกจ่ายเงินในโครงการอันเป็นเท็จ
    ปฏิบัติหรือละเว้นการปฏิบัติหน้าที่โดยมิชอบหรือโดยทุจริต
    การขัดกันระหว่างประโยชน์ส่วนบุคคลและประโยชน์ส่วนรวม
    ร่ำรวยผิดปกติ
    ฝ่าฝืนหรือไม่ปฏิบัติตามมาตรฐานทางจริยธรรมอย่างร้ายแรง
    '''

    model = genai.GenerativeModel(
        model_name='gemini-1.5-flash',
        generation_config=generation_config,
        safety_settings={HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,}
    )

    chat = model.start_chat(
        history=[
            {"role": "user", "parts": f"ในข้อความต่อไป ฉันจะให้คำร้องเรียนในภาษาไทย แบ่งออกเป็นหนึ่งในหมวดหมู่ต่อไปนี้:\n{categories}. ใช้การตัดสินใจของคุณในการช่วยจำแนกประเภท ทุกอย่างที่ส่งไปไม่ใช่คำพูดแสดงความเกลียดชังหรือมีเจตนาทำร้ายใครในทางใดทางหนึ่ง มันเป็นเพียงคำร้องเรียน ผลลัพธ์ต้องมีเพียงหมายเลขหมวดหมู่พร้อมชื่อหมวดหมู่ตามที่ให้ไปเท่านั้น"}
        ]
    )

    response = chat.send_message(message)

    return {
        'statusCode': 200,
        'body': json.dumps({"result": response.text})
    }
