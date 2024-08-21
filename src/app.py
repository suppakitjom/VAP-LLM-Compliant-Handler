import json
import os
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from langchain_openai import ChatOpenAI
from typing import Optional, List
from langchain_core.pydantic_v1 import BaseModel, Field

# Replace S3 bucket usage with environment variables
CLIENT_SECRET_JSON = os.environ.get('CLIENT_SECRET_JSON')

SCOPES = ['https://www.googleapis.com/auth/generative-language.retriever']

def create_file_from_env(content, path):
    """Create a file from the environment variable content."""
    with open(path, 'w') as file:
        file.write(content)

def load_creds():
    creds = None
    token_path = '/tmp/token.json'
    
    # Create client_secret.json from environment variable
    create_file_from_env(CLIENT_SECRET_JSON, '/tmp/client_secret.json')

    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                '/tmp/client_secret.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open(token_path, 'w') as token:
            token.write(creds.to_json())
    
    return creds

def CategorizerHandler(event):
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

def SummarizerHandler(event):
    body = json.loads(event['body'])
    text_to_summarize = body.get('message', '')

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

    class Summarizer(BaseModel):
        Summary: str = Field(description="the summary, be concise")
        Perpetrator: Optional[List[str]] = Field(description="The person(people) or/and organization(s) being accused")
        Action: Optional[str] = Field(description="the action")
        Where: Optional[str] = Field(description="province, district, or subdistrict")
        Amount: Optional[str] = Field(description="the amount of money, if there is any involved")

        def __str__(self):
            return f"Summary: {self.Summary}\nPerpetrator: {self.Perpetrator}\nAction: {self.Action}\nWhere: {self.Where}\nAmount: {self.Amount}"

    model = ChatOpenAI(model="gpt-4o-mini")
    structured_model = model.with_structured_output(Summarizer)

    result = structured_model.invoke(text_to_summarize)

    return {
        'statusCode': 200,
        'body': json.dumps({"summary": str(result)})
    }

def lambda_handler(event, context):
    # Check if 'httpMethod' is in the event to determine if it came from API Gateway
    if 'httpMethod' in event:
        if event['httpMethod'] == 'POST':
            path = event.get('path', '')
            body = json.loads(event.get('body', '{}'))
            
            if path == '/categorize':
                return CategorizerHandler(event)
            elif path == '/summarize':
                return SummarizerHandler(event)
            else:
                return {
                    'statusCode': 404,
                    'body': json.dumps({"error": "Not Found"})
                }
        else:
            return {
                'statusCode': 405,
                'body': json.dumps({"error": "Method Not Allowed"})
            }
    else:
        return {
            'statusCode': 400,
            'body': json.dumps({"error": "Invalid event structure"})
        }
