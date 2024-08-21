import json
import os
from langchain_openai import ChatOpenAI
from typing import Optional, List
from langchain_core.pydantic_v1 import BaseModel, Field

def SummarizerHandler(event, context):
    # Debugging: Print the event structure
    print("Event received:", json.dumps(event))

    # Safeguard: Check if 'body' is in the event
    if 'body' not in event:
        return {
            'statusCode': 400,
            'body': json.dumps({"error": "No 'body' found in the event"})
        }

    body = json.loads(event['body'])
    text_to_summarize = body.get('message', '')

    if not text_to_summarize:
        return {
            'statusCode': 400,
            'body': json.dumps({"error": "No 'message' found in the request body"})
        }

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

if __name__ == '__main__':
    event = {
        'body': json.dumps({"message": "This is a test message"})
    }
    SummarizerHandler(event, 'None')