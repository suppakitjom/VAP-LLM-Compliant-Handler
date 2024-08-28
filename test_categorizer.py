from langchain_core.prompts import ChatPromptTemplate
from langserve import RemoteRunnable


categorizer = RemoteRunnable("http://localhost:8000/categorizer/")

r = categorizer.invoke({
        "complaint": 'ตำรวจเรียกรับสินบนจากประชาชน',
    })
print(r.strip())