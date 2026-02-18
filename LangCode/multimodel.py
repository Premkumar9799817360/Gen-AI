from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
import os
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

MULTI_MODEL = os.getenv("MULTI_MODEL_NAME")

llm = ChatGroq(
    model=MULTI_MODEL,
    groq_api_key=GROQ_API_KEY,
    temperature=1
)

message = HumanMessage(
    content=[
        {"type":"text", "text":"What do you see in this image?"},
        {
            "type":"image_url",
            "image_url":{"url":"https://i.imgur.com/7ntlWNJ.png"}
        }
    ]
)

response = llm.invoke([message])

print(response.content)