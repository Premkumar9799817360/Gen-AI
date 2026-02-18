import os 
from dotenv import load_dotenv
load_dotenv()
from langchain_groq import ChatGroq


GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("MODEL_NAME")

print(GROQ_MODEL)
print(GROQ_API_KEY)
llm = ChatGroq(
    model=GROQ_MODEL,
    api_key=GROQ_API_KEY,
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2
)

print("LLM loaded successfully ✅")

message = [
    {"role": "system", "content": "You are a helpful assistant."},
]
ai_msg = llm.invoke(message)
print(ai_msg.contedfasgasd)