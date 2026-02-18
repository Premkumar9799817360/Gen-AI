import os
from dotenv import load_dotenv
from groq import Groq


load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

client = Groq(api_key=GROQ_API_KEY)


def LLM_MODEL(prompt: str):
    completion = client.chat.completions.create(
    model=MODEL_NAME,
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ])
    return completion.choices[0].message.content

print("LLM loaded successfully ✅")
print(LLM_MODEL("What is the capital of France?"))