import os
from langchain.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_groq import ChatGroq

from langchain_groq import ChatGroq
import os 
from dotenv import load_dotenv
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("MODEL_NAME")
MULTI_MODEL = os.getenv("MULTI_MODEL_NAME")

llm = ChatGroq(
    model=GROQ_MODEL,
    api_key=GROQ_API_KEY,
    temperature=0.6,
    max_tokens=512,
    timeout=30
)


@tool
def get_user_name(user_name: str) -> str:
    """give user name details """
    return f"Nice to meet you {user_name}! I will remember your name."


DB_URI = "postgresql://postgres:prem123@localhost:5432/langgraph_db"

print("Connecting to DB...")

with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    checkpointer.setup()

    print("Creating agent...")

    agent = create_agent(
        model=llm,
        tools=[get_user_name],
        checkpointer=checkpointer
    )

    print("AI Assistant Started (type exit to stop)\n")

    while True:
        user_input = input("You: ")

        if user_input.lower() == "exit":
            break

        response = agent.invoke(
            {
                "messages": [
                    {"role": "user", "content": user_input}
                ]
            },
            config={
                "configurable": {"thread_id": "user_1"}
            }
        )

        print("AI Response:", response["messages"][-1].content)
