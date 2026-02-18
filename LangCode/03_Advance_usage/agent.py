# agent.py
import os
import asyncio
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langchain.messages import AIMessage, HumanMessage

load_dotenv()

GROQ_MODEL = os.getenv("MODEL_NAME")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    model=GROQ_MODEL,
    api_key=GROQ_API_KEY,
    temperature=0,
)
print("Groq LLM loaded successfully ✅")

async def main():
    client = MultiServerMCPClient(
        {
            "utils": {
                "transport": "http",
                "url": "http://localhost:8000/mcp"
            }
        }
    )

    tools = await client.get_tools()
    print("Loaded MCP tools:", [t.name for t in tools])

    agent = create_agent(model=llm, tools=tools)

    async def ask(query: str):
        # ask agent
        result = await agent.ainvoke(
            {"messages":[{"role":"user","content":query}]}
        )
        # extract last AI message text
        last_ai = None
        for m in reversed(result["messages"]):
            if isinstance(m, AIMessage):
                last_ai = m
                break
        return last_ai.text if last_ai else "<no response>"

    print("\nCRICKET:", await ask("What is India’s cricket score?"))
    print("\nSTOCK:", await ask("What is the current stock price of AAPL?"))
    print("\nCOMBINED:", await ask("Tell me both in one answer."))

if __name__ == "__main__":
    asyncio.run(main())