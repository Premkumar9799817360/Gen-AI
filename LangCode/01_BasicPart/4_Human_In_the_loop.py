from langchain.agents import create_agent
from langchain.tools import tool
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver

import os
import uuid
import pprint
from dotenv import load_dotenv


# =====================================================
# 🔐 Load Environment Variables
# =====================================================
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("MODEL_NAME")

if not GROQ_API_KEY:
    raise ValueError("❌ GROQ_API_KEY not found in .env")

if not GROQ_MODEL:
    raise ValueError("❌ MODEL_NAME not found in .env")


# =====================================================
# 🤖 Model Setup
# =====================================================
model = ChatGroq(
    model=GROQ_MODEL,
    api_key=GROQ_API_KEY,
    temperature=0,
)


# =====================================================
# 🛠 Tools
# =====================================================
@tool
def send_email(to: str, subject: str, body: str) -> dict:
    """Send an email. Requires human approval."""
    return {
        "status": "success",
        "content": f'Email sent to {to} with subject "{subject}"',
    }


@tool
def delete_file(path: str) -> dict:
    """Delete a file. Requires human approval."""
    return {
        "status": "success",
        "content": f'File "{path}" deleted',
    }


@tool
def read_file(path: str) -> dict:
    """Read file contents. No approval needed."""
    return {
        "status": "success",
        "content": f"Contents of {path}...",
    }


# =====================================================
# 🧠 Memory Setup
# =====================================================
memory = MemorySaver()


# =====================================================
# 👤 Human-in-the-Loop Middleware
# =====================================================
middleware = HumanInTheLoopMiddleware(
    interrupt_on={
        "send_email": {
            "allowed_decisions": ["approve", "edit", "reject"],
            "description": "📧 Review email before sending",
        },
        "delete_file": {
            "allowed_decisions": ["approve", "reject"],
            "description": "🗑️ Confirm file deletion",
        },
        "read_file": False,  # auto-approved
    }
)


# =====================================================
# 🚀 Create Agent
# =====================================================
agent = create_agent(
    model=model,
    tools=[send_email, delete_file, read_file],
    middleware=[middleware],
    checkpointer=memory,
)


# =====================================================
# 🖥 MAIN PROGRAM
# =====================================================
if __name__ == "__main__":

    # Unique thread id (conversation memory)
    # thread_id = str(uuid.uuid4())

    config = {
        "configurable": {
            "thread_id": "user_id_1"
        }
    }

    print("\n=====================================")
    print("🚀 LangChain Agent Started")
    print("Type 'exit' to quit")
    print("Press ENTER twice to submit message")
    print("=====================================\n")

    while True:

        print("👤 You (multi-line input, empty line to send):")

        # Multi-line input support
        lines = []
        while True:
            line = input()
            if line == "":
                break
            lines.append(line)

        user_input = "\n".join(lines)

        if user_input.lower() == "exit":
            print("👋 Exiting...")
            break

        if not user_input.strip():
            continue

        # ==========================
        # 🔄 Invoke Agent
        # ==========================
        result = agent.invoke(
            {"messages": [{"role": "user", "content": user_input}]},
            config=config,
        )

        # ==========================
        # 🤖 Agent Final Response
        # ==========================
        print("\n🤖 Agent Response:\n")
        print(result["messages"][-1].content)

        # ==========================
        # 📦 Print Full Agent State
        # ==========================
        print("\n============================")
        print("📦 FULL AGENT STATE")
        print("============================\n")
        pprint.pprint(result)

        # ==========================
        # 🧠 Print Memory State
        # ==========================
        print("\n============================")
        print("🧠 MEMORY STATE")
        print("============================\n")
        state = memory.get(config)
        pprint.pprint(state)

        print("\n" + "=" * 60 + "\n")