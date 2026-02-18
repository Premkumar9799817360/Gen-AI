import asyncio
from fastmcp import Client

async def main():
    client = Client("server.py")

    async with client:
        text = "FastMCP lets you expose real tools as a service. It can summarize, translate or extract keywords."

        # Summarize
        summary = await client.call_tool(
            "summarize",
            {"text": text, "max_len": 10}
        )
        print("Summary:", summary)

        # Translate
        translated = await client.call_tool(
            "translate",
            {"text": "Hello world from UtilityService!", "to_language": "fr"}
        )
        print("Translated:", translated)

        # Extract Keywords
        keywords = await client.call_tool("extract_keywords", {"text": text})
        print("Keywords:", keywords)

        # Stylize Text
        styled = await client.call_tool("stylize", {"text": summary})
        print("Stylized:\n", styled)

if __name__ == "__main__":
    asyncio.run(main())