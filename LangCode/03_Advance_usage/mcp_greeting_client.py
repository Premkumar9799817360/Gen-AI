import asyncio
from fastmcp import Client

async def main():
    # Pass the server script path
    client = Client("mcp_greeting_server.py")

    async with client:
        result = await client.call_tool("hello", {"name": "Alice"})
        print(result)

if __name__ == "__main__":
    asyncio.run(main())