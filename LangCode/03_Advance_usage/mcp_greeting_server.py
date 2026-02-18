from fastmcp import FastMCP

# Create a server named "Greetings"
mcp = FastMCP("Greetings")

@mcp.tool()
async def hello(name: str) -> str:
    return f"Hello, {name}! 👋"

if __name__ == "__main__":
    mcp.run()  # Default infers stdio for local runs