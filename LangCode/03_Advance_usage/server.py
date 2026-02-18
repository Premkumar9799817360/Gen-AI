# server.py
from fastmcp import FastMCP
import random

mcp = FastMCP("RealWorldUtils")

@mcp.tool()
async def get_cricket_score(team: str) -> str:
    runs = random.randint(100, 360)
    wickets = random.randint(0, 10)
    return f"{team.title()} scored {runs}/{wickets}"

@mcp.tool()
async def get_stock_price(symbol: str) -> str:
    price = round(random.uniform(100.0, 1500.0), 2)
    return f"{symbol.upper()} stock price is ${price}"

if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0", port=8000)