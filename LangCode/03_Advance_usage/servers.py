from fastmcp import FastMCP

mcp = FastMCP("UtilityService")

# Tool — summarize text
@mcp.tool()
async def summarize(text: str, max_len: int = 50) -> str:
    """
    Summarize given text into shorter form.
    (Simple split & truncate for demo; real ML can be used inside this)
    """
    words = text.split()
    summary = " ".join(words[:max_len])
    return summary + ("..." if len(words) > max_len else "")

# Tool — mock translate
@mcp.tool()
async def translate(text: str, to_language: str = "es") -> str:
    """
    Mock translation tool — adds a tag.
    (In real use: call actual translation API)
    """
    return f"[{to_language} translation] {text}"

# Tool — extract simple keywords
@mcp.tool()
async def extract_keywords(text: str) -> list[str]:
    """
    Extract simple keywords: take unique words longer than 3 chars
    """
    words = text.replace(".", "").lower().split()
    keywords = list({w for w in words if len(w) > 3})
    return keywords

# Tool — format text with uppercase header
@mcp.tool()
async def stylize(text: str) -> str:
    """
    Make stylized text — an uppercase header + text
    """
    header = "~~~ RESULT ~~~"
    return f"{header}\n{text}"

if __name__ == "__main__":
    mcp.run()