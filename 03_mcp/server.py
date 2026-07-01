"""
MCP server — exposes tools to any MCP-compatible client (including Claude).

Run standalone:  python server.py
Or launched as a subprocess by client.py via stdio transport.

Requires:  pip install mcp
"""
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("ToolServer")


# ---------------------------------------------------------------------------
# Math tools
# ---------------------------------------------------------------------------

@mcp.tool()
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


@mcp.tool()
def subtract(a: float, b: float) -> float:
    """Subtract b from a."""
    return a - b


@mcp.tool()
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


@mcp.tool()
def divide(a: float, b: float) -> float:
    """Divide a by b. Raises if b is zero."""
    if b == 0:
        raise ValueError("Division by zero")
    return a / b


@mcp.tool()
def power(base: float, exponent: float) -> float:
    """Raise base to the power of exponent."""
    return base ** exponent


# ---------------------------------------------------------------------------
# String tools
# ---------------------------------------------------------------------------

@mcp.tool()
def reverse_string(text: str) -> str:
    """Return the characters of text in reverse order."""
    return text[::-1]


@mcp.tool()
def count_vowels(text: str) -> dict:
    """Count the vowels in text, returning total and per-vowel breakdown."""
    vowels = "aeiouAEIOU"
    breakdown = {v: text.count(v) for v in set(vowels) if text.count(v) > 0}
    return {
        "text": text,
        "total_vowels": sum(breakdown.values()),
        "breakdown": breakdown,
    }


@mcp.tool()
def word_count(text: str) -> dict:
    """Return word, character, and sentence counts for text."""
    return {
        "words": len(text.split()),
        "characters": len(text),
        "sentences": sum(text.count(p) for p in ".!?"),
    }


if __name__ == "__main__":
    mcp.run()
