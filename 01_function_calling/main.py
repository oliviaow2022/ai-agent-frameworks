"""
Function calling / tool use with Claude.

Demonstrates a manual agentic loop where Claude decides when and which tools
to call, executes them, and loops until it produces a final answer.
"""
import json
import anthropic

client = anthropic.Anthropic()
MODEL = "claude-opus-4-8"

TOOLS = [
    {
        "name": "get_weather",
        "description": (
            "Get current weather for a city. Call this when the user asks about "
            "weather, temperature, or conditions in a location."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name, e.g. 'Tokyo'"},
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature unit (default: celsius)",
                },
            },
            "required": ["city"],
        },
    },
    {
        "name": "calculate",
        "description": (
            "Evaluate a mathematical expression. Call this for any arithmetic, "
            "algebra, or numeric computation."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Math expression, e.g. '2 * (3 + 4)' or '15.5 * 8 + 42'",
                },
            },
            "required": ["expression"],
        },
    },
    {
        "name": "search_web",
        "description": (
            "Search the web for factual, current, or time-sensitive information. "
            "Call this when you need data beyond your training cutoff."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query string"},
            },
            "required": ["query"],
        },
    },
]


def execute_tool(name: str, input_data: dict) -> str:
    """Dispatch a tool call and return the result as a JSON string."""
    if name == "get_weather":
        city = input_data["city"]
        unit = input_data.get("unit", "celsius")
        temp = 22 if unit == "celsius" else 72  # stub value
        return json.dumps({
            "city": city,
            "temperature": temp,
            "unit": unit,
            "condition": "Partly cloudy",
            "humidity": "65%",
        })

    if name == "calculate":
        expr = input_data["expression"]
        # Guard: only allow numeric and operator characters
        allowed = set("0123456789+-*/()., eE")
        if not all(c in allowed for c in expr):
            return json.dumps({"error": "Expression contains invalid characters"})
        try:
            result = eval(expr)  # noqa: S307 — guarded by character whitelist above
            return json.dumps({"expression": expr, "result": result})
        except Exception as exc:
            return json.dumps({"error": str(exc)})

    if name == "search_web":
        query = input_data["query"]
        # Stub search results — swap in a real API (e.g. Brave, Serper) here
        return json.dumps({
            "query": query,
            "results": [
                {
                    "title": f"Top result for: {query}",
                    "snippet": "Stub result — replace with a real search API integration.",
                    "url": "https://example.com/result-1",
                },
                {
                    "title": f"Second result for: {query}",
                    "snippet": "Another stub result with relevant-looking content.",
                    "url": "https://example.com/result-2",
                },
            ],
        })

    return json.dumps({"error": f"Unknown tool: {name}"})


def run_agent(user_message: str) -> str:
    """Run the agentic loop until Claude produces a final text response."""
    messages = [{"role": "user", "content": user_message}]

    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            tools=TOOLS,
            messages=messages,
        )

        if response.stop_reason == "end_turn":
            return next((b.text for b in response.content if b.type == "text"), "")

        if response.stop_reason != "tool_use":
            return f"Unexpected stop_reason: {response.stop_reason}"

        # Append Claude's response (including tool_use blocks) to history
        messages.append({"role": "assistant", "content": response.content})

        # Execute every tool Claude requested and collect results
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            result = execute_tool(block.name, block.input)
            print(f"  [tool] {block.name}({json.dumps(block.input)}) → {result[:100]}")
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": result,
            })

        messages.append({"role": "user", "content": tool_results})


if __name__ == "__main__":
    queries = [
        "What's the weather like in Tokyo and Paris right now?",
        "What is (15.5 * 8) + 42, and what's the square root of 144?",
        "Search for recent news about large language models.",
    ]

    for query in queries:
        print(f"\n{'─'*60}")
        print(f"User: {query}")
        answer = run_agent(query)
        print(f"Claude: {answer}")
