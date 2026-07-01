"""
MCP client — connects to server.py via stdio, discovers its tools, then lets
Claude use those tools through the SDK's async tool runner.

Usage:  python client.py
Requires:  pip install anthropic[mcp] mcp
"""
import asyncio
import sys
from pathlib import Path

import anthropic
from anthropic.lib.tools.mcp import async_mcp_tool
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SERVER_PATH = Path(__file__).parent / "server.py"


async def main() -> None:
    client = anthropic.AsyncAnthropic()

    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(SERVER_PATH)],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as mcp_client:
            await mcp_client.initialize()

            # Discover all tools the server exposes
            tools_result = await mcp_client.list_tools()
            print(f"Discovered {len(tools_result.tools)} MCP tools:")
            for t in tools_result.tools:
                print(f"  • {t.name}: {t.description}")

            # Build async-callable wrappers for the tool runner
            mcp_tools = [async_mcp_tool(t, mcp_client) for t in tools_result.tools]

            questions = [
                "What is (3.5 + 4.5) * 2?  And what is 2 to the power of 10?",
                "Reverse the string 'Hello, MCP World!' and then count its vowels.",
                "How many words and sentences are in: 'The quick brown fox jumps. It was fast!'",
            ]

            for question in questions:
                print(f"\n{'─'*60}")
                print(f"User: {question}")

                # tool_runner is sync (returns the runner object); iterate async
                runner = client.beta.messages.tool_runner(
                    model="claude-opus-4-8",
                    max_tokens=4096,
                    messages=[{"role": "user", "content": question}],
                    tools=mcp_tools,
                )

                async for message in runner:
                    for block in message.content:
                        if block.type == "text":
                            print(f"Claude: {block.text}")


if __name__ == "__main__":
    asyncio.run(main())
