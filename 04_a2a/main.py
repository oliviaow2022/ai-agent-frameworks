"""
Agent-to-Agent (A2A) communication.

An orchestrator agent delegates work to three specialist sub-agents:
  • researcher  — gathers structured background on a topic
  • writer      — drafts a readable article from the research
  • critic      — scores and reviews the draft

The orchestrator uses tool_use to call each sub-agent in sequence, then
synthesises the final result.  Each sub-agent is a separate Claude call with
its own system prompt — a lightweight, dependency-free A2A pattern.
"""
import json
import anthropic

client = anthropic.Anthropic()
MODEL = "claude-opus-4-8"


# ---------------------------------------------------------------------------
# Sub-agent helpers
# ---------------------------------------------------------------------------

def call_agent(system_prompt: str, user_message: str, max_tokens: int = 2048) -> str:
    """Invoke a specialist agent and return its text response."""
    response = client.messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
        thinking={"type": "adaptive"},
    )
    return next((b.text for b in response.content if b.type == "text"), "")


RESEARCHER_PROMPT = """You are a research specialist. Given a topic, produce a JSON object with:
- "key_facts": list of 3-5 important facts
- "perspectives": list of 2-3 main schools of thought or viewpoints
- "open_questions": list of 2-3 unresolved questions or debates
- "examples": list of 2-3 concrete real-world examples

Respond with valid JSON only. No markdown fences."""

WRITER_PROMPT = """You are a technical writer. Given research notes, produce:
1. An executive summary (2-3 sentences)
2. Three short paragraphs expanding on the key points
3. A single "Key Takeaway" sentence

Write clearly for a non-specialist audience. Use plain prose, no bullet lists."""

CRITIC_PROMPT = """You are a critical reviewer. Score the draft on three dimensions (1-10):
- "clarity": how readable and clear is it?
- "accuracy": does it faithfully represent the research?
- "completeness": does it cover the main points adequately?

Also provide "suggestions": a list of 2-3 specific improvements.

Respond with valid JSON only. No markdown fences."""


def researcher_agent(topic: str) -> dict:
    print(f"  [researcher] investigating: {topic!r}")
    raw = call_agent(RESEARCHER_PROMPT, f"Research this topic: {topic}")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"raw_research": raw}


def writer_agent(research: dict, topic: str) -> str:
    print("  [writer] drafting article...")
    prompt = f"Topic: {topic}\n\nResearch notes (JSON):\n{json.dumps(research, indent=2)}\n\nWrite the article."
    return call_agent(WRITER_PROMPT, prompt, max_tokens=3000)


def critic_agent(draft: str) -> dict:
    print("  [critic] reviewing draft...")
    raw = call_agent(CRITIC_PROMPT, f"Review this draft:\n\n{draft}")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"raw_review": raw}


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

ORCHESTRATOR_TOOLS = [
    {
        "name": "delegate_research",
        "description": "Send a topic to the research specialist agent and receive structured background data.",
        "input_schema": {
            "type": "object",
            "properties": {
                "topic": {"type": "string", "description": "Topic to research"},
            },
            "required": ["topic"],
        },
    },
    {
        "name": "delegate_writing",
        "description": "Send research data to the writer agent and receive a drafted article.",
        "input_schema": {
            "type": "object",
            "properties": {
                "topic": {"type": "string", "description": "Article topic"},
                "research_json": {
                    "type": "string",
                    "description": "JSON string of structured research data from the researcher agent",
                },
            },
            "required": ["topic", "research_json"],
        },
    },
    {
        "name": "delegate_review",
        "description": "Send a draft to the critic agent and receive a quality review with scores.",
        "input_schema": {
            "type": "object",
            "properties": {
                "draft": {"type": "string", "description": "Article draft to review"},
            },
            "required": ["draft"],
        },
    },
]

ORCHESTRATOR_PROMPT = """You are an editorial orchestrator. To produce a high-quality article:
1. Call delegate_research to gather background on the topic.
2. Call delegate_writing, passing the research as JSON, to produce a draft.
3. Call delegate_review to get quality scores and feedback on the draft.
4. Summarise the article and include the quality scores in your final response.

Always use the delegation tools in order. Never write the article yourself."""


def run_orchestrator(topic: str) -> str:
    print(f"\nOrchestrator: producing article on {topic!r}")
    messages = [{"role": "user", "content": f"Produce a reviewed article about: {topic}"}]
    state: dict = {}  # shared state passed between tool calls

    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=6000,
            system=ORCHESTRATOR_PROMPT,
            tools=ORCHESTRATOR_TOOLS,
            messages=messages,
        )

        if response.stop_reason == "end_turn":
            return next((b.text for b in response.content if b.type == "text"), "")

        if response.stop_reason != "tool_use":
            return f"Unexpected stop_reason: {response.stop_reason}"

        messages.append({"role": "assistant", "content": response.content})
        results = []

        for block in response.content:
            if block.type != "tool_use":
                continue

            if block.name == "delegate_research":
                data = researcher_agent(block.input["topic"])
                state["research"] = data
                content = json.dumps(data)

            elif block.name == "delegate_writing":
                research = json.loads(block.input.get("research_json", "{}"))
                draft = writer_agent(research, block.input["topic"])
                state["draft"] = draft
                content = draft

            elif block.name == "delegate_review":
                review = critic_agent(block.input["draft"])
                state["review"] = review
                content = json.dumps(review)

            else:
                content = json.dumps({"error": f"Unknown delegation: {block.name}"})

            results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": content,
            })

        messages.append({"role": "user", "content": results})


if __name__ == "__main__":
    topics = [
        "How transformer attention mechanisms work",
    ]
    for topic in topics:
        result = run_orchestrator(topic)
        print(f"\n{'='*60}")
        print("FINAL OUTPUT FROM ORCHESTRATOR:")
        print(result)
