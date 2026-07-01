"""
Agent skills / plugin system.

Skills are Python functions decorated with metadata and registered in a
SkillRegistry. The registry converts them to Claude tool definitions, so the
agent can invoke any registered skill via the normal tool-use loop.
"""
import json
from dataclasses import dataclass, field
from typing import Any, Callable
import anthropic

client = anthropic.Anthropic()
MODEL = "claude-opus-4-8"


# ---------------------------------------------------------------------------
# Skill registry
# ---------------------------------------------------------------------------

@dataclass
class Skill:
    name: str
    description: str
    func: Callable
    parameters: dict = field(default_factory=dict)

    def to_tool(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": self.parameters,
                "required": list(self.parameters.keys()),
            },
        }

    def execute(self, **kwargs: Any) -> str:
        result = self.func(**kwargs)
        return json.dumps(result) if not isinstance(result, str) else result


class SkillRegistry:
    def __init__(self):
        self._skills: dict[str, Skill] = {}

    def register(self, description: str, parameters: dict):
        """Decorator factory — annotates a function and registers it as a skill."""
        def decorator(func: Callable) -> Callable:
            self._skills[func.__name__] = Skill(
                name=func.__name__,
                description=description,
                func=func,
                parameters=parameters,
            )
            return func
        return decorator

    def list_skills(self) -> list[str]:
        return list(self._skills.keys())

    def to_tools(self) -> list[dict]:
        return [s.to_tool() for s in self._skills.values()]

    def execute(self, name: str, kwargs: dict) -> str:
        skill = self._skills.get(name)
        if not skill:
            return json.dumps({"error": f"Unknown skill: {name}"})
        try:
            return skill.execute(**kwargs)
        except Exception as exc:
            return json.dumps({"error": str(exc)})


registry = SkillRegistry()


# ---------------------------------------------------------------------------
# Registered skills (plug in your own below)
# ---------------------------------------------------------------------------

@registry.register(
    description="Summarize a block of text into 3-5 concise bullet points.",
    parameters={
        "text": {"type": "string", "description": "The text to summarize"},
    },
)
def summarize_text(text: str) -> dict:
    # Stub — in production: call another LLM or an NLP service
    sentences = [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]
    bullets = [f"• {s}" for s in sentences[:5]]
    return {"bullets": bullets, "original_length": len(text)}


@registry.register(
    description=(
        "Convert a numeric value between common units: "
        "length (km/miles), weight (kg/lbs), temperature (celsius/fahrenheit)."
    ),
    parameters={
        "value": {"type": "number", "description": "Numeric value to convert"},
        "from_unit": {"type": "string", "description": "Source unit, e.g. 'km', 'celsius'"},
        "to_unit": {"type": "string", "description": "Target unit, e.g. 'miles', 'fahrenheit'"},
    },
)
def convert_units(value: float, from_unit: str, to_unit: str) -> dict:
    converters: dict[tuple, Callable] = {
        ("km", "miles"):          lambda x: x * 0.621371,
        ("miles", "km"):          lambda x: x * 1.60934,
        ("kg", "lbs"):            lambda x: x * 2.20462,
        ("lbs", "kg"):            lambda x: x * 0.453592,
        ("celsius", "fahrenheit"): lambda x: x * 9 / 5 + 32,
        ("fahrenheit", "celsius"): lambda x: (x - 32) * 5 / 9,
        ("meters", "feet"):       lambda x: x * 3.28084,
        ("feet", "meters"):       lambda x: x * 0.3048,
    }
    key = (from_unit.lower(), to_unit.lower())
    if key not in converters:
        return {"error": f"No conversion registered for {from_unit} → {to_unit}"}
    converted = converters[key](value)
    return {
        "input": f"{value} {from_unit}",
        "output": f"{converted:.4f} {to_unit}",
        "factor": converted / value if value else None,
    }


@registry.register(
    description="Count characters, words, sentences, and paragraphs in text.",
    parameters={
        "text": {"type": "string", "description": "Text to analyse"},
    },
)
def analyse_text(text: str) -> dict:
    return {
        "characters": len(text),
        "words": len(text.split()),
        "sentences": sum(text.count(p) for p in ".!?"),
        "paragraphs": len([p for p in text.split("\n\n") if p.strip()]),
        "avg_word_length": (
            round(sum(len(w) for w in text.split()) / len(text.split()), 1)
            if text.split() else 0
        ),
    }


@registry.register(
    description="Generate a slug (URL-friendly string) from a title or phrase.",
    parameters={
        "text": {"type": "string", "description": "Text to slugify"},
    },
)
def slugify(text: str) -> dict:
    import re
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower().strip()).strip("-")
    return {"original": text, "slug": slug}


@registry.register(
    description="Look up the definition or explanation of a technical term (stub).",
    parameters={
        "term": {"type": "string", "description": "Term to look up"},
    },
)
def lookup_term(term: str) -> dict:
    # Stub dictionary — swap in a real API (dictionary, Wikipedia, etc.)
    glossary = {
        "api": "Application Programming Interface — a contract for how software components communicate.",
        "llm": "Large Language Model — a neural network trained on text to generate or understand language.",
        "mcp": "Model Context Protocol — a standard for exposing tools and resources to LLMs.",
        "a2a": "Agent-to-Agent — a communication pattern where AI agents delegate tasks to each other.",
    }
    key = term.lower().strip()
    if key in glossary:
        return {"term": term, "definition": glossary[key]}
    return {"term": term, "definition": f"No definition found for '{term}' in the glossary."}


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

def run_agent(user_message: str) -> str:
    """Run the agentic loop, making the full skill registry available to Claude."""
    tools = registry.to_tools()
    messages = [{"role": "user", "content": user_message}]

    print(f"  [registry] {len(tools)} skills loaded: {registry.list_skills()}")

    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            tools=tools,
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
            output = registry.execute(block.name, block.input)
            print(f"  [skill:{block.name}] {json.dumps(block.input)} → {output[:120]}")
            results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": output,
            })
        messages.append({"role": "user", "content": results})


if __name__ == "__main__":
    tasks = [
        "How many miles is 100 km? And how many lbs is 75 kg?",
        "What does MCP stand for and what is it?",
        (
            "Analyse this text, summarize it, and generate a URL slug for it:\n"
            "The quick brown fox jumps over the lazy dog. "
            "This sentence is a pangram — it contains every letter of the English alphabet. "
            "Pangrams are commonly used to test fonts and keyboards."
        ),
    ]
    for task in tasks:
        print(f"\n{'─'*60}")
        print(f"User: {task[:80]}{'...' if len(task) > 80 else ''}")
        answer = run_agent(task)
        print(f"Claude: {answer}")
