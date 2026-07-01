"""
Harness engineering — agent orchestration scaffolding.

Demonstrates:
  • ToolRegistry   — decorator-based tool registration
  • Middleware      — pre/post hooks per tool call (logging, rate-limiting, ACL)
  • LifecycleHooks — per-turn and per-session callbacks
  • AgentState     — explicit state machine tracking the agent's phase
  • AgentHarness   — the event loop that wires everything together
"""
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable

import anthropic

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

client = anthropic.Anthropic()
MODEL = "claude-opus-4-8"


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------

class AgentState(Enum):
    IDLE              = auto()
    RUNNING           = auto()
    WAITING_FOR_TOOLS = auto()
    DONE              = auto()
    ERROR             = auto()


@dataclass
class AgentContext:
    session_id: str
    messages: list = field(default_factory=list)
    tool_calls: list = field(default_factory=list)   # audit trail
    state: AgentState = AgentState.IDLE
    metadata: dict = field(default_factory=dict)
    start_time: float = field(default_factory=time.monotonic)

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.start_time

    @property
    def turn_count(self) -> int:
        return sum(1 for m in self.messages if m.get("role") == "assistant")


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, tuple[dict, Callable]] = {}

    def register(self, schema: dict):
        """Decorator — pair a JSON schema with its implementation function."""
        def decorator(func: Callable) -> Callable:
            self._tools[schema["name"]] = (schema, func)
            return func
        return decorator

    def definitions(self) -> list[dict]:
        return [schema for schema, _ in self._tools.values()]

    def execute(self, name: str, input_data: dict) -> str:
        if name not in self._tools:
            return json.dumps({"error": f"Unknown tool: {name}"})
        _, func = self._tools[name]
        try:
            result = func(**input_data)
            return json.dumps(result) if not isinstance(result, str) else result
        except Exception as exc:
            log.exception("Tool %r raised an exception", name)
            return json.dumps({"error": str(exc)})


registry = ToolRegistry()


# ---------------------------------------------------------------------------
# Registered tools
# ---------------------------------------------------------------------------

@registry.register({
    "name": "read_file",
    "description": (
        "Read a file from the virtual filesystem. "
        "Call this to retrieve stored data before processing it."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Absolute virtual path, e.g. /data/users.json"},
        },
        "required": ["path"],
    },
})
def read_file(path: str) -> Any:
    vfs = {
        "/data/users.json": {
            "users": [
                {"id": 1, "name": "Alice", "role": "admin"},
                {"id": 2, "name": "Bob",   "role": "editor"},
                {"id": 3, "name": "Carol", "role": "viewer"},
            ]
        },
        "/data/config.json": {
            "model": MODEL,
            "max_retries": 3,
            "timeout_seconds": 30,
            "features": {"streaming": True, "caching": True},
        },
        "/logs/events.log": (
            "2026-06-30 09:00 INFO  Agent session started\n"
            "2026-06-30 09:01 INFO  Task assigned: summarise users\n"
            "2026-06-30 09:02 INFO  Task completed successfully\n"
        ),
    }
    if path in vfs:
        return {"path": path, "content": vfs[path]}
    return {"error": f"File not found: {path}"}


@registry.register({
    "name": "write_memory",
    "description": "Persist a key-value pair to agent memory for later retrieval in this session.",
    "input_schema": {
        "type": "object",
        "properties": {
            "key":   {"type": "string", "description": "Memory key"},
            "value": {"type": "string", "description": "Value to store"},
        },
        "required": ["key", "value"],
    },
})
def write_memory(key: str, value: str) -> dict:
    # In production: write to Redis, a database, or an embeddings store
    log.info("  [memory] stored key=%r len=%d", key, len(value))
    return {"ok": True, "key": key}


@registry.register({
    "name": "send_notification",
    "description": "Send a notification to an external channel (email, Slack, or SMS).",
    "input_schema": {
        "type": "object",
        "properties": {
            "channel": {
                "type": "string",
                "enum": ["email", "slack", "sms"],
                "description": "Delivery channel",
            },
            "message": {"type": "string", "description": "Notification body"},
        },
        "required": ["channel", "message"],
    },
})
def send_notification(channel: str, message: str) -> dict:
    log.info("  [notification → %s] %s", channel, message[:80])
    return {"sent": True, "channel": channel, "preview": message[:60]}


@registry.register({
    "name": "run_query",
    "description": "Run a named read-only query against the virtual database.",
    "input_schema": {
        "type": "object",
        "properties": {
            "query_name": {
                "type": "string",
                "enum": ["active_users", "recent_events", "system_stats"],
                "description": "Name of the predefined query to run",
            },
        },
        "required": ["query_name"],
    },
})
def run_query(query_name: str) -> dict:
    results = {
        "active_users": {"count": 42, "online_now": 7, "last_updated": "2026-07-01T08:00:00Z"},
        "recent_events": {"events": ["deploy@09:00", "alert@09:15", "resolved@09:20"], "total": 3},
        "system_stats": {"cpu_pct": 34, "mem_pct": 61, "disk_pct": 22, "uptime_days": 14},
    }
    return results.get(query_name, {"error": f"Unknown query: {query_name}"})


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

@dataclass
class MiddlewareDecision:
    allow: bool = True
    error_message: str | None = None


class Middleware:
    """Base — override pre_tool and/or post_tool."""
    def pre_tool(self, ctx: AgentContext, name: str, inp: dict) -> MiddlewareDecision:
        return MiddlewareDecision()

    def post_tool(self, ctx: AgentContext, name: str, output: str) -> str:
        return output


class LoggingMiddleware(Middleware):
    def pre_tool(self, ctx: AgentContext, name: str, inp: dict) -> MiddlewareDecision:
        log.info("[%s] → %s(%s)", ctx.session_id[:8], name, json.dumps(inp)[:80])
        return MiddlewareDecision()

    def post_tool(self, ctx: AgentContext, name: str, output: str) -> str:
        log.info("[%s] ← %s = %s", ctx.session_id[:8], name, output[:80])
        return output


class RateLimitMiddleware(Middleware):
    def __init__(self, max_calls: int = 20, window_seconds: float = 60.0):
        self._calls: list[float] = []
        self.max_calls = max_calls
        self.window = window_seconds

    def pre_tool(self, ctx: AgentContext, name: str, inp: dict) -> MiddlewareDecision:
        now = time.monotonic()
        self._calls = [t for t in self._calls if now - t < self.window]
        if len(self._calls) >= self.max_calls:
            log.warning("Rate limit hit for session %s", ctx.session_id[:8])
            return MiddlewareDecision(allow=False, error_message="Rate limit exceeded — slow down")
        self._calls.append(now)
        return MiddlewareDecision()


class ACLMiddleware(Middleware):
    """Block access to specific paths or resources."""
    BLOCKED_PATHS = frozenset({"/etc/passwd", "/etc/shadow", "/root/.ssh/id_rsa"})

    def pre_tool(self, ctx: AgentContext, name: str, inp: dict) -> MiddlewareDecision:
        if name == "read_file" and inp.get("path") in self.BLOCKED_PATHS:
            log.warning("ACL blocked read of %r in session %s", inp["path"], ctx.session_id[:8])
            return MiddlewareDecision(allow=False, error_message=f"Access denied: {inp['path']}")
        return MiddlewareDecision()


# ---------------------------------------------------------------------------
# Lifecycle hooks
# ---------------------------------------------------------------------------

class LifecycleHooks:
    def on_session_start(self, ctx: AgentContext) -> None:
        log.info("Session %s started", ctx.session_id[:8])

    def on_turn_start(self, ctx: AgentContext) -> None:
        log.debug("Turn %d starting (%.1fs elapsed)", ctx.turn_count + 1, ctx.elapsed)

    def on_turn_end(self, ctx: AgentContext, response: anthropic.types.Message) -> None:
        total_tokens = response.usage.input_tokens + response.usage.output_tokens
        log.info(
            "Turn %d done — %d tokens, stop_reason=%s",
            ctx.turn_count,
            total_tokens,
            response.stop_reason,
        )

    def on_tool_call(self, ctx: AgentContext, name: str) -> None:
        ctx.tool_calls.append({"tool": name, "at": time.monotonic() - ctx.start_time})

    def on_session_done(self, ctx: AgentContext, result: str) -> None:
        log.info(
            "Session %s complete — %.1fs, %d turns, %d tool calls",
            ctx.session_id[:8],
            ctx.elapsed,
            ctx.turn_count,
            len(ctx.tool_calls),
        )


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

class AgentHarness:
    def __init__(
        self,
        registry: ToolRegistry,
        middleware: list[Middleware] | None = None,
        hooks: LifecycleHooks | None = None,
        system_prompt: str = "You are a helpful assistant with access to tools.",
        max_turns: int = 20,
    ):
        self.registry = registry
        self.middleware = middleware or []
        self.hooks = hooks or LifecycleHooks()
        self.system_prompt = system_prompt
        self.max_turns = max_turns

    def _pre(self, ctx: AgentContext, name: str, inp: dict) -> tuple[bool, str | None]:
        for mw in self.middleware:
            decision = mw.pre_tool(ctx, name, inp)
            if not decision.allow:
                return False, decision.error_message
        return True, None

    def _post(self, ctx: AgentContext, name: str, output: str) -> str:
        for mw in self.middleware:
            output = mw.post_tool(ctx, name, output)
        return output

    def run(self, user_message: str, session_id: str | None = None) -> str:
        session_id = session_id or str(uuid.uuid4())
        ctx = AgentContext(session_id=session_id)
        ctx.messages = [{"role": "user", "content": user_message}]
        ctx.state = AgentState.RUNNING

        self.hooks.on_session_start(ctx)

        for _ in range(self.max_turns):
            self.hooks.on_turn_start(ctx)

            response = client.messages.create(
                model=MODEL,
                max_tokens=4096,
                system=self.system_prompt,
                tools=self.registry.definitions(),
                messages=ctx.messages,
            )

            self.hooks.on_turn_end(ctx, response)

            if response.stop_reason == "end_turn":
                ctx.state = AgentState.DONE
                result = next((b.text for b in response.content if b.type == "text"), "")
                self.hooks.on_session_done(ctx, result)
                return result

            if response.stop_reason != "tool_use":
                ctx.state = AgentState.ERROR
                return f"Unexpected stop_reason: {response.stop_reason}"

            ctx.state = AgentState.WAITING_FOR_TOOLS
            ctx.messages.append({"role": "assistant", "content": response.content})
            tool_results = []

            for block in response.content:
                if block.type != "tool_use":
                    continue

                self.hooks.on_tool_call(ctx, block.name)
                allowed, err = self._pre(ctx, block.name, block.input)

                if not allowed:
                    output = json.dumps({"error": err or "Blocked by middleware"})
                else:
                    output = self.registry.execute(block.name, block.input)
                    output = self._post(ctx, block.name, output)

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": output,
                })

            ctx.messages.append({"role": "user", "content": tool_results})
            ctx.state = AgentState.RUNNING

        ctx.state = AgentState.ERROR
        return "Max turns exceeded"


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    harness = AgentHarness(
        registry=registry,
        middleware=[
            LoggingMiddleware(),
            RateLimitMiddleware(max_calls=30, window_seconds=60.0),
            ACLMiddleware(),
        ],
        hooks=LifecycleHooks(),
        system_prompt=(
            "You are an operations agent. You have access to a virtual filesystem, "
            "a memory store, a notification system, and a database query interface. "
            "Complete tasks methodically using the available tools."
        ),
        max_turns=20,
    )

    tasks = [
        # Normal workflow: read → notify
        (
            "Read /data/users.json and /data/config.json, then send a Slack notification "
            "summarising the user count and the configured model name."
        ),
        # ACL test: blocked path falls back to allowed path
        (
            "Try to read /etc/passwd, and if that fails, read /data/config.json instead. "
            "Then run the system_stats query and summarise everything."
        ),
        # Multi-step with memory
        (
            "Run the active_users and recent_events queries, store a summary in memory "
            "under the key 'daily_report', then send an email notification with that summary."
        ),
    ]

    for task in tasks:
        print(f"\n{'='*60}")
        print(f"Task: {task[:80]}{'...' if len(task) > 80 else ''}")
        print("─" * 60)
        result = harness.run(task)
        print(f"\nResult:\n{result}")
