# AI Agent Frameworks

Five working Python examples covering the core patterns for building AI agents with the Anthropic SDK. Each example is self-contained and runnable.

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your_key_here
```

---

## The Five Patterns

### How they relate

These patterns are not competing alternatives — they are layers. You can (and often should) combine them:

```
┌──────────────────────────────────────────────────────────┐
│  05  Harness Engineering  (orchestrates everything below) │
│  ┌────────────────────────────────────────────────────┐   │
│  │  04  A2A  (agents call other agents as sub-tasks)  │   │
│  │  ┌──────────────────────────────────────────────┐  │   │
│  │  │  02  Agent Skills  (modular tool library)    │  │   │
│  │  │  ┌──────────────────────────────────────┐   │  │   │
│  │  │  │  01  Function Calling  (one tool call)│   │  │   │
│  │  │  └──────────────────────────────────────┘   │  │   │
│  │  └──────────────────────────────────────────────┘  │   │
│  └────────────────────────────────────────────────────┘   │
│                                                            │
│  03  MCP  (standardised transport layer for tools)         │
└──────────────────────────────────────────────────────────┘
```

MCP is orthogonal — it is a wire protocol that can replace or supplement how any of the other patterns expose tools to Claude.

---

## 01 — Function Calling

**What it is:** The most fundamental pattern. You define tools as JSON schemas, pass them to Claude, and run a loop that executes whatever tools Claude requests until it produces a final text response.

**The core loop:**
1. Send a message + tool definitions to Claude
2. If `stop_reason == "tool_use"`, execute the requested tools and send results back
3. Repeat until `stop_reason == "end_turn"`

**When to use it:**
- You have a fixed, small set of tools
- You want complete control over tool execution (approval gates, logging, retries)
- You are building the foundation that the other patterns sit on

**Key file:** `01_function_calling/main.py`

```bash
python 01_function_calling/main.py
```

**What the example shows:** Three tools (weather lookup, calculator, web search) with stub implementations. Claude decides which tools to call and in what order — your code just executes them and keeps the loop going.

---

## 02 — Agent Skills / Plugins

**What it is:** A way to organise tools as a composable, registerable library rather than hardcoded JSON blobs. A `SkillRegistry` accepts decorated Python functions and converts them automatically into Claude tool definitions.

**The difference from plain function calling:** The *pattern* of the loop is identical — skills are still invoked via Claude's tool-use mechanism. What changes is *how you author and manage tools*. Skills are:
- Self-describing (the decorator carries the description and parameter schema)
- Independently testable (each skill is a plain Python function)
- Easy to add or remove without touching the agent loop

**When to use it:**
- You have many tools and want them organised in one place
- Different parts of the codebase own different capabilities
- You want to build a plugin system where tools can be registered at runtime

**Key file:** `02_agent_skills/main.py`

```bash
python 02_agent_skills/main.py
```

**What the example shows:** A `SkillRegistry` with five skills (unit conversion, text summarisation, text analysis, slugify, glossary lookup). Adding a new skill is a single decorated function — no changes to the agent loop.

---

## 03 — Model Context Protocol (MCP)

**What it is:** A standardised, language-agnostic *protocol* (not a pattern) for exposing tools and resources to LLMs. An MCP server runs as a separate process and advertises its capabilities; any MCP-compatible client can discover and invoke them without knowing how they are implemented.

**The difference from agent skills:** Skills are Python functions registered in the same process as the agent. MCP tools live in a separate server process and communicate over a defined protocol (stdio, HTTP, WebSockets). The separation means:
- The server can be written in any language
- The same server can be reused by multiple agents or even multiple LLM providers
- Tools can run with different permissions, in Docker, on a remote machine, etc.

**When to use it:**
- You want tools to be reusable across projects or teams
- Tools need to run in isolation (different environment, language, or security context)
- You are integrating with third-party MCP servers (databases, file systems, APIs)
- You are building tooling for Claude Desktop or other MCP-compatible hosts

**Key files:** `03_mcp/server.py` and `03_mcp/client.py`

```bash
# The client launches the server as a subprocess automatically
python 03_mcp/client.py
```

**What the example shows:** A `FastMCP` server exposing eight tools (math and string operations). The client connects via stdio, discovers the tools at runtime using the MCP protocol, and passes them to Claude via the async tool runner — Claude never knows (or cares) that the tools live in a different process.

---

## 04 — Agent-to-Agent (A2A)

**What it is:** A pattern where one agent (the orchestrator) delegates subtasks to other agents (specialists) by calling them as if they were tools. Each specialist is a separate Claude call with its own focused system prompt.

**The difference from the patterns above:** In the previous patterns, tools are deterministic functions — they run code and return data. In A2A, the "tool" is itself an LLM call. The orchestrator does not know *how* the specialist solves the problem; it only knows the input/output contract.

This enables:
- **Division of labour** — a researcher, a writer, and a critic each excel at their narrow job
- **Specialisation** — each agent has a system prompt tuned for its role
- **Parallelism** — independent sub-tasks can be dispatched concurrently
- **Separation of concerns** — the orchestrator reasons about *what* to do; specialists reason about *how*

**Core Benefits**: 
https://www.ibm.com/think/topics/agent2agent-protocol
- Privacy
    - The protocol treats agentic AI as opaque agents. This opacity means autonomous agents can collaborate without having to reveal their inner workings, such as internal memory, proprietary logic or particular tool implementations. This helps preserve data privacy and intellectual property.
- Seamless integration:
    - A2A is built on established standards e.g. HTTP, JSON-RPC and SSE
    - Easier for enterprises to adopt protocol and helps ensure compatibility with current tech stack.
- Security: 
    - It supports enterprise-grade authentication and authorization mechanisms and allows for secure information exchange.

**When to use it:**
- A task is too large or complex for a single prompt
- Different subtasks require meaningfully different expertise or personas
- You want to pipeline stages (research → draft → review)
- You need an agent that can spawn and coordinate other agents

**Key file:** `04_a2a/main.py`

```bash
python 04_a2a/main.py
```

**What the example shows:** An editorial pipeline with three specialist agents. The orchestrator exposes `delegate_research`, `delegate_writing`, and `delegate_review` as tools to itself. When Claude calls one of these "tools", the harness makes a real Claude API call to a specialist agent with a different system prompt and returns the result.

---

## 05 — Harness Engineering

**What it is:** The scaffolding that orchestrates everything else. A harness provides a reusable, production-ready event loop with cross-cutting concerns separated from business logic: state tracking, middleware, lifecycle hooks, error handling, and observability.

**The difference from the patterns above:** The other patterns answer *what the agent does*. The harness answers *how it is run*. You could swap the tools, the model, or even the LLM provider without changing the harness — and you could use the same harness across many different agents.

**Key components:**

| Component | Responsibility |
|---|---|
| `ToolRegistry` | Maps tool names to schemas + implementations via a decorator |
| `AgentState` | Explicit state machine: `IDLE → RUNNING → WAITING_FOR_TOOLS → DONE / ERROR` |
| `AgentContext` | Per-session data: message history, tool audit trail, elapsed time, metadata |
| `Middleware` | Pre/post hooks around every tool call (logging, rate limiting, ACL checks) |
| `LifecycleHooks` | Callbacks at session start/end and turn start/end (metrics, tracing) |
| `AgentHarness` | The event loop that wires all of the above together |

**When to use it:**
- You are building an agent that will run in production
- You need observability (logging, metrics, tracing) without polluting tool logic
- You want to enforce policies (rate limits, access control) across all tools uniformly
- You are running many concurrent agent sessions and need per-session state isolation

**Key file:** `05_harness/main.py`

```bash
python 05_harness/main.py
```

**What the example shows:** A harness with three middleware layers (logging, rate limiting, ACL path blocking) and lifecycle hooks. The three demo tasks exercise the happy path, an ACL block with fallback, and a multi-step workflow with memory persistence.

---

## Quick comparison

| | Function Calling | Agent Skills | MCP | A2A | Harness |
|---|---|---|---|---|---|
| **What it solves** | LLM ↔ code bridge | Tool organisation | Tool portability | Task decomposition | Operational concerns |
| **Tool location** | Same process | Same process | Separate process | Another LLM call | Any of the above |
| **Tool author** | You | You | Anyone | Another agent | You |
| **Complexity** | Low | Low | Medium | Medium | Medium–High |
| **Best for** | Starting point | Growing tool libraries | Cross-team / cross-language | Complex multi-stage tasks | Production deployments |

---

## Further reading

- [Anthropic tool use docs](https://docs.anthropic.com/en/docs/tool-use)
- [Model Context Protocol spec](https://modelcontextprotocol.io)
- [Anthropic Python SDK](https://github.com/anthropics/anthropic-sdk-python)
