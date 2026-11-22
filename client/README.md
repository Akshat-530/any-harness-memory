# Harness integration (make memory core-injected)

The MCP server is pull-only. To make memory behave like part of the agent's
core — context injected automatically, capture automatic — wire these hooks in
your harness. `memory_client.py` is a ready Python client.

## Hooks

| Hook | Call | Why |
|---|---|---|
| `on_session_start` | `client.context()` (and `profile()`, `core()`) | inject standing context into the system prompt once |
| `on_before_model(msg)` | `client.recall(msg, top_k=3)` | RAG at the loop level; inject a small "relevant memory" block |
| `on_tool_error(err, fix)` | `client.capture(..., tool_events=[...])` | turn errors into reusable procedures |
| `on_after_turn(user, asst)` | `client.capture(user, asst)` | durable capture without a tool round-trip |
| `on_task_end(task, outcome)` | `client.task_end(task, outcome, lessons=[...])` | close the loop; feeds promotion/dream |
| `on_feedback` | `client.feedback(id, q, "useful"/"irrelevant")` | self-correcting ranking |

## Minimal example

```python
import asyncio
from memory_client import MemoryClient

async def main():
    mem = MemoryClient("http://127.0.0.1:4000/mcp")   # add token=... for auth
    system_extra = await mem.context()                 # once per session
    hits = await mem.recall("what does the user prefer?")
    # ... run your model ...
    await mem.capture("user said X", "I did Y")
    await mem.task_end("ship feature", "success", lessons=["use fnmatch"])

asyncio.run(main())
```

## Budgets
Keep pre-model injection small (~600 tokens). The server already budgets each
section; request `top_k=3` and bias by relevance, not volume. Cache the
session context — don't re-read `memory://context` every turn.

## Harnesses without code hooks
Paste `memory_protocol.md` into the system prompt and expose the tool profile to
the model. It will call the lifecycle tools itself.