# openai-agents-superlocalmemory

OpenAI Agents SDK
[`SessionABC`](https://openai.github.io/openai-agents-python/) (conversation
history store) backed by the local data root of
[SuperLocalMemory V3](https://github.com/qualixar/superlocalmemory).

Give an OpenAI Agents SDK runner durable, ordered conversation history stored
in your local SLM data root and visible through every other SLM surface (CLI,
MCP, dashboard).  Records stay in your local SLM data root (optional SLM
providers, connectors, backup, and downloads have separate network behavior).

## Prerequisites

- Python 3.11+
- [SuperLocalMemory V3](https://github.com/qualixar/superlocalmemory) installed in the same environment
- `openai-agents >= 0.18.3`

## Installation

```bash
pip install openai-agents-superlocalmemory
```

Or from source:

```bash
cd ide/integrations/openai-agents
pip install -e .
```

## Quick start

```python
from agents import Runner
from openai_agents_superlocalmemory import SLMSession

# Create a session backed by the local SLM database
session = SLMSession(session_id="user-42-conv-7")

runner = Runner(session=session)
result = await runner.run("What is the capital of France?")
```

### Custom database path

```python
session = SLMSession(
    session_id="my-session",
    db_path="/path/to/my/memory.db",
    session_settings={"model": "gpt-4o"},
)
```

## How it works

`SessionABC` manages the ordered conversation history (the list of
`TResponseInputItem` dicts) for one `session_id`.  `SLMSession` implements
four async methods:

- `add_items(items)` — appends items to the end of the conversation history.
  Each item is stored as a separate SLM memory keyed by
  `openai-agents:{session_id}:{uuid4}`.  Concurrent appends from different
  threads never collide.
- `get_items(limit=None)` — returns stored items oldest-first; when *limit*
  is given, returns the last *limit* items.  Ordering uses
  `(created_at ASC, rowid ASC)` for deterministic stability.
- `pop_item()` — removes and returns the most-recent item (or `None` if the
  session is empty).  Uses `(created_at DESC, rowid DESC)` for determinism.
- `clear_session()` — permanently removes all items for this session.

All async methods wrap the synchronous SLM engine via `asyncio.to_thread` so
the event loop is never blocked.

## Session isolation

Each `SLMSession` instance is scoped to its `session_id`.  Multiple sessions
sharing the same database do not interfere with each other, and the prefix
used in SQL LIKE scans is escaped to prevent wildcard collisions.

## License

AGPL-3.0 — see [LICENSE](../../../LICENSE).

## Links

- [SuperLocalMemory V3](https://github.com/qualixar/superlocalmemory)
- [Documentation](https://superlocalmemory.com/)
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/)
