# autogen-superlocalmemory

AutoGen [`Memory`](https://microsoft.github.io/autogen/) backed by the local
data root of [SuperLocalMemory V3](https://github.com/qualixar/superlocalmemory).

Give an AutoGen agent durable memory stored in your local SLM data root and
visible through every other SLM surface (CLI, MCP, dashboard). Records stay in
your local SLM data root; optional SLM providers, connectors, backup, and
downloads have separate network behavior.

> **Note:** `autogen-agentchat` is under active maintenance. If you are
> building with **Microsoft Agent Framework** (`agent-framework-core`), see the
> shipped `agent_framework_superlocalmemory` adapter which implements the
> `HistoryProvider` / `ContextProvider` protocol.

## Prerequisites

- Python 3.11+
- [SuperLocalMemory V3](https://github.com/qualixar/superlocalmemory) installed in the same environment
- `autogen-agentchat >= 0.7.5` (pulls `autogen-core` as a dependency)

## Installation

```bash
pip install autogen-superlocalmemory
```

Or from source:

```bash
cd ide/integrations/autogen
pip install -e .
```

## Quick start

```python
from autogen_superlocalmemory import SuperLocalMemoryMemory
from autogen_core.memory import MemoryContent, MemoryMimeType

mem = SuperLocalMemoryMemory()

# Add memories
await mem.add(MemoryContent(content="Ada prefers dark mode.", mime_type=MemoryMimeType.TEXT))

# Query
result = await mem.query("Ada preferences")
for item in result.results:
    print(item.content)

# Use with an agent
from autogen_agentchat.agents import AssistantAgent
agent = AssistantAgent("assistant", memory=[mem], model_client=...)
```

## How it works

- Each `MemoryContent` is one SLM memory with a unique `session_id = "autogen-mem:{uuid}"`.
- The stored content is a descriptive JSON envelope so even a short string
  produces a queryable SLM row.
- `query()` uses SLM's `recall(fast=True)` text-search pipeline — the full
  SLM knowledge base enriches AutoGen's context, not just AutoGen-added memories.
- `update_context()` injects recent memories as a `SystemMessage` into the
  model context before each LLM call, mirroring AutoGen's `ListMemory` pattern.
- `clear()` deletes only `autogen-mem:` rows, leaving all other SLM memories
  (personal, shared, etc.) untouched.
- All async methods run the synchronous SLM engine via `asyncio.to_thread` so
  async agents are never blocked.

## Known limitations

- **Vector similarity search** is not exposed by this adapter; `query()` uses
  SLM's text-recall pipeline (BM25 + semantic) rather than ANN. For dense
  vector ranking, use SLM's own recall surfaces.
- `update_context()` injects recent memories rather than performing a
  query-specific retrieval; for retrieval-augmented context use `query()` then
  inject manually.

## License

AGPL-3.0 — see [LICENSE](../../../LICENSE).

## Links

- [SuperLocalMemory V3](https://github.com/qualixar/superlocalmemory)
- [Documentation](https://superlocalmemory.com/)
- [AutoGen](https://microsoft.github.io/autogen/)
- [Microsoft Agent Framework adapter](../agent-framework/)
