# google-adk-superlocalmemory

Google ADK [`BaseMemoryService`](https://google.github.io/adk-docs/) backed by
the local data root of
[SuperLocalMemory V3](https://github.com/qualixar/superlocalmemory).

Give a Google ADK agent durable session memory stored in your local SLM data
root and visible through every other SLM surface (CLI, MCP, dashboard).
Records stay in your local SLM data root (optional SLM providers, connectors,
backup, and downloads have separate network behavior).

## Prerequisites

- Python 3.11+
- [SuperLocalMemory V3](https://github.com/qualixar/superlocalmemory) installed in the same environment
- `google-adk >= 2.5.0`

## Installation

```bash
pip install google-adk-superlocalmemory
```

Or from source:

```bash
cd ide/integrations/google-adk
pip install -e .
```

## Quick start

```python
from google.adk.agents import Agent
from google.adk.runners import Runner
from google_adk_superlocalmemory import SuperLocalMemoryService

# Attach SLM as the memory service for your runner
service = SuperLocalMemoryService()
runner = Runner(agent=agent, app_name="my-app", memory_service=service)

# After a session runs, its events are automatically stored and retrievable
# via SLM's CLI, MCP surface, and dashboard.
```

### Custom database path

```python
service = SuperLocalMemoryService(db_path="/path/to/my/memory.db")
```

## How it works

`BaseMemoryService` declares two abstract async methods:

- `add_session_to_memory(session)` — persists every event in the ADK session
  to SuperLocalMemory.  Each event is one SLM memory keyed by
  `adk:{app_name}:{user_id}:{session_id}:{event_index}`.  Calling this again
  for the same session replaces previous events (idempotent upsert).
- `search_memory(*, app_name, user_id, query)` — runs SLM's semantic recall
  pipeline and post-filters results to the target app+user namespace, returning
  a `SearchMemoryResponse`.

Session and event fields are read defensively (`getattr`) to stay compatible
across google-adk pydantic-model versions.

## Known limitations

- Dense-vector ANN ranking uses SuperLocalMemory's native recall pipeline;
  results are semantically ranked but `score` is not exposed in
  `SearchMemoryResponse`.
- Very short event text (< ~15 characters) may not produce atomic facts in
  SLM's ingestion pipeline; such events are still stored and returned by the
  DB fallback path in `search_memory`.

## License

AGPL-3.0 — see [LICENSE](../../../LICENSE).

## Links

- [SuperLocalMemory V3](https://github.com/qualixar/superlocalmemory)
- [Documentation](https://superlocalmemory.com/)
- [Google ADK](https://google.github.io/adk-docs/)
