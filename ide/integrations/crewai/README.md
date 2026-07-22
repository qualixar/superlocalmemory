# crewai-superlocalmemory

CrewAI [`StorageBackend`](https://docs.crewai.com/) (memory storage) backed by
the local data root of [SuperLocalMemory V3](https://github.com/qualixar/superlocalmemory).

Give a CrewAI agent or crew durable, scope-aware memory stored in your local
SLM data root and visible through every other SLM surface (CLI, MCP,
dashboard). Records stay in your local SLM data root; optional SLM providers,
connectors, backup, and downloads have separate network behavior.

## Prerequisites

- Python 3.11+
- [SuperLocalMemory V3](https://github.com/qualixar/superlocalmemory) installed in the same environment
- `crewai >= 1.14.6`

## Installation

```bash
pip install crewai-superlocalmemory
```

Or from source:

```bash
cd ide/integrations/crewai
pip install -e .
```

## Quick start

```python
from crewai_superlocalmemory import SuperLocalMemoryBackend

# Use as storage backend for CrewAI memory configuration.
backend = SuperLocalMemoryBackend()

# Direct usage (framework-free core):
from crewai_superlocalmemory._v3_crewai_store import V3CrewAIStore

store = V3CrewAIStore("~/.superlocalmemory/memory.db")
store.save({
    "id": "r1",
    "content": "Ada prefers dark mode.",
    "embedding": [0.1, 0.2, 0.3],
    "scope": "/project/alpha",
    "category": "contextual",
    "metadata": {},
})

results = store.search([0.1, 0.2, 0.3], scope_prefix="/project", limit=5)
for record, score in results:
    print(record["id"], score)

store.close()
```

## How it works

- Each `MemoryRecord` is one SLM memory with `session_id = "crewai-rec:{id}"`.
- The stored content is a descriptive JSON envelope so even a one-word record
  persists as a queryable SLM row (raw content below ~15 chars would otherwise
  extract no facts).
- CrewAI owns the embedder and supplies pre-computed embeddings. This adapter
  persists them and ranks candidates by cosine similarity in pure Python (no
  numpy dependency).
- Scope-prefix filtering uses hierarchical matching: `/project` matches
  `/project/alpha` but never `/projectx`.
- Async methods (`asave`, `asearch`, `adelete`) run the synchronous engine
  off the event loop via `asyncio.to_thread` so async agents are never blocked.

## Known limitations

- **ANN indexing** is not used. Search loads scope-filtered candidates (bounded
  at 10,000 rows) and ranks in Python. For large collections, use SLM's native
  recall surfaces for semantic ranking.
- **metadata_filter** supports equality checks only (`key == value`).

## License

AGPL-3.0 — see [LICENSE](../../../LICENSE).

## Links

- [SuperLocalMemory V3](https://github.com/qualixar/superlocalmemory)
- [Documentation](https://superlocalmemory.com/)
- [CrewAI](https://docs.crewai.com/)
