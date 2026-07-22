# langgraph-superlocalmemory

LangGraph [`BaseStore`](https://langchain-ai.github.io/langgraph/) (long-term
memory) backed by the local data root of
[SuperLocalMemory V3](https://github.com/qualixar/superlocalmemory).

Give a LangGraph agent durable, namespaced long-term memory stored in your
local SLM data root and visible through every other SLM surface (CLI, MCP,
dashboard). Optional SLM providers, connectors, backup, and downloads have
separate network behavior.

## Prerequisites

- Python 3.11+
- [SuperLocalMemory V3](https://github.com/qualixar/superlocalmemory) installed in the same environment
- `langgraph >= 1.0.0`

## Installation

```bash
pip install langgraph-superlocalmemory
```

Or from source:

```bash
cd ide/integrations/langgraph
pip install -e .
```

## Quick start

```python
from langgraph_superlocalmemory import SuperLocalMemoryStore

store = SuperLocalMemoryStore()

# Namespaced key-value writes
store.put(("users", "1"), "profile", {"name": "Ada", "role": "engineer"})
store.put(("users", "1"), "prefs", {"theme": "dark"})

# Read one item
item = store.get(("users", "1"), "profile")
print(item.value)          # {'name': 'Ada', 'role': 'engineer'}
print(item.created_at)     # datetime, preserved across updates

# Prefix search + filter
for hit in store.search(("users",), filter={"role": "engineer"}):
    print(hit.namespace, hit.key, hit.value)

# List namespaces
print(store.list_namespaces())          # [('users', '1'), ...]
print(store.list_namespaces(max_depth=1))  # [('users',)]

store.delete(("users", "1"), "prefs")
```

Use it as any LangGraph store, including with `create_react_agent(..., store=store)`
and `StateGraph(...).compile(store=store)`.

## How it works

`BaseStore` declares only `batch()` and `abatch()` as abstract; every
convenience method (`get`, `put`, `search`, `delete`, `list_namespaces` and the
async variants) is inherited and delegates to them. This adapter implements
those two methods and maps each op onto SuperLocalMemory:

- Each item is one SLM memory whose `session_id` encodes the namespace + key.
- The stored content is a JSON envelope `{namespace, key, value, created_at,
  updated_at}`; `created_at` is preserved across updates, `updated_at` refreshed.
- Prefix search and `list_namespaces` compare the real namespace **tuple**
  element-wise, so `("users",)` matches `("users", "1")` but never `("users2",)`.
- `abatch()` runs the synchronous engine off the event loop via a worker
  thread, so async graphs are never blocked.

## Known limitations

- **Vector search** (`SearchOp.query`) is not indexed; `search()` performs
  namespace-prefix + `filter` matching and returns `score=None`. Use SLM's own
  recall surfaces for semantic ranking.
- **Wildcard namespace paths** (`"*"` segments in `list_namespaces`
  match conditions) are treated literally in this release.
- **TTL** is not supported (`supports_ttl = False`).

## License

AGPL-3.0 — see [LICENSE](../../../LICENSE).

## Links

- [SuperLocalMemory V3](https://github.com/qualixar/superlocalmemory)
- [Documentation](https://superlocalmemory.com/)
- [LangGraph](https://langchain-ai.github.io/langgraph/)
