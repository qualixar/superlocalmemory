# Framework Adapters

SuperLocalMemory V3.8.0 ships nine framework adapters. Each adapter implements
the memory or history interface that its target framework defines, writing data
through the SLM V3 ingestion contract so all records are visible across the
CLI, MCP, and dashboard surfaces.

All adapters write to the configured SLM data root. Optional SLM providers,
connectors, backup, and downloads retain their separately documented network
behavior.

## Adapter summary

| Framework | PyPI package | Interface | Minimum framework version |
|---|---|---|---|
| LangGraph | `langgraph-superlocalmemory` | `BaseStore` (long-term memory) | `langgraph >= 1.0.0` |
| Semantic Kernel | `semantic-kernel-superlocalmemory` | `VectorStoreCollection` | `semantic-kernel >= 1.34.0` |
| Microsoft Agent Framework | `agent-framework-superlocalmemory` | `ContextProvider` + `HistoryProvider` | `agent-framework-core >= 1.5.0` |
| LangChain | `langchain-superlocalmemory` | `BaseChatMessageHistory` | `langchain-core >= 1.0.0` |
| LlamaIndex | `llama-index-storage-chat-store-superlocalmemory` | `BaseChatStore` | Python 3.11+ |
| CrewAI | `crewai-superlocalmemory` | `StorageBackend` | `crewai >= 1.14.6` |
| AutoGen | `autogen-superlocalmemory` | `Memory` | `autogen-agentchat >= 0.7.5` |
| Google ADK | `google-adk-superlocalmemory` | `BaseMemoryService` | `google-adk >= 2.5.0` |
| OpenAI Agents | `openai-agents-superlocalmemory` | `SessionABC` | `openai-agents >= 0.18.3` |

All adapters require Python 3.11+ and SuperLocalMemory V3 installed in the
same virtual environment.

## Installation

Install the adapter alongside the target framework:

```bash
pip install superlocalmemory <adapter-package>
```

For example, to add LangGraph support:

```bash
pip install superlocalmemory langgraph-superlocalmemory
```

Source installs are available under `ide/integrations/<framework>/` in the
repository.

## Per-adapter quick reference

### LangGraph

Implements `BaseStore` (long-term memory), providing namespaced key-value
storage accessible to `create_react_agent` and `StateGraph`.

```python
from langgraph_superlocalmemory import SuperLocalMemoryStore

store = SuperLocalMemoryStore()
store.put(("users", "1"), "profile", {"name": "Ada", "role": "engineer"})
item = store.get(("users", "1"), "profile")
results = store.search(("users",), filter={"role": "engineer"})
```

Namespace prefix matching is element-wise: `("users",)` matches `("users", "1")`
but not `("users2",)`. TTL is not supported. Vector similarity search delegates
to SLM's native recall.

### Semantic Kernel

Implements `VectorStoreCollection` (SK 1.34+ preview API) for record
persistence and CRUD lifecycle operations.

```python
from semantic_kernel_superlocalmemory import SuperLocalMemoryVectorStore

store = SuperLocalMemoryVectorStore()
collection = store.get_collection(Doc, collection_name="docs")
await collection.ensure_collection_exists()
await collection.upsert(Doc(id="d1", text="hello world"))
```

Search performs field-filter retrieval; dense-vector ANN ranking is a
documented follow-up. Validate in CI against the installed SK version — the
vector-store API is still marked preview.

### Microsoft Agent Framework

Provides `SuperLocalMemoryContextProvider` and `SuperLocalMemoryHistoryProvider`
for the Agent Framework `before_run` / `after_run` hook model (GA, 1.5.0+).

```python
from agent_framework_superlocalmemory import (
    SuperLocalMemoryContextProvider,
    SuperLocalMemoryHistoryProvider,
)

history = SuperLocalMemoryHistoryProvider()
memory = SuperLocalMemoryContextProvider(max_recall=10)
# agent = ChatAgent(..., context_providers=[memory], history_provider=history)
```

`ContextProvider.before_run` injects recent session history; wider semantic
recall across the SLM store is a documented follow-up.

### LangChain

Implements `BaseChatMessageHistory`. Each session is isolated by a namespaced
SHA-256 session identifier; messages are tagged `langchain:session:<id>`.

```python
from langchain_superlocalmemory import SuperLocalMemoryChatMessageHistory

history = SuperLocalMemoryChatMessageHistory(session_id="my-chat-session")
history.add_messages([HumanMessage(content="What is SLM?"), ...])
for msg in history.messages:
    print(f"{msg.type}: {msg.content}")
history.clear()
```

All standard LangChain message types are supported. `additional_kwargs`
round-trip through serialization.

### LlamaIndex

Implements `BaseChatStore`. Integrates with `ChatMemoryBuffer` for automatic
conversation management.

```python
from llama_index.storage.chat_store.superlocalmemory import SuperLocalMemoryChatStore
from llama_index.core.memory import ChatMemoryBuffer

chat_store = SuperLocalMemoryChatStore()
memory = ChatMemoryBuffer.from_defaults(
    chat_store=chat_store, chat_store_key="user-123", token_limit=3000
)
```

Session keys are stored with a `llamaindex:<sha256(key)>` session identifier.

### CrewAI

Implements `StorageBackend` for scope-aware memory records. CrewAI supplies
pre-computed embeddings; the adapter persists them and ranks by cosine
similarity in pure Python (no numpy dependency).

```python
from crewai_superlocalmemory import SuperLocalMemoryBackend

backend = SuperLocalMemoryBackend()
```

Scope-prefix filtering uses hierarchical matching (`/project` matches
`/project/alpha`, never `/projectx`). `metadata_filter` supports equality
checks only. For large collections, use SLM's native recall for semantic
ranking.

### AutoGen

Implements `autogen_core.memory.Memory`. `query()` uses SLM's BM25 + semantic
recall pipeline, giving the agent access to the full SLM knowledge base, not
only AutoGen-added memories.

```python
from autogen_superlocalmemory import SuperLocalMemoryMemory

mem = SuperLocalMemoryMemory()
await mem.add(MemoryContent(content="Ada prefers dark mode.", mime_type=MemoryMimeType.TEXT))
result = await mem.query("Ada preferences")
```

`clear()` removes only `autogen-mem:` rows; personal and shared SLM memories
are unaffected.

> **Note:** For Microsoft Agent Framework (`agent-framework-core`) see the
> `agent_framework_superlocalmemory` adapter above; `autogen-agentchat` and
> Agent Framework are separate products.

### Google ADK

Implements `BaseMemoryService`. Persists ADK session events and serves
semantic recall with post-filtering by `app_name` + `user_id`.

```python
from google_adk_superlocalmemory import SuperLocalMemoryService

service = SuperLocalMemoryService()
runner = Runner(agent=agent, app_name="my-app", memory_service=service)
```

`add_session_to_memory` is idempotent: re-running for the same session
replaces previous events.

### OpenAI Agents

Implements `SessionABC` for ordered conversation history. Items are appended
with deterministic ordering (`created_at ASC, rowid ASC`).

```python
from openai_agents_superlocalmemory import SLMSession
from agents import Runner

session = SLMSession(session_id="user-42-conv-7")
runner = Runner(session=session)
result = await runner.run("What is the capital of France?")
```

Multiple sessions sharing the same database do not interfere; the prefix used
in SQL scans is escaped to prevent wildcard collisions.

## Common behaviors

All nine adapters:

- Write through the SLM V3 ingestion contract — records are immediately
  queryable via `slm recall`, the dashboard, and any MCP-connected client.
- Isolate sessions with a SHA-256-namespaced or scoped session identifier.
- Handle async via `asyncio.to_thread` over the synchronous SLM engine, so
  async agents are never blocked on memory I/O.
- Store data at the configured SLM data root (default
  `~/.superlocalmemory/memory.db`). Pass `db_path=` to override.

## Known limitations across adapters

| Limitation | Applies to |
|---|---|
| ANN vector search is not indexed; search uses SLM recall or in-process cosine | LangGraph, Semantic Kernel, LangChain, LlamaIndex, CrewAI, AutoGen |
| TTL not supported | LangGraph |
| `metadata_filter` equality-only | CrewAI |
| SK vector-store API is still marked preview | Semantic Kernel |
| `score` not exposed in `SearchMemoryResponse` | Google ADK |

---
*Part of [Qualixar](https://qualixar.com) | Created by [Varun Pratap Bhardwaj](https://varunpratap.com)*
