# Framework Adapters
> SuperLocalMemory V3 Documentation
> https://superlocalmemory.com | Part of Qualixar

Nine Python packages back their framework's native memory interface with the local SLM data root. Install alongside the target framework; all data stays in the configured data root unless an optional SLM provider, connector, or backup feature is explicitly enabled.

**Prerequisites:** Python 3.11+, SuperLocalMemory V3 installed in the same environment.

---

## LangGraph

**Install:** `pip install langgraph-superlocalmemory`  
**Requires:** `langgraph >= 1.0.0`  
**Implements:** `BaseStore`  
**Class:** `SuperLocalMemoryStore`

```python
from langgraph_superlocalmemory import SuperLocalMemoryStore

store = SuperLocalMemoryStore()
store.put(("users", "1"), "profile", {"name": "Ada", "role": "engineer"})
item = store.get(("users", "1"), "profile")
results = list(store.search(("users",), filter={"role": "engineer"}))
store.delete(("users", "1"), "profile")
```

Drop in as the `store=` argument to `create_react_agent(...)` or `StateGraph(...).compile(store=store)`.

---

## Semantic Kernel

**Install:** `pip install semantic-kernel-superlocalmemory`  
**Requires:** `semantic-kernel >= 1.34.0`  
**Implements:** `VectorStore`; `get_collection` returns a `VectorStoreCollection`  
**Class:** `SuperLocalMemoryVectorStore`

```python
from semantic_kernel_superlocalmemory import SuperLocalMemoryVectorStore

store = SuperLocalMemoryVectorStore()
collection = store.get_collection(Doc, collection_name="docs")
await collection.ensure_collection_exists()
await collection.upsert(Doc(id="d1", text="hello world"))
doc = await collection.get("d1")
```

Written against Semantic Kernel 1.44. Uses the post-1.34 `semantic_kernel.data.vector` API; the deprecated `MemoryStoreBase` is not used.

---

## Microsoft Agent Framework

**Install:** `pip install agent-framework-superlocalmemory`  
**Requires:** `agent-framework-core >= 1.5.0`  
**Implements:** `ContextProvider` and `HistoryProvider`  
**Classes:** `SuperLocalMemoryContextProvider`, `SuperLocalMemoryHistoryProvider`

```python
from agent_framework_superlocalmemory import (
    SuperLocalMemoryContextProvider,
    SuperLocalMemoryHistoryProvider,
)

history = SuperLocalMemoryHistoryProvider()
memory = SuperLocalMemoryContextProvider(max_recall=10)
# agent = ChatAgent(..., context_providers=[memory], history_provider=history)
```

`HistoryProvider` implements `get_messages` / `save_messages` per session.  
`ContextProvider` overrides `before_run` (inject context) and `after_run` (persist turn).

---

## LangChain

**Install:** `pip install langchain-superlocalmemory`  
**Requires:** `langchain-core >= 1.0.0`  
**Implements:** `BaseChatMessageHistory`  
**Class:** `SuperLocalMemoryChatMessageHistory`

```python
from langchain_core.messages import AIMessage, HumanMessage
from langchain_superlocalmemory import SuperLocalMemoryChatMessageHistory

history = SuperLocalMemoryChatMessageHistory(session_id="my-chat-session")
history.add_messages([
    HumanMessage(content="What is SuperLocalMemory?"),
    AIMessage(content="A local-first memory system for AI assistants."),
])
messages = history.messages   # chronological list
history.clear()
```

---

## LlamaIndex

**Install:** `pip install llama-index-storage-chat-store-superlocalmemory`  
**Requires:** SuperLocalMemory V3 installed in the same virtual environment  
**Implements:** `BaseChatStore`  
**Class:** `SuperLocalMemoryChatStore`

```python
from llama_index.storage.chat_store.superlocalmemory import SuperLocalMemoryChatStore
from llama_index.core.memory import ChatMemoryBuffer

chat_store = SuperLocalMemoryChatStore()
memory = ChatMemoryBuffer.from_defaults(
    chat_store=chat_store, chat_store_key="user-123", token_limit=3000)
# Or directly:
chat_store.add_message("session-1", ChatMessage(role=MessageRole.USER, content="Hi"))
```

---

## CrewAI

**Install:** `pip install crewai-superlocalmemory`  
**Requires:** `crewai >= 1.14.6`  
**Implements:** `StorageBackend`  
**Class:** `SuperLocalMemoryBackend`

```python
from crewai_superlocalmemory import SuperLocalMemoryBackend

backend = SuperLocalMemoryBackend()
# Pass as storage_backend= in your CrewAI memory configuration.
```

Each `MemoryRecord` is stored as one SLM memory with `session_id = "crewai-rec:<id>"`. The backend supports scoped search and cosine-ranked retrieval over stored embeddings.

---

## AutoGen

**Install:** `pip install autogen-superlocalmemory`  
**Requires:** `autogen-agentchat >= 0.7.5`  
**Implements:** `Memory` (autogen-core)  
**Class:** `SuperLocalMemoryMemory`

```python
from autogen_superlocalmemory import SuperLocalMemoryMemory
from autogen_core.memory import MemoryContent, MemoryMimeType

mem = SuperLocalMemoryMemory()
await mem.add(MemoryContent(content="Ada prefers dark mode.", mime_type=MemoryMimeType.TEXT))
result = await mem.query("Ada preferences")
# agent = AssistantAgent("assistant", memory=[mem], model_client=...)
```

> For projects using **Microsoft Agent Framework** (`agent-framework-core`), prefer the `agent-framework-superlocalmemory` adapter instead.

---

## Google ADK

**Install:** `pip install google-adk-superlocalmemory`  
**Requires:** `google-adk >= 2.5.0`  
**Implements:** `BaseMemoryService`  
**Class:** `SuperLocalMemoryService`

```python
from google.adk.agents import Agent
from google.adk.runners import Runner
from google_adk_superlocalmemory import SuperLocalMemoryService

service = SuperLocalMemoryService()
runner = Runner(agent=agent, app_name="my-app", memory_service=service)
# After a session runs, events are stored and retrievable via SLM CLI, MCP, and dashboard.
```

Implements `add_session_to_memory` (idempotent upsert per session) and `search_memory` (semantic recall). Custom database path: `SuperLocalMemoryService(db_path="/path/to/memory.db")`.

---

## OpenAI Agents

**Install:** `pip install openai-agents-superlocalmemory`  
**Requires:** `openai-agents >= 0.18.3`  
**Implements:** `SessionABC`  
**Class:** `SLMSession`

```python
from agents import Runner
from openai_agents_superlocalmemory import SLMSession

session = SLMSession(session_id="user-42-conv-7")
runner = Runner(session=session)
result = await runner.run("What is the capital of France?")
```

Manages the ordered `TResponseInputItem` conversation history for one `session_id`. Custom path: `SLMSession(session_id="s1", db_path="/path/to/memory.db")`.

---

## All Adapters at a Glance

| Framework | Install name | Interface | Class |
|-----------|-------------|-----------|-------|
| LangGraph | `langgraph-superlocalmemory` | `BaseStore` | `SuperLocalMemoryStore` |
| Semantic Kernel | `semantic-kernel-superlocalmemory` | `VectorStore` | `SuperLocalMemoryVectorStore` |
| Microsoft Agent Framework | `agent-framework-superlocalmemory` | `ContextProvider` / `HistoryProvider` | `SuperLocalMemoryContextProvider`, `SuperLocalMemoryHistoryProvider` |
| LangChain | `langchain-superlocalmemory` | `BaseChatMessageHistory` | `SuperLocalMemoryChatMessageHistory` |
| LlamaIndex | `llama-index-storage-chat-store-superlocalmemory` | `BaseChatStore` | `SuperLocalMemoryChatStore` |
| CrewAI | `crewai-superlocalmemory` | `StorageBackend` | `SuperLocalMemoryBackend` |
| AutoGen | `autogen-superlocalmemory` | `Memory` | `SuperLocalMemoryMemory` |
| Google ADK | `google-adk-superlocalmemory` | `BaseMemoryService` | `SuperLocalMemoryService` |
| OpenAI Agents | `openai-agents-superlocalmemory` | `SessionABC` | `SLMSession` |

Source is under `ide/integrations/<framework>/` in the repository. Each adapter has its own README with extended usage examples, prerequisites, and scope notes.

---

*SuperLocalMemory V3 — Copyright 2026 Varun Pratap Bhardwaj. AGPL-3.0-or-later. Part of Qualixar.*
