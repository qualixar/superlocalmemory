# API Reference
> SuperLocalMemory V3 Documentation
> https://superlocalmemory.com | Part of Qualixar

Use SuperLocalMemory programmatically from Python. This is for developers building integrations, custom workflows, or framework adapters.

---

## Installation

```bash
npm install -g superlocalmemory
```

Or, if you installed via npm, the Python package is available at:

```python
import sys
sys.path.insert(0, "/path/to/superlocalmemory/python")
```

## Quick Start

```python
from superlocalmemory.core.engine import MemoryEngine
from superlocalmemory.core.config import SLMConfig

# Initialize with default config (Mode A, default profile)
config = SLMConfig()
engine = MemoryEngine(config)

# Store a memory
engine.store("The API uses OAuth 2.0 with PKCE for mobile clients")

# Recall memories
results = engine.recall("authentication method for mobile")
for result in results:
    print(f"[{result.score:.2f}] {result.content}")

# Clean up
engine.close()
```

## Configuration

### SLMConfig

```python
from superlocalmemory.core.config import SLMConfig

# Default configuration
config = SLMConfig()

# Custom configuration
config = SLMConfig(
    mode="a",                      # "a", "b", or "c"
    profile="work",                # Active profile name
    data_dir="~/.superlocalmemory",  # Database location
    embedding_model="all-MiniLM-L6-v2",
    max_results=10,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mode` | str | `"a"` | Operating mode |
| `profile` | str | `"default"` | Active profile |
| `data_dir` | str | `"~/.superlocalmemory"` | Data directory |
| `embedding_model` | str | `"all-MiniLM-L6-v2"` | Sentence transformer model |
| `max_results` | int | `10` | Default max results for recall |

## Core Operations

### Store

```python
# Basic store
memory_id = engine.store("React 19 uses the new compiler by default")

# Store with tags
memory_id = engine.store(
    "Deploy to staging before 2pm on Fridays",
    tags=["process", "deployment"]
)

# Store with metadata
memory_id = engine.store(
    "Maria approved the new database schema",
    metadata={"source": "slack", "channel": "#backend"}
)
```

**Returns:** `int` — the ID of the stored memory.

### Recall

```python
# Basic recall
results = engine.recall("database schema approval")

# With limit
results = engine.recall("deployment schedule", limit=5)
```

**Returns:** `list[RecallResult]`

Each `RecallResult` has:

| Attribute | Type | Description |
|-----------|------|-------------|
| `id` | int | Memory ID |
| `content` | str | Memory text |
| `score` | float | Relevance score (0.0 to 1.0) |
| `timestamp` | datetime | When the memory was stored |
| `tags` | list[str] | Tags attached to the memory |
| `metadata` | dict | Additional metadata |

### Recall with Trace

```python
# Get channel-by-channel breakdown
results = engine.recall_trace("who approved the schema")

for result in results:
    print(f"[{result.score:.2f}] {result.content}")
    print(f"  Semantic: {result.trace.semantic:.2f}")
    print(f"  BM25:     {result.trace.bm25:.2f}")
    print(f"  Entity:   {result.trace.entity:.2f}")
    print(f"  Temporal:  {result.trace.temporal:.2f}")
```

### Delete

```python
# Delete by ID
engine.delete(memory_id=42)

# Delete by query (deletes best match)
engine.delete(query="old staging credentials")
```

### List Recent

```python
memories = engine.list_recent(limit=20)
for m in memories:
    print(f"[{m.id}] {m.content[:80]}...")
```

## Profile Management

```python
# List profiles
profiles = engine.list_profiles()

# Switch profile
engine.switch_profile("client-acme")

# Create profile
engine.create_profile("new-project")

# Delete profile
engine.delete_profile("old-project")
```

## Health and Diagnostics

```python
# System status
status = engine.status()
print(f"Mode: {status.mode}")
print(f"Profile: {status.profile}")
print(f"Memories: {status.memory_count}")

# Health check
health = engine.health()
print(f"Fisher-Rao: {health.fisher}")
print(f"Sheaf: {health.sheaf}")
print(f"Langevin: {health.langevin}")

# Consistency check
contradictions = engine.consistency_check()
for c in contradictions:
    print(f"Conflict: '{c.memory_a}' vs '{c.memory_b}'")
```

## Framework Integrations

### LangChain

```python
from superlocalmemory.integrations.langchain import SLMMemory

memory = SLMMemory(mode="a", profile="langchain-app")

# Use as LangChain memory
from langchain.chains import ConversationChain
from langchain.llms import OpenAI

chain = ConversationChain(
    llm=OpenAI(),
    memory=memory,
)
```

### LlamaIndex

```python
from superlocalmemory.integrations.llamaindex import SLMMemoryStore

store = SLMMemoryStore(mode="a", profile="llamaindex-app")

# Use as a LlamaIndex document store
```

### Direct MCP Integration

If you are building an MCP-compatible tool and want to add SLM as a backend:

```python
from superlocalmemory.mcp import create_mcp_server

server = create_mcp_server(
    mode="a",
    profile="my-tool",
)

# The server exposes all 24 tools and 6 resources
server.run()
```

## Context Manager

For scripts that need clean resource management:

```python
from superlocalmemory.core.engine import MemoryEngine
from superlocalmemory.core.config import SLMConfig

with MemoryEngine(SLMConfig()) as engine:
    engine.store("Ephemeral context for this script run")
    results = engine.recall("context")
    # engine.close() called automatically
```

## Error Handling

```python
from superlocalmemory.core.exceptions import (
    SLMError,            # Base exception
    ProfileNotFoundError,
    MemoryNotFoundError,
    ConfigError,
    DatabaseError,
)

try:
    engine.switch_profile("nonexistent")
except ProfileNotFoundError:
    print("Profile does not exist. Create it first.")

try:
    engine.delete(memory_id=99999)
except MemoryNotFoundError:
    print("Memory not found.")
```

## Thread Safety

The `MemoryEngine` uses SQLite WAL mode and is safe for concurrent reads from multiple threads. Writes are serialized at the database level.

For multi-threaded applications, create one engine instance and share it:

```python
# Safe: one engine, multiple threads reading
engine = MemoryEngine(SLMConfig())

# Each thread can call engine.recall() concurrently
# Writes (engine.store()) are automatically serialized
```

---

*SuperLocalMemory V3 — Copyright 2026 Varun Pratap Bhardwaj. AGPL-3.0-or-later. Part of Qualixar.*
