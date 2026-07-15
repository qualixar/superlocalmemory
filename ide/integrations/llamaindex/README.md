# LlamaIndex Chat Store — SuperLocalMemory V3

A LlamaIndex `BaseChatStore` integration for [SuperLocalMemory V3](https://github.com/qualixar/superlocalmemory).

## Prerequisites

- Python 3.11+
- SuperLocalMemory installed in the same Python virtual environment

```bash
python -m pip install superlocalmemory
```

## Installation

```bash
pip install llama-index-storage-chat-store-superlocalmemory
```

## Quick Start

```python
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.storage.chat_store.superlocalmemory import SuperLocalMemoryChatStore

# Create the chat store (uses default SLM database)
chat_store = SuperLocalMemoryChatStore()

# Use with ChatMemoryBuffer for automatic conversation management
memory = ChatMemoryBuffer.from_defaults(
    chat_store=chat_store,
    chat_store_key="user-123",
    token_limit=3000,
)

# Or use directly for manual message management
chat_store.add_message("session-1", ChatMessage(role=MessageRole.USER, content="Hello!"))
chat_store.add_message("session-1", ChatMessage(role=MessageRole.ASSISTANT, content="Hi there!"))

messages = chat_store.get_messages("session-1")
print(messages)  # [ChatMessage(role=user, content="Hello!"), ChatMessage(role=assistant, content="Hi there!")]

# List all session keys
keys = chat_store.get_keys()

# Delete a session
chat_store.delete_messages("session-1")
```

## Features

- **Local data root** — Chat history is written to the configured SLM storage path
- **Explicit network boundary** — The adapter invokes the installed SLM runtime; configured SLM providers retain their documented network behavior
- **Shared runtime** — Documented SLM clients can access the same configured memory service when authorized
- **Session Isolation** — Each chat key is isolated with a namespaced SHA-256 session identifier
- **Persistent** — Survives process restarts (SQLite-backed, not in-memory)
- **Full BaseChatStore API** — `set_messages`, `get_messages`, `add_message`, `delete_messages`, `delete_message`, `delete_last_message`, `get_keys`
- **Async Support** — Async methods inherited from BaseChatStore (delegates to sync via `asyncio.to_thread`)

## How It Works

Each chat message is submitted through SuperLocalMemory V3's canonical ingestion
contract. The exact serialized payload remains available for chat-store round trips:
- **Content**: JSON-serialized `{role, content, additional_kwargs}`
- **Session**: `llamaindex:<sha256(session_key)>` for bounded, injection-safe isolation
- **Tag**: `li:chat:<hash>` in metadata for identification
- **Project**: `llamaindex` for easy identification
- **Importance**: 3 (low, since chat messages are transient)

## Custom Database Path

```python
# Use a custom database file
chat_store = SuperLocalMemoryChatStore(db_path="/path/to/custom/memory.db")
```

## Links

- [SuperLocalMemory V3](https://github.com/qualixar/superlocalmemory)
- [LlamaIndex Documentation](https://docs.llamaindex.ai/)
- [LlamaIndex Chat Stores Guide](https://docs.llamaindex.ai/en/stable/module_guides/storing/chat_stores/)
