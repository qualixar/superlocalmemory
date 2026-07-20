# langchain-superlocalmemory

LangChain chat message history backed by the local data root of [SuperLocalMemory V3](https://github.com/qualixar/superlocalmemory).

This adapter writes chat messages to the configured SLM data root. Optional SLM providers, connectors, backup, and downloads have separate network behavior.

## Prerequisites

- Python 3.11+
- [SuperLocalMemory V3](https://github.com/qualixar/superlocalmemory) installed in the same Python environment
- `langchain-core >= 1.0.0`

## Installation

```bash
pip install langchain-superlocalmemory
```

Or install from source:

```bash
cd integrations/langchain
pip install -e .
```

## Quick Start

```python
from langchain_core.messages import AIMessage, HumanMessage
from langchain_superlocalmemory import SuperLocalMemoryChatMessageHistory

# Create a history for a conversation session
history = SuperLocalMemoryChatMessageHistory(session_id="my-chat-session")

# Add messages
history.add_messages([
    HumanMessage(content="What is SuperLocalMemory?"),
    AIMessage(content="It's a local-first memory system for AI assistants."),
])

# Retrieve messages (chronological order)
for msg in history.messages:
    print(f"{msg.type}: {msg.content}")

# Clear the session
history.clear()
```

## Features

- **Local-first storage** -- all data stays in `~/.superlocalmemory/memory.db`
- **Session isolation** -- each `session_id` is completely independent
- **Full LangChain compatibility** -- implements `BaseChatMessageHistory`
- **Persistent across restarts** -- SQLite-backed, survives process exit
- **Works alongside SLM** -- messages are queryable via CLI, MCP, Skills, and REST API
- **All message types** -- HumanMessage, AIMessage, SystemMessage, FunctionMessage, ToolMessage
- **additional_kwargs preserved** -- metadata round-trips through serialization

## Multi-Session Example

```python
from langchain_superlocalmemory import SuperLocalMemoryChatMessageHistory

# Two independent conversations
support = SuperLocalMemoryChatMessageHistory(session_id="support-ticket-42")
coding = SuperLocalMemoryChatMessageHistory(session_id="code-review-pr-99")

# Messages are isolated -- support session cannot see coding session
support.add_messages([HumanMessage(content="My app is crashing")])
coding.add_messages([HumanMessage(content="Review this PR please")])

assert len(support.messages) == 1
assert len(coding.messages) == 1
```

## Custom Database Path

By default the package uses `~/.superlocalmemory/memory.db`. You can point to a different database:

```python
history = SuperLocalMemoryChatMessageHistory(
    session_id="my-session",
    db_path="/path/to/custom/memory.db",
)
```

## How It Works

Each LangChain message is submitted through SuperLocalMemory V3's canonical
ingestion contract. The exact serialized message remains in the parent memory
row for lossless chat-history round trips, while SLM builds searchable facts:

- **Content**: JSON-serialized message (type, content, additional_kwargs)
- **Session isolation**: a SHA-256-namespaced session identifier
- **Tags**: `["langchain", "langchain:session:<session_id>"]` in metadata
- **Importance**: 3 (lower than user memories, so chat history does not crowd search results)
- **Project**: `"langchain"`

This means your LangChain conversations are visible in the SLM dashboard, searchable via `slm recall`, and accessible from any SLM-integrated tool.

## License

AGPL-3.0 -- see [LICENSE](../../../LICENSE) for details.

## Links

- [SuperLocalMemory V3 Repository](https://github.com/qualixar/superlocalmemory)
- [Documentation](https://superlocalmemory.com/)
- [LangChain Documentation](https://python.langchain.com/)
