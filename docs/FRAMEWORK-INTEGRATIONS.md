# Framework Integrations

SuperLocalMemory V2 integrates with popular AI frameworks as a memory backend — 100% local, zero cloud dependencies.

---

## LangChain Integration

Use SuperLocalMemory as a chat message history store in LangChain applications.

### Installation

```bash
pip install langchain-superlocalmemory
```

### Basic Usage

```python
from langchain_superlocalmemory import SuperLocalMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

# Create chat history with session-based memory
history = SuperLocalMemoryChatMessageHistory(session_id="my-session")

# Build a conversational chain
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

chain = prompt | ChatOpenAI()

# Wrap with message history
chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: SuperLocalMemoryChatMessageHistory(session_id=session_id),
    input_messages_key="input",
    history_messages_key="history",
)

# Use the chain
response = chain_with_history.invoke(
    {"input": "What is AI?"},
    config={"configurable": {"session_id": "my-session"}}
)
```

### Advanced Features

**Session Isolation:**
```python
# Different sessions have isolated message histories
history_user1 = SuperLocalMemoryChatMessageHistory(session_id="user-1")
history_user2 = SuperLocalMemoryChatMessageHistory(session_id="user-2")
```

**Profile Support:**
```python
# Use different memory profiles for different contexts
history_work = SuperLocalMemoryChatMessageHistory(
    session_id="work-chat",
    profile="work"
)
history_personal = SuperLocalMemoryChatMessageHistory(
    session_id="personal-chat",
    profile="personal"
)
```

**Message Filtering:**
```python
# Retrieve messages with limits
recent_messages = history.get_messages(limit=10)

# Clear session history
history.clear()
```

### Storage Details

- Messages persist in `~/.claude-memory/memory.db`
- Each message stored as a memory with tags: `langchain`, `chat`, `session:<session_id>`
- Supports all LangChain message types (HumanMessage, AIMessage, SystemMessage)
- Automatic timestamp and metadata tracking

---

## LlamaIndex Integration

Use SuperLocalMemory as a chat store for LlamaIndex's memory system.

### Installation

```bash
pip install llama-index-storage-chat-store-superlocalmemory
```

### Basic Usage

```python
from llama_index.storage.chat_store.superlocalmemory import SuperLocalMemoryChatStore
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.llms.openai import OpenAI

# Create chat store
chat_store = SuperLocalMemoryChatStore()

# Create memory with chat store
memory = ChatMemoryBuffer.from_defaults(
    chat_store=chat_store,
    chat_store_key="user-1"
)

# Use with a chat engine
llm = OpenAI(model="gpt-4")
chat_engine = SimpleChatEngine.from_defaults(
    llm=llm,
    memory=memory
)

# Chat
response = chat_engine.chat("What is machine learning?")
print(response)
```

### Advanced Features

**Multiple Users:**
```python
# Separate memory for each user
memory_user1 = ChatMemoryBuffer.from_defaults(
    chat_store=chat_store,
    chat_store_key="user-1"
)
memory_user2 = ChatMemoryBuffer.from_defaults(
    chat_store=chat_store,
    chat_store_key="user-2"
)
```

**Profile Support:**
```python
# Use different profiles for different contexts
chat_store_work = SuperLocalMemoryChatStore(profile="work")
chat_store_personal = SuperLocalMemoryChatStore(profile="personal")

memory_work = ChatMemoryBuffer.from_defaults(
    chat_store=chat_store_work,
    chat_store_key="project-x"
)
```

**Message Management:**
```python
# Get messages for a specific chat
messages = chat_store.get_messages("user-1")

# Set messages
from llama_index.core.base.llms.types import ChatMessage
chat_store.set_messages(
    "user-1",
    [ChatMessage(role="user", content="Hello")]
)

# Delete messages
chat_store.delete_messages("user-1")

# List all chat keys
all_chats = chat_store.get_keys()
```

### Storage Details

- Messages persist in `~/.claude-memory/memory.db`
- Each message stored as a memory with tags: `llamaindex`, `chat`, `key:<chat_store_key>`
- Supports all LlamaIndex ChatMessage roles (user, assistant, system)
- Automatic timestamp tracking
- Full profile isolation support

---

## Why Use SuperLocalMemory with Frameworks?

| Benefit | Description |
|---------|-------------|
| **100% Local** | No cloud dependencies, all data stays on your machine |
| **Zero Configuration** | Works with default settings, no API keys needed |
| **Cross-Framework** | Same local database used by all frameworks and tools |
| **Profile Isolation** | Separate memories for work, personal, clients |
| **Persistent** | Memories survive across sessions and reboots |
| **Free Forever** | No usage limits, no subscriptions |

---

## Common Patterns

### Multi-Context Applications

```python
# LangChain for customer support
support_history = SuperLocalMemoryChatMessageHistory(
    session_id="customer-123",
    profile="customer-support"
)

# LlamaIndex for internal documentation
docs_store = SuperLocalMemoryChatStore(profile="internal-docs")
docs_memory = ChatMemoryBuffer.from_defaults(
    chat_store=docs_store,
    chat_store_key="team-wiki"
)
```

### Session Management

```python
# Create sessions with metadata
from langchain_core.messages import HumanMessage, AIMessage

history = SuperLocalMemoryChatMessageHistory(session_id="session-123")
history.add_user_message("What is Python?")
history.add_ai_message("Python is a high-level programming language...")

# Later, retrieve full conversation
messages = history.get_messages()
```

### Memory Cleanup

```python
# LangChain: Clear specific session
history.clear()

# LlamaIndex: Delete specific chat
chat_store.delete_messages("user-1")

# CLI: Reset entire profile
# superlocalmemoryv2-reset soft --profile customer-support
```

---

## Troubleshooting

### Import Errors

If you get import errors, ensure packages are installed:

```bash
# For LangChain
pip install langchain-superlocalmemory langchain-core

# For LlamaIndex
pip install llama-index-storage-chat-store-superlocalmemory llama-index-core
```

### Database Locked

If you see "database is locked" errors:

```bash
# Check if SuperLocalMemory is running correctly
superlocalmemoryv2-status

# Restart any MCP servers
# (Close and reopen Cursor/Windsurf)
```

### Profile Not Found

If a profile doesn't exist:

```bash
# List available profiles
superlocalmemoryv2-profile list

# Create the profile
superlocalmemoryv2-profile create work
```

---

## Learn More

- **[LangChain Wiki Guide](https://github.com/varun369/SuperLocalMemoryV2/wiki/LangChain-Integration)** — Full integration tutorial
- **[LlamaIndex Wiki Guide](https://github.com/varun369/SuperLocalMemoryV2/wiki/LlamaIndex-Integration)** — Complete setup guide
- **[API Reference](API-REFERENCE.md)** — Python API documentation
- **[Profiles Guide](PROFILES-GUIDE.md)** — Multi-context management

---

<p align="center">
  <strong>Built by <a href="https://github.com/varun369">Varun Pratap Bhardwaj</a></strong><br/>
  MIT License • <a href="https://superlocalmemory.com">superlocalmemory.com</a>
</p>
