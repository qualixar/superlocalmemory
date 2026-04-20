# Configuration
> SuperLocalMemory V3 Documentation
> https://superlocalmemory.com | Part of Qualixar

Control how SuperLocalMemory stores, retrieves, and processes your memories.

---

## Three Operating Modes

SuperLocalMemory runs in one of three modes. You pick the trade-off between privacy and power.

| Mode | What it does | Needs API key? | Data leaves your machine? |
|------|-------------|:--------------:|:-------------------------:|
| **A: Zero-Cloud** | Math-based retrieval. No LLM calls. | No | Never |
| **B: Local LLM** | Mode A + a local LLM via Ollama. | No | Never |
| **C: Cloud LLM** | Mode B + a cloud LLM for maximum recall quality. | Yes | Yes (queries only) |

### Check your current mode

```bash
slm mode
```

### Switch modes

```bash
slm mode a    # Zero-cloud (default)
slm mode b    # Local LLM
slm mode c    # Cloud LLM
```

Switching modes takes effect immediately. No data is lost.

### Mode A: Zero-Cloud (Default)

All operations run locally. Retrieval uses four channels (semantic similarity, keyword search, entity graph, and temporal context) combined with mathematical scoring. No network calls.

Best for: privacy-sensitive work, air-gapped environments, EU AI Act compliance.

### Mode B: Local LLM

Everything from Mode A, plus a local LLM (via Ollama) that improves recall by understanding query intent and reranking results.

**Setup:**

```bash
# Install Ollama (if not already installed)
brew install ollama          # macOS
curl -fsSL https://ollama.com/install.sh | sh  # Linux

# Pull a model
ollama pull llama3.2

# Switch to Mode B
slm mode b
```

Best for: developers who want better recall without sending data to the cloud.

### Mode C: Cloud LLM

Everything from Mode B, plus a cloud LLM for cross-encoder reranking and agentic multi-round retrieval. Highest recall quality.

**Setup:**

```bash
slm mode c
slm provider set openai
```

You will be prompted for your API key (stored locally in your config file, never transmitted except to the provider you choose).

Best for: maximum recall quality when privacy constraints allow cloud calls.

## Provider Configuration

Mode C supports multiple LLM providers.

### Set your provider

```bash
slm provider           # Show current provider
slm provider set       # Interactive provider selector
```

### Supported providers

| Provider | Command | Env variable |
|----------|---------|-------------|
| OpenAI | `slm provider set openai` | `OPENAI_API_KEY` |
| Anthropic | `slm provider set anthropic` | `ANTHROPIC_API_KEY` |
| Azure OpenAI | `slm provider set azure` | `AZURE_OPENAI_API_KEY` |
| Ollama (local) | `slm provider set ollama` | None needed |
| OpenRouter | `slm provider set openrouter` | `OPENROUTER_API_KEY` |

### Set API keys

You can set keys interactively or via environment variables:

```bash
# Interactive (stored in config file)
slm provider set openai
# Prompts: Enter your OpenAI API key: sk-...

# Via environment variable (takes precedence)
export OPENAI_API_KEY="sk-..."
```

## Config File

All settings live in:

```
~/.superlocalmemory/config.json
```

### Example config

```json
{
  "mode": "a",
  "profile": "default",
  "provider": {
    "name": "openai",
    "model": "gpt-4o-mini",
    "api_key_env": "OPENAI_API_KEY"
  },
  "auto_capture": true,
  "auto_recall": true,
  "embedding_model": "all-MiniLM-L6-v2",
  "max_recall_results": 10,
  "retention": {
    "default_policy": "indefinite"
  }
}
```

### Key settings

| Setting | Default | Description |
|---------|---------|-------------|
| `mode` | `"a"` | Operating mode: `a`, `b`, or `c` |
| `profile` | `"default"` | Active memory profile |
| `auto_capture` | `true` | Automatically store decisions and context |
| `auto_recall` | `true` | Automatically inject relevant memories |
| `embedding_model` | `"all-MiniLM-L6-v2"` | Sentence transformer for semantic search |
| `max_recall_results` | `10` | Maximum memories returned per query |

## Environment Variables

These override config file settings when set:

| Variable | Purpose |
|----------|---------|
| `SLM_MODE` | Override operating mode |
| `SLM_PROFILE` | Override active profile |
| `SLM_DATA_DIR` | Override data directory (default: `~/.superlocalmemory/`) |
| `OPENAI_API_KEY` | OpenAI API key for Mode C |
| `ANTHROPIC_API_KEY` | Anthropic API key for Mode C |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI API key for Mode C |
| `OPENROUTER_API_KEY` | OpenRouter API key for Mode C |

## Database Location

All data is stored locally in:

```
~/.superlocalmemory/memory.db    # SQLite database
~/.superlocalmemory/config.json  # Configuration
~/.superlocalmemory/backups/     # Automatic backups
```

To use a custom location:

```bash
export SLM_DATA_DIR="/path/to/your/data"
```

---

*SuperLocalMemory V3 — Copyright 2026 Varun Pratap Bhardwaj. AGPL-3.0-or-later. Part of Qualixar.*
