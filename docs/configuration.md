# Configuration
> SuperLocalMemory V3 Documentation
> https://superlocalmemory.com | Part of Qualixar

Control how SuperLocalMemory stores, retrieves, and processes your memories.

---

## Three Operating Modes

SuperLocalMemory runs in one of three modes. You pick the trade-off between privacy and power.

| Mode | What it does | Needs API key? | Data leaves your machine? |
|------|-------------|:--------------:|:-------------------------:|
| **A: Local** | Retrieval without a model-provider call in the core path. | No | Optional integrations may transmit data |
| **B: Local LLM** | Mode A + a local LLM via Ollama. | No | Depends on the Ollama endpoint and optional integrations |
| **C: Cloud LLM** | Mode B + configured cloud-provider enrichment and/or answer construction. | Yes | Configured query, ingestion, or enrichment content may be sent |

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

Core memory operations run against the local data root. Optional model and dependency downloads, connectors, backup, and other enabled integrations can use the network.

Best for: deployments that want a local core path and can govern optional integrations explicitly. Regulatory compliance still requires deployment-specific assessment.

### Mode B: Local LLM

Everything from Mode A, plus a local LLM (via Ollama) that improves recall by understanding query intent and reranking results.

**Setup:**

```bash
# Install Ollama using its reviewed package/instructions for your platform.
# macOS example:
brew install ollama

# Pull a model
ollama pull llama3.2

# Switch to Mode B
slm mode b
```

Best for: developers who can operate the selected local model and separately
govern optional networked integrations.

### Mode C: Cloud LLM

Everything from Mode B, plus a cloud LLM for cross-encoder reranking and agentic multi-round retrieval. Highest recall quality.

**Setup:**

```bash
slm mode c
slm provider set openai
```

You will be prompted for your API key (stored locally in your config file, never transmitted except to the provider you choose).

Best for: deployments that have approved the configured provider data path.

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

## Multi-Machine Mesh (v3.4.48+)

| Variable | Default | Description |
|---|---|---|
| `SLM_MESH_PEER_URL` | unset | Full URL of remote SLM instance (e.g., `http://192.168.1.100:8765`) |
| `SLM_MESH_SHARED_SECRET` | unset | Shared bearer token — same on both machines. Required when `SLM_MESH_HOST` is not localhost. |
| `SLM_MESH_HOST` | `127.0.0.1` | IP to bind this machine's mesh listener |
| `SLM_MESH_WS_PORT` | `7900` | Port used for mDNS service announcement |
| `SLM_MESH_DISCOVERY` | `on` | Set to `off` to disable mDNS auto-discovery |

See [Multi-Machine Setup](./multi-machine.md) for full setup guide.

---

## Optimize Configuration (v3.6)

SLM v3.6 adds the **Optimize** module — Cache + Compress + Align for LLM cost reduction. Configuration lives in a separate file at `~/.superlocalmemory/optimize.json` and hot-reloads within 2 seconds — no daemon restart required.

### Master Switches

| Setting | Default | Description |
|---------|---------|-------------|
| `optimize enabled` | `true` | Master ON/OFF |
| `cache enabled` | `true` | Cache lookups (exact match) |
| `semantic cache` | `false` | vCache semantic (opt-in) |
| `compression` | `safe` | `safe` (lossless) or `aggressive` (lossy prose allowed) |

### Quick Toggle via CLI

```bash
slm optimize on                     # Enable all
slm optimize off                    # Disable all
slm cache semantic on               # Enable semantic cache
slm compress mode aggressive        # Enable aggressive compression
```

### Config File Location

```bash
~/.superlocalmemory/optimize.json   # Written by UI, CLI, and API
```

### Hot-Reload

The daemon polls this file every 2 seconds. On change, all settings take effect without restart. Config version is auto-incremented for change tracking.

### Full Reference

See [docs/optimize-config.md](./optimize-config.md) for all 45+ config fields with defaults and descriptions.

---

*SuperLocalMemory V3 — Copyright 2026 Varun Pratap Bhardwaj. AGPL-3.0-or-later. Part of Qualixar.*
