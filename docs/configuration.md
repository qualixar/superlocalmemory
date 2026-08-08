# Configuration
> SuperLocalMemory V4 Documentation
> https://superlocalmemory.com | Part of Qualixar

Control how SuperLocalMemory stores, retrieves, and processes your memories.

---

## Three Operating Modes

For the published LoCoMo result scopes behind Modes A and C, see
[Benchmark Evidence](benchmarks.md). Mode B has no separately published LoCoMo
run.

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
# Interactive (stored in config file — plaintext, atomic 0600)
slm provider set openai
# Prompts: Enter your OpenAI API key: sk-...
# File: ~/.superlocalmemory/config.json (0600; see core/config.py:SLMConfig.save)

# Via environment variable (takes precedence, avoids disk persistence)
export OPENAI_API_KEY="sk-..."
export SLM_CROSS_ENCODER_API_KEY="..."  # for remote reranker Bearer
```

- Interactive storage is **plaintext** protected only by an atomic `0600` write;
  env avoids writing the secret to disk.
- Keychain is **not** used for provider/reranker keys — those live in
  `config.json` (`0600`) or env. Keychain (`keyring` + fallback
  `~/.superlocalmemory/.credentials.json` `0600`) is for cloud-backup and
  ingest credentials (`infra/cloud_backup.py`, `ingestion/credentials.py`).

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

> **Recall result limit:** The default is 20 results per query (CLI: `slm recall --limit N`; MCP `recall` tool: `limit` parameter). There is no config file key for this — override it per-call with `--limit N`.

## Remote embedding and rerank endpoints

Both halves of the retrieval stack can be served by a remote OpenAI-compatible
endpoint. This is how a non-English deployment replaces the bundled models: the
default reranker, `cross-encoder/ms-marco-MiniLM-L-12-v2`, is English-only and
cannot score a Chinese, Japanese, or Arabic corpus meaningfully.

| | Config block | Keys | Route | Since |
|---|---|---|---|---|
| Embeddings | `embedding` | `provider: "openai"`, `api_endpoint`, `model_name`, `dimension` | `POST /v1/embeddings` | v3.4.24 (#16) |
| Reranking | `retrieval` | `cross_encoder_backend: "openai"`, `cross_encoder_endpoint`, `cross_encoder_model` | `POST /v1/rerank` | v3.8.12 (#105) |

```json
{
  "embedding": {
    "provider": "openai",
    "api_endpoint": "https://models.example.test/v1/embeddings",
    "model_name": "Qwen3-Embedding",
    "dimension": 1024
  },
  "retrieval": {
    "use_cross_encoder": true,
    "cross_encoder_backend": "openai",
    "cross_encoder_endpoint": "https://models.example.test/v1/rerank",
    "cross_encoder_model": "/root/model/reranker.gguf",
    "cross_encoder_timeout_seconds": 15.0
  }
}
```

### Reranker settings

| Setting | Default | Description |
|---------|---------|-------------|
| `use_cross_encoder` | `true` | Master switch for reranking |
| `cross_encoder_backend` | `""` | `""` / `"onnx"` run locally; `"openai"` / `"remote"` use `cross_encoder_endpoint` |
| `cross_encoder_endpoint` | `""` | Full or base rerank URL. `/rerank` is appended when absent. HTTPS is required off-host; HTTP is loopback-only; userinfo (`user:pass@`), query strings and fragments are rejected; redirects are not followed |
| `cross_encoder_model` | `cross-encoder/ms-marco-MiniLM-L-12-v2` | Local HF id, or the model name the endpoint serves |
| `cross_encoder_api_key` | `""` | Optional bearer token. Prefer `SLM_CROSS_ENCODER_API_KEY`; persisted config is owner-readable (`0600`) |
| `cross_encoder_timeout_seconds` | `15.0` | Per-request read budget for the remote endpoint |

Works with any Cohere-shaped `/v1/rerank` service — llama-server,
text-embeddings-inference, Infinity, vLLM — serving a multilingual reranker
such as `BAAI/bge-reranker-v2-m3`.

**Behaviour.** The remote path runs in the parent process: no reranker
subprocess, no machine-wide worker lock, and no 130 MB local model download
(`slm setup` and `slm doctor` both stop requiring it). If the endpoint is
unreachable, slow, or returns an unrecognised payload, SLM logs an error and
returns fusion-ranked results **without** reranking — it does not silently
substitute the local English model. Setting `cross_encoder_endpoint` while
`cross_encoder_backend` is a local value is reported as a configuration error
rather than ignored (issue #103).

**Privacy and network boundary — remote reranker
(`src/superlocalmemory/retrieval/remote_reranker.py`).** Remote reranking
sends the **recall query and every candidate's text** (`{model, query,
documents}`) to the configured `cross_encoder_endpoint`. SLM applies
`redact_secrets` + `redact_pii_text` as a **best-effort pre-transmission
filter** (recognized secrets/PII patterns, not a DLP guarantee). The endpoint
URL is validated: `http`/`https` only, must have a host, rejects embedded
userinfo (`user:password@`), rejects `?query` and `#fragment` (Bearer goes in
`Authorization`, not the URL), requires `https` for non-loopback hosts
(`http` allowed only for `localhost`/`127.0.0.1`/`::1` etc. via
`_is_loopback_host`), `follow_redirects=False` (3xx raises), transport + 5xx
retry once, error bodies are suppressed ("response body suppressed"), and
malformed value paths never log raw payloads. The Bearer token destination is
that endpoint; configure only a service you trust and keep reranking local
when memory text must not leave the machine.

**Remote embedding runtime.** `POST /v1/embeddings` in
`core/embeddings.py:_openai_compatible_embed_batch` sends raw `texts` as
`{model, input: texts}` with optional `Authorization: Bearer <api_key>`. No
remote-reranker URL hardening, no secret/PII pre-filter, and no scoped SSRF
claim applies. The `provider="openai"` token is the generic OpenAI-compatible
endpoint selector, not a claim of SSRF hardening.

## Environment Variables

These override config file settings when set:

| Variable | Purpose |
|----------|---------|
| `SLM_MODE` | Override operating mode |
| `SLM_PROFILE` | Override active profile |
| `SLM_DATA_DIR` | Override data directory (default: `~/.superlocalmemory/`) |
| `SLM_CROSS_ENCODER_API_KEY` | Runtime-only bearer token for a remote rerank endpoint; overrides any owner-only config value |
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

*SuperLocalMemory V4 — Copyright 2026 Varun Pratap Bhardwaj. AGPL-3.0-or-later. Part of Qualixar.*
