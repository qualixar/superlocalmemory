# FAQ

Frequently asked questions about SuperLocalMemory V3.

## General

### What is SuperLocalMemory?

SuperLocalMemory is a persistent memory system for AI assistants. It stores your decisions, bug fixes, project context, and preferences locally, then automatically provides them to your AI in future sessions. Your AI stops forgetting you.

### Is it really free?

Yes. SuperLocalMemory is open-source (GNU Affero General Public License v3.0 or later) and completely free. No usage limits, no credit system, no subscription. Forever.

### Where is my data stored?

Core memory is SQLite-backed inside the configured SLM data root. That root also contains configuration, logs, queues, models, and derived state. Mode C sends configured query or enrichment content to its provider; optional connectors, backup, and downloads have their own network behavior in every mode.

### Which IDEs are supported?

Run `slm connect --list` for the release's documented client names. MCP-compatible clients can also be configured manually, but a client is considered verified only when it passes the release integration matrix.

### Does it work offline?

Mode A and Mode B work fully offline. Mode C requires internet for the cloud LLM.

## Installation

### What are the requirements?

- **Python** 3.11+ (required for V3 engine)
- **Node.js** 14+ (if installing via npm)
- Any supported IDE
- For Mode B: Ollama with a pulled model
- For Mode C: API key for your cloud LLM provider

### How do I install it?

```bash
# npm (recommended)
npm install -g superlocalmemory
slm setup
slm warmup    # Optional — pre-download embedding model

# or pip
pip install superlocalmemory
slm setup
```

### How do I update?

```bash
npm install -g superlocalmemory@latest
# or: pip install --upgrade superlocalmemory
```

### I am upgrading from V2. Will I lose my data?

No. Run `slm migrate` after updating. All memories, profiles, and settings are preserved. A backup is created automatically. See [Migration from V2](Migration-from-V2) for details.

## Usage

### How does auto-recall work?

When you start a conversation in your IDE, SuperLocalMemory automatically retrieves relevant memories and injects them into your AI's context. You do not need to call "recall" explicitly — it happens in the background via the MCP server.

### How do I store a memory?

```bash
slm remember "The deploy script needs AWS_REGION set to us-east-1"
```

### What do queryable, enriching, complete, and failed mean?

- `queryable` means raw evidence and the SQLite relational/FTS projection are durable and recallable.
- `enriching` means a lease-owning worker is running configured derivation stages.
- `complete` means every declared derivation and configured projector succeeded.
- `failed` retains the raw evidence, error, attempt count, and retry timing; it is not silent data loss.

### How do I search memories?

```bash
slm recall "deploy configuration"
```

### How do I see which retrieval channels found what?

```bash
slm trace "deploy configuration"
```

This shows per-channel scores (Semantic, BM25, Entity Graph, Temporal) for each result.

### How do I delete a memory?

```bash
slm forget "search query"     # Delete matching memories (with confirmation)
```

## Modes

### Which mode should I use?

- **Mode A** if you need privacy, compliance, or offline operation
- **Mode B** if you want composed answers and have a capable machine (16GB+ RAM)
- **Mode C** if you want maximum accuracy and cloud access is acceptable

### Can I switch modes after setup?

Yes: `slm mode a`, `slm mode b`, or `slm mode c`. Your memories are shared across all modes.

### What are the accuracy differences?

The papers contain versioned LoCoMo experiments, not current-release guarantees:
- Historical V3 research result: **74.8%** used local retrieval with GPT-4.1-mini answer construction.
- Historical Mode C result: **87.7%** on **81 questions from one conversation** with cloud-assisted components.

The current V3.7 result is unknown. See the linked preprints for their original protocols and limitations.

## Privacy and Security

### Can anyone else see my memories?

No. Your database is a local file on your machine. It is not synced, uploaded, or shared with anyone — including us.

### Does it guarantee regulatory compliance?

No software package certifies the complete deployment. SLM supplies local storage, erasure, provenance, retention, access-policy, and audit controls; applicability and sufficiency depend on the operator, use case, configuration, providers, and surrounding systems.

### Can I export my data?

The database is a standard SQLite file at `~/.superlocalmemory/memory.db`. You can copy it, back it up, or query it directly.

### Can I delete all my data?

`slm forget "query"` deletes matching memories. To delete everything, remove the database: `rm ~/.superlocalmemory/memory.db`.

## Troubleshooting

### My AI does not seem to remember anything.

1. Check that SuperLocalMemory is running: `slm status`
2. Check that you have stored memories: `slm recall "test"`
3. Verify your IDE connection: restart the IDE after configuring MCP
4. Check the active profile: `slm profile list`

### Recall returns irrelevant results.

Try more specific queries. Use `slm trace "query"` to see which channels contribute — this helps diagnose whether the issue is semantic, keyword, or entity matching.

### The setup wizard does not detect my IDE.

Use manual configuration. See [IDE Setup](IDE-Setup) for per-IDE config paths.

### Where can I report bugs?

Open an issue at [github.com/qualixar/superlocalmemory/issues](https://github.com/qualixar/superlocalmemory/issues).

---
*Part of [Qualixar](https://qualixar.com) | Created by [Varun Pratap Bhardwaj](https://varunpratap.com)*
