# FAQ — V4.0.0

Frequently asked questions about SuperLocalMemory V4.0.0.

## General

### What is SuperLocalMemory?

SuperLocalMemory is a persistent memory system for AI assistants. It stores your decisions, bug fixes, project context, and preferences locally, then automatically provides them to your AI in future sessions via 5-channel retrieval (semantic, BM25, temporal, Hopfield, spreading activation — plus graph enhancement). Your AI stops forgetting you.

### Is it really free?

Yes. SuperLocalMemory is open-source (GNU Affero General Public License v3.0 or later) and completely free. No usage limits, no credit system, no subscription. Forever.

### Where is my data stored?

Core memory is SQLite-backed inside the configured SLM data root. That root also contains configuration, logs, queues, models, and derived state. Mode C sends configured query or enrichment content to its provider; optional connectors, backup, and downloads have their own network behavior in every mode.

### Which IDEs and platforms are supported?

Run `slm connect --list` for the release's documented client names. MCP-compatible clients can also be configured manually, but a client is considered verified only when it passes the release integration matrix. V4 platform boundary: **Apple Silicon macOS, 64-bit Windows, 64-bit Linux** — Intel Mac and 32-bit Windows are not supported (`cryptography==50.0.0`).

### Does it work offline?

Mode A and Mode B work fully offline. Mode C requires internet for the cloud LLM. Optional connectors, backup, and model downloads require network in any mode when explicitly enabled.

## Installation

### What are the requirements?

- **Python** 3.11 – 3.14 (required for V4 engine)
- **Node.js** 18+ (if installing via npm)
- **Platform:** Apple Silicon macOS, 64-bit Windows, or 64-bit Linux (Intel Mac / Win32 not supported)
- Any supported IDE
- For Mode B: Ollama with a pulled model
- For Mode C: API key for your cloud LLM provider

### How do I install it?

```bash
# npm (recommended)
npm install -g superlocalmemory
slm setup
slm warmup    # Optional — pre-download embedding model

# or inside an activated Python virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install superlocalmemory
slm setup
```

See [Installation](Installation) for platform notes and troubleshooting.

### How do I update?

```bash
npm install -g superlocalmemory@latest
# or, while the SLM virtual environment is active:
python -m pip install --upgrade superlocalmemory
# then:
slm restart && slm doctor
```

### I am upgrading from V2. Will I lose my data?

No. Run `slm migrate` (V2→V3 data migration) after updating — or `slm db migrate` for additive schema maintenance; they are different commands (see [CLI Reference](CLI-Reference)). All memories, profiles, and settings are preserved. A backup is created automatically. See [Migration from V2](Migration-from-V2) for details.

## Usage

### How does auto-recall work?

When you start a conversation in your IDE, SuperLocalMemory automatically retrieves relevant memories and injects them into your AI's context via the MCP server (`slm mcp` or HTTP `http://127.0.0.1:8765/mcp/`). You do not need to call "recall" explicitly — it happens according to the client's configured instructions/hooks.

### How do I store a memory?

```bash
slm remember "The deploy script needs AWS_REGION set to us-east-1"
slm remember "Decision" --scope shared --shared-with team-a
```

### What do queryable, enriching, complete, and failed mean?

- `queryable` means raw evidence and the SQLite relational/FTS projection are durable and recallable.
- `enriching` means a lease-owning worker is running configured derivation stages.
- `complete` means every declared derivation and configured projector succeeded (and a hash-verifiable manifest is sealed).
- `failed` retains the raw evidence, error, attempt count, and retry timing; it is not silent data loss (see `slm ops list` / `list_failed_operations`).

### How do I search memories?

```bash
slm recall "deploy configuration"
```

Current recall uses five candidate producers (semantic, BM25, temporal, Hopfield, spreading activation) plus entity-graph score enhancement.

### How do I see which retrieval channels found what?

```bash
slm trace "deploy configuration"
```

This shows per-channel scores (Semantic, BM25, Temporal, Hopfield, Spreading Activation) for each result. Entity-graph data can enhance a post-fusion score but is not a separate recall channel. The current implementation is **five producers**, not four — earlier docs that said four-channel are obsolete.

### How do I delete a memory?

```bash
slm forget "search query"     # Delete matching memories (with confirmation)
slm delete <fact_id> --yes    # Delete one fact by ID (use slm list to find IDs)
```

Use `slm ops list` or `list_failed_operations` (MCP, `power`/`whole` profile) to inspect stuck operations.

## Modes

### Which mode should I use?

- **Mode A** if you need privacy, compliance, or offline operation
- **Mode B** if you want composed answers and have a capable machine (16GB+ RAM)
- **Mode C** if you want maximum accuracy and cloud access is acceptable

### Can I switch modes after setup?

Yes: `slm mode a`, `slm mode b`, or `slm mode c`. Your memories are shared across all modes.

### What are the accuracy differences?

The V3 paper provides **published LoCoMo evidence carried into V4 (not a newly rerun V4 benchmark)** — `arXiv:2603.14588`:

- **60.4%** Mode A Raw across 10 conversations / 1,276 questions with zero-LLM answer construction.
- **74.8%** Mode A Retrieval across the same scope with local retrieval and GPT-4.1-mini answer synthesis.
- **87.7%** Mode C on Conv-30 / 81 questions with cloud embeddings and GPT-4.1-mini answer generation and judge.

The figures retain their original protocol scope; they are not a newly rerun V4 package benchmark and are not comparable across vendors without matching protocol (conversation scope, question count, retrieval stack, answer model, judge, release artifact). See the linked preprint for category results, ablations, and limitations. The non-rerun nature is explicit in [README](https://github.com/qualixar/superlocalmemory/blob/main/README.md#published-locomo-evidence-v3-architecture-carried-into-v4) and `docs/benchmarks.md`.

## V4 Reliability Contract

### What is actually verified with 2,200/2,200?

Only one stress figure is verified in V4.0.0: **2,200/2,200 trials (100%)** from `benchmark/run_all.py --trials 200` (11 experiments × 200). Source: `benchmark/results/SUMMARY.md` (4.0.0, Python 3.13.13, macOS-26.5.2-arm64, 2026-08-08) and `benchmark/README.md` honesty notes (exp1 `embedding_metadata`-only without sqlite-vec ANN, exp2 lightweight `_TrackingOwner`, exp7 `_generation` set directly, etc.). No universal latency, p99, or throughput claim is made — use the measured `exp_governed_latency` p50 or run the harness on your own machine.

## Privacy and Security

### Can anyone else see my memories?

No. Your database is a local file on your machine. It is not synced, uploaded, or shared with anyone — including us — unless you explicitly enable Mesh peering, cloud backup, or a provider-backed mode.

### Does it guarantee regulatory compliance?

No software package certifies the complete deployment. SLM supplies local storage, memory erasure (`slm forget`/`slm delete`) and dashboard profile erasure, provenance, retention, access-policy (RBAC — see [RBAC and Teams](RBAC-Teams)), and hash-chained audit controls (see [GDPR Compliance](GDPR-Compliance) and [Compliance](Compliance)); applicability and sufficiency depend on the operator, use case, configuration, providers, and surrounding systems.

### Can I export my data?

The database is a standard SQLite file at `~/.superlocalmemory/memory.db`. Use `slm evidence export` for a checksummed JSONL bundle (see `slm evidence --help` and [GDPR Compliance](GDPR-Compliance)), or copy the data root. Configuration, logs, queues, models, derived indexes, and optional backend state also live in the data root.

### Can I delete all my data?

`slm forget "query"` deletes matching memories, and `slm delete <fact_id>` deletes an exact fact. V4 does not expose a `slm profile delete` CLI command; non-default profiles can be erased through **Dashboard → Governance → Data Privacy → Erase** with typed confirmation. To delete the complete installation, follow the documented erasure/uninstall procedure — do not assume removing only `memory.db` covers configuration, logs, queues, models, derived indexes, and optional backend state.

## Troubleshooting

### My AI does not seem to remember anything.

1. Check that SuperLocalMemory is running: `slm status` / `slm health`
2. Check that you have stored memories: `slm recall "test" --json`
3. Verify your IDE connection: restart the IDE after configuring MCP (`slm connect --list`)
4. Check the active profile: `slm profile list`
5. Inspect stuck ops: `slm ops list` or dashboard Operations

### Recall returns irrelevant results.

Try more specific queries. Use `slm trace "query"` to see which channels contribute — this helps diagnose whether the issue is semantic, keyword, temporal, associative, or entity matching.

### The setup wizard does not detect my IDE.

Use manual configuration. See [IDE Setup](IDE-Setup) for per-IDE config paths. Supported list via `slm connect --list`.

### Where do I find Bounded Loops, Framework Adapters, GDPR/RBAC, Multi-Agent Memory?

- Bounded Loops: [Bounded Loops](Bounded-Loops) + `slm loop --help` + MCP `slm_loop_*` (`code`/`full`/`power`/`whole`)
- Framework Adapters: [Framework Adapters](Framework-Adapters) — 9 adapters under `ide/integrations/`
- GDPR/RBAC: [GDPR Compliance](GDPR-Compliance), [RBAC and Teams](RBAC-Teams), [Compliance](Compliance)
- Multi-Agent Memory: [Multi-Agent Memory](Multi-Agent-Memory) + `SLM_AGENT_ID`

### Where can I report bugs?

Open an issue at [github.com/qualixar/superlocalmemory/issues](https://github.com/qualixar/superlocalmemory/issues).

---
*Part of [Qualixar](https://qualixar.com) | Created by [Varun Pratap Bhardwaj](https://varunpratap.com)*
