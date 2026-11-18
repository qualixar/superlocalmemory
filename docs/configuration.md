# Configuration

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
| **B: Smart Local** | Mode A + a model on this machine: Ollama by default, or any local OpenAI-compatible server. | No | Depends on the Ollama endpoint and optional integrations |
| **C: Full Power** | Your own endpoint or a cloud provider for enrichment and recall quality. | Yes for a cloud provider; no for a keyless custom endpoint | Configured query, ingestion, or enrichment content may be sent |

### Check your current mode

```bash
slm mode
```

### Switch modes

```bash
slm mode a    # Local Guardian (default)
slm mode b    # Smart Local
slm mode c    # Full Power
slm restart   # apply the new mode
```

Switching modes keeps your embedding, retrieval, forgetting and other settings
and loses no data. Run `slm restart` to apply it. If you switch to Mode C with
no provider configured, `slm mode c` tells you to run `slm provider set`.

### Mode A: Local Guardian (default)

Memories are saved and recalled without a language model, and nothing leaves this device. Core memory operations run against the local data root. Optional model and dependency downloads, connectors, backup, and other enabled integrations can use the network. On Apple Silicon the on-device answer check runs in this mode too; it adds no outbound call.

Best for: deployments that want a local core path and can govern optional integrations explicitly. Regulatory compliance still requires deployment-specific assessment.

### Mode B: Local LLM

Everything from Mode A, plus a model running on your machine that improves recall. Ollama is the default; any local OpenAI-compatible server (llama.cpp, vLLM, LM Studio) works too.

**Setup:**

```bash
# Install Ollama using its instructions for your platform.
# macOS example:
brew install ollama

# Pull a model
ollama pull llama3.2

# Switch to Mode B
slm mode b
slm restart

# Or point Mode B at another local server
slm provider set custom --endpoint http://localhost:8041/v1 --mode b
```

Best for: developers who can operate the selected local model and separately
govern optional networked integrations.

**What it costs.** Saving one memory asks the model to extract facts, about one
call per save. Then each stored fact costs one more call for its context
sentence, plus up to five calls checking it against similar facts for
contradictions. So a large backlog costs about six model calls per fact. Two
settings (4.2.14) trade that enrichment for speed:

```bash
# Contradiction checks: 0 = embedding check only, no model calls (default 5)
slm config set temporal_validator.max_llm_checks 0
# Context sentences written by rule instead of by the model (default true)
slm config set encoding.context_generation_llm false
slm restart
```

Contradiction checks only file proposals for you to review
(`slm corrections dismiss` clears unreviewed ones); they never change recall on
their own. Each mode keeps its own values, like its model choice.

### Mode C: Full Power

Uses your own endpoint, or a cloud provider such as OpenAI or Anthropic, for the best recall quality. Queries leave the device, and a cloud provider needs a key; a custom endpoint does not.

**Setup:**

```bash
slm mode c
slm provider set openai
slm restart
```

You will be prompted for your API key. It is stored locally in your config file and sent only to the provider you choose.

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
| Your own OpenAI-compatible endpoint | `slm provider set custom --endpoint URL` | None needed |

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

Settings live in the data folder, `~/.superlocalmemory/` by default (see
[Database Location](#database-location)):

| File | What it holds |
|------|---------------|
| `config.json` | The active mode, active profile, LLM and embedding settings, retrieval, evolution, scope and the other settings below |
| `mode_a.json`, `mode_b.json`, `mode_c.json` and `current_mode` | A saved copy of the settings for each mode, and the letter of the mode in use |
| `answer_check.json` | [Answer check](answer-check.md) settings, the same in every mode |
| `memory_kinds.json` | [Memory kind](memory-kinds.md) settings |
| `optimize.json` | [Optimize](optimize-config.md) settings |
| `store-check.json` | The result of the memory-store check; see [Troubleshooting](troubleshooting.md#memory-store-check) |

Most settings have a command or a dashboard page, so you rarely edit the JSON.
`slm config get KEY` reads any value in `config.json` with dot notation, and
`slm config set KEY VALUE` changes one of a short list of keys (see the [CLI
reference](cli-reference.md#configuration-and-adapters)).

### Shape of `config.json`

```json
{
  "mode": "a",
  "active_profile": "default",
  "llm": { "provider": "", "model": "", "base_url": "" },
  "embedding": {
    "model_name": "nomic-ai/nomic-embed-text-v1.5",
    "dimension": 768,
    "provider": ""
  },
  "retrieval": { "use_cross_encoder": true },
  "scope": { "default_scope": "personal" }
}
```

The file holds many more keys than this sketch shows, and SLM writes it for you.

### Key settings

| Setting | Default | Description |
|---------|---------|-------------|
| `mode` | `"a"` | Operating mode: `a`, `b` or `c`. Change it with `slm mode` |
| `active_profile` | `"default"` | Active memory profile. Change it with `slm profile switch` |
| `embedding.model_name`, `embedding.dimension`, `embedding.provider` | `nomic-ai/nomic-embed-text-v1.5`, `768`, auto-detect | Embedding model. Change it with `slm embedder switch`, which re-indexes in the background |
| `scope.default_scope` | `personal` | Where a new memory goes when you do not say. Shared memory is opt-in |
| `scope.recall_include_global`, `scope.recall_include_shared` | off | Whether recall includes other profiles' memories. See [Shared memory](shared-memory.md) |
| `mesh_enabled` | on | The agent mesh |
| `evolution.enabled` | off | Skill evolution, which makes background LLM calls when on |

> **Recall result limit:** The default is 20 results per query (CLI: `slm recall --limit N`; MCP `recall` tool: `limit` parameter). There is no config file key for this; override it per call.

## Remote embedding and rerank endpoints

Both halves of the retrieval stack can be served by a remote OpenAI-compatible
endpoint. This is how a non-English deployment replaces the bundled models: the
default reranker, `cross-encoder/ms-marco-MiniLM-L-12-v2`, is English-only and
cannot score a Chinese, Japanese, or Arabic corpus meaningfully.

| | Config block | Keys | Route |
|---|---|---|---|
| Embeddings | `embedding` | `provider: "openai"`, `api_endpoint`, `model_name`, `dimension` | `POST /v1/embeddings` |
| Reranking | `retrieval` | `cross_encoder_backend: "openai"`, `cross_encoder_endpoint`, `cross_encoder_model` | `POST /v1/rerank` |

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
rather than ignored.

Each remote rerank call has one time limit of 1 second (or your configured
timeout, if lower; the environment variable `SLM_REMOTE_RERANK_DEADLINE_S`
allows a longer one). After failures the endpoint is paused — 10 seconds,
doubling up to 5 minutes — and checked in the background; recall carries on
with its usual ranking meanwhile.

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

## Answer Check

[Answer check](answer-check.md) is configured from **Settings → Answer
check** in the dashboard — there is no config file workflow for it, since
most SLM users are not editing JSON by hand.

These settings are the same in every operating mode, so they are not kept in
`config.json` or the per-mode `mode_a/b/c.json` files: they live in one
owner-only file, `answer_check.json`, in the data folder. Switching modes
(`slm mode`, the MCP `set_mode` tool, or the dashboard) never changes them.
An install that never touched the answer check has no such file yet and uses
the defaults below. The dashboard writes these keys on your behalf, and the
settings API accepts a change only with SLM's own credential (the
dashboard's install token, the daemon capability, or an API key):

| Setting | Default | Description |
|---------|---------|-------------|
| `sufficiency_judge` | `"auto"` | `auto`, `laya` (on this Mac), `jev` (online), or `off`. `auto` is the starting state before anyone chooses: the on-device check once an install exists that passed its check, otherwise exactly as `off`. It never resolves to `jev` on its own, and the dashboard never offers it — choosing writes an explicit `laya`, `jev`, or `off` |
| `sufficiency_python`, `sufficiency_model`, `sufficiency_hf_home` | unset | Where the on-device model lives; written by the dashboard's setup/adopt flow, cleared by **Remove** |
| `sufficiency_timeout_s` | `1.5` | Seconds the on-device check waits before giving up on one recall |
| `sufficiency_jev_provider` | `"typesafe"` | `typesafe` or `openrouter` |
| `sufficiency_jev_consent` | `false` | Must be explicitly `true`; set only by the dashboard's consent checkbox |
| `sufficiency_jev_timeout_s` | `2.0` | Seconds the online check waits for a reply |
| `sufficiency_jev_rerank`, `sufficiency_jev_rerank_consent` | `false` | Reordering with Jev and its own consent; both must be `true`; set only by the dashboard switch |
| `sufficiency_jev_rerank_k` | `20` | How many top results reordering sends (clamped to 5–30) |

Recall's answer-support decision is configured under `retrieval` in `config.json`. Defaults:
`answer_support_gate` on, `query_rewrite` on (base forms and known aliases for the keyword
search only; the decision always reads your original question), `query_rewrite_llm` off
(in Mode B, a few keywords from the local model, bounded by `query_rewrite_timeout_s`, default
0.8, and failing open), and `min_semantic_evidence` 0.60 (the lowest raw cosine that counts as
meaning-only evidence).

The Answer Check tab's history lives in `config.json` under
`retrieval`, with the other recall settings. A change applies at the next
daemon start:

| Setting | Default | Description |
|---------|---------|-------------|
| `answer_check_history` | `true` | Keep outcomes and timings of recent checks (never questions or memories) for the dashboard's Answer Check tab |
| `answer_check_history_days` | `30` | Days kept (clamped to 1–365) |
| `answer_check_history_max_rows` | `10000` | Most checks kept per profile (clamped to 1,000–10,000; a data export reads at most 10,000 rows per table) |

The provider key is never stored in any of these files; it lives in its own
owner-only key store. Hand-editing `answer_check.json` works but is
unsupported — use the dashboard. A damaged file is read as "no consent".

## What SLM measures

SLM measures whether memory helps your agents, locally and as ids and numbers only. These settings live in `features.json` in the data folder (not `config.json`), changed with `slm measurement` or `slm flags set <flag> on|off`, or by the dashboard's Settings card "Help SLM measure what works". An environment variable `SLM_FEATURE_<FLAG>` (dots become underscores, upper case, for example `SLM_FEATURE_OUTCOMES_HOLDOUT=off`) overrides the file.

| Flag | Default | What it does |
|------|---------|--------------|
| `outcomes.receipts` | on | Writes a receipt each time SLM hands memories to an agent: memory ids, rank, score and the logged probability of each choice |
| `outcomes.holdout` | on | In about 1 in 10 session starts, leaves out one eligible memory or switches off one retrieval channel for that fetch (MCP session start only), by a logged random draw. Never pinned rules, standing rules, core blocks, explicit recall, cues or lease warnings |
| `outcomes.swaps` | on | In about 1 in 20 of an agent's explicit recalls, two neighbouring results trade places. Nothing is left out or added |
| `outcomes.usage` | off | Compares an agent's last reply with the memories it was given to see which were used; ids and scores only |
| `delivery.need_gate` | off | Decides per prompt whether memory is needed; logs each prompt's rule, score and a keyed digest (never the prompt text), and recalls 5% of the prompts it would skip to measure its own misses |

### Memory on every prompt, and reminders on cues

Two delivery features are off by default and turned on with `slm flags set <flag> on`:

- **`delivery.need_gate`: memory on every prompt.** When on, the UserPromptSubmit hook (`slm hook auto_recall`, registered in the Claude Code family of plugins and in Codex) asks memory before the agent answers, but only when the prompt needs it. Fixed rules skip the obvious prompts (an acknowledgement such as "yes" or "go ahead", a greeting, pure arithmetic, a translation request, a slash command). The rest are ranked by a simple score, and prompts under a threshold are skipped too. The threshold is never chosen by hand: it is set from your own logged prompts so that at most 10% of the prompts that needed memory are skipped (conformal risk control), and until enough prompts have a known outcome the score skips nothing. "Needed" means an agent used a memory delivered for that prompt within 24 hours, or asked memory itself before the next prompt; use detection finds some uses, not all, so this is a lower bound. A memory the conversation already got is never shown again. The hook answers within 7 seconds or adds nothing. `shadow` logs the threshold it would use without applying it. `slm health --memory` shows how many prompts were skipped and the bound.
- **`delivery.cues`: reminders when the agent touches something.** A memory saved with cues (`remember(cues=["path:src/**/schema.py", "tool:Bash:git push", "error:database is locked", "repo:myproject", "date:2026-12-01"])`, or `slm remember --cue path:...`) comes back right before or after the matching step: editing a matching file, running a matching command, a tool output with that error, working in that repository, or on or after that date. Plans that name a file get a path cue automatically. Editing a file that memories were saved from also brings a note that those memories may now be out of date. Each memory fires at most once per conversation. `slm cues list` shows them.

**Harm gate (shadow only).** `slm health --memory` also reports which memories, if any, measurably make the agent worse when included (from the randomised withholding above, with an anytime-valid bound per memory and false-discovery control across memories). In this release it only reports: it never removes a memory from what an agent sees.

**Opinions are labelled.** A remembered opinion or preference is shown to agents as "a view or preference someone stated, not a checked fact", so a model does not agree with it just because it was remembered.

`slm measurement off` turns `outcomes.holdout` and `outcomes.swaps` off together; add `--receipts` to turn `outcomes.receipts` off as well. `slm flags kill on` (or `SLM_4230_OFF=1`) turns every 4.2.30 setting off at once. The notice, the version of it you acknowledged and each on/off change are kept in `disclosures.json` in the data folder.

### Where a memory came from, and what may replace it (`trust.taint`)

Off by default (`slm flags set trust.taint shadow|on`). Every memory records how it arrived (typed by you, saved by an agent, a tool's output, a web page or mail, an imported file, built by a model) and SLM turns that into one of ten classes, best first: `user_direct`, `user_confirmed`, `system`, `agent_inferred`, `tool_output`, `mesh_peer`, `web`, `imported_file`, `model_generated`, plus `unknown` for memories saved before 4.2.30 or whose source was not kept. A memory's **taint** is the lowest class among it and everything it was built from, so a note cut from a web page is never trusted more than the page. Text a model rewrote is capped at `model_generated`, because a model can no longer vouch for who said it. The mapping is one table with a reason on each row (`trust/origin_class.py`).

- **`shadow`** computes and stores each memory's class and taint (table `fact_trust`, filled at save time and in small batches by the maintenance cycle, never on a read) and adds `origin` to what recall and the dashboard show; it changes nothing else. The slot pass counts how many replacements the rule below would have held.
- **`on`** adds the supersession rule: nothing replaces a memory automatically unless the newer one's class is at least as trusted as the older one's. A web page cannot overwrite something you said, and a tool's output cannot overwrite an agent's own words. When the rule refuses, the two memories stay and SLM asks you which is true now (reason `origin_below_head`; no judge answers it). `unknown` on either side always asks. Your own answer, in the dashboard or with `slm disputes settle`, always counts.

The command line is told apart from an agent in a shell by one fact: whether it sat at an interactive terminal. Every `slm` call to the daemon says so (`X-SLM-Client-Shell: tty` or `no_tty`, believed only from a caller that holds the daemon's private capability) and the daemon records it with the memory (`fact_origin.agent_id_source` = `shell:tty` or `shell:no_tty`). A save with no agent id and **no terminal** is classed `agent_inferred`, not `user_direct` (mapping row `remember.shell_no_tty`). The dashboard (a person at a browser) and a terminal stay `user_direct`. Two limits, plainly: a terminal is evidence, not proof (an agent that allocates a pseudo-terminal looks like a person), and memories saved before this change carry no such signal, so they are not reclassified.

### The instruction firewall (`trust.taint` on)

A memory is data. It becomes an **instruction** when SLM puts it where an agent obeys it: a pin, a standing rule or decision, a core memory block, or the sections a session starts with. With `trust.taint` set to `on`, a memory whose taint is **below `system`** (a tool's output, a peer agent's message, a web page, an imported file, a model's text, an agent's own words, or anything saved before 4.2.30, which is `unknown`) cannot be any of those until a person approves it. It is still found by recall, as evidence inside the untrusted frame, and nothing is deleted.

- **Where it holds a memory back:** the standing rules and decisions of a session start; pins (the `core_memory` pin answers `needs_approval` and records the request); the sources of core memory blocks (a block compiled earlier from such a memory is not read back); the core section of any rendered context (a memory whose taint the renderer was not told is never promoted to core); and the recent and project memories the command-line session start adds.
- **Who approves:** a person, in two places. `slm trust approve FACT_ID` (or `--all`) works only at an interactive terminal; the dashboard card **Waiting for your OK** has **Approve** and **Approve all**. Approving raises the memory to `user_confirmed` and records who, when and through which door (table `trust_approvals`, M081; the class itself is in `fact_trust`). An agent can read the queue (`slm trust pending --json`, or `waiting_items` in `slm health --memory`) and can never approve: the command refuses without a terminal, and the server refuses a request that says it had none or that claims the dashboard's door while holding the daemon capability. A refused pin is applied when the person approves it.
- **The first switch-on:** memories saved while the flag was off have no class until the maintenance batch computes it (a few thousand per cycle); until then they are held back, because a memory with no class is unknown. Everything saved before 4.2.30 is `unknown`, so a store's old pins and rules wait: `slm trust pending` lists them with where each came from, and `slm trust approve --all` clears the list in one answer.
- **`shadow`** changes nothing and lets the Trust section of `slm health --memory` and the dashboard card show what `on` would hold. The kill switch reads as off. The floor (`system`) is one constant, `trust/firewall.py:FLOOR`.

### Signed memories and the keyed audit chain (`trust.signing`)

Off by default (`slm flags set trust.signing shadow|on`, then restart the daemon). The daemon reads one random key from your operating system's keychain at start (macOS Keychain, Windows Credential Manager, libsecret; no keychain means signing is unavailable, never a key file) and signs each memory it saves, plus the ones saved elsewhere in small background batches. Signatures detect a memory changed by something that has your files but not your login session (a synced or restored copy, another user account, a disk image); they do not stop a process running as you. With the flag on, each new audit-chain link is also an HMAC under the same key. `slm doctor --truth` and `slm signing status` report modified and unsigned memories by id. A lost key loses no data. Export, import, folds and restores: [Signed memories](signed-memories.md).

### The secret screen (`trust.secret_screen`)

Until 4.2.30 a key, token or private key you saved was stored as you wrote it, searchable and journaled (only text leaving the machine was redacted). With `trust.secret_screen` set to `on`, every door that saves text (`remember` on the MCP tools, the command line, the dashboard and the HTTP API; `observe`; the hooks; imports, documents and pictures; edits of a memory; the metadata of a save) replaces anything shaped like a credential with a typed placeholder first: `[REDACTED:GITHUB:ab12]` keeps the type and, for a long key, its last four characters. The original is kept nowhere: not in the memory tables, the search index, the admission journal, the hook spool or a log line (logs carry counts per type only). It screens what a key looks like (vendor key formats, PEM private keys, a long high-entropy run), not a sentence such as "the wifi password is hunter2", which is stored as written because redaction at storage cannot be undone. If the screen cannot run while it is on, the save is refused rather than stored in clear. `shadow` logs what it would replace and stores the text as written. `slm health --memory` counts the placeholders held in the store by type, never a value. A retry of a save made before the screen was turned on with the same idempotency key is reported as a conflict, as with personal-data redaction.

### The living wiki and the dream cycle (`compiler.wiki`, `compiler.dream`)

Both are off by default. `compiler.dream` (`slm flags set compiler.dream on`) lets the background cycle queue extractive pages and exact-duplicate merges for your review, at most every six hours per profile; it asks no model, applies nothing and writes only proposal tables. `compiler.wiki` (`on`, or `shadow` to see what would be written) lets `slm wiki build` write the approved pages as Markdown and `slm wiki sync` read your edits back as proposals. The folder is `SLM_WIKI_DIR`, or `dir` in `wiki.json` in the data folder (`slm wiki dir PATH`), or `wiki/` in the data folder. `compiler.auto_apply` stays off and no certificate covers a page or an edit of your words, so a person decides. The kill switch reads both as off. [The living wiki](living-wiki.md).

## Consistency Checking at Store Time

The optional consistency check that runs when a memory is stored
(`math.sheaf_at_encoding`) is **off by default**, including on
installs upgraded from an earlier release — a measured comparison on a real
store found it changed no recall answer while running on every store and
every maintenance pass.

**What turning it off changes.** In Mode A (and in Mode B or C whenever no
language model is reachable) this check is also how storing a memory that
contradicts an older one marks the older one as no longer current. With it
off, both memories stay current and recall ranks them as usual. Nothing is
deleted either way. Mode B and C with a reachable model still check for
contradictions with the model; they pick what to check by shared names
instead.

**Turning it back on.** Set `"sheaf_at_encoding": true` and
`"sheaf_default_reviewed": true` under `math` in `config.json`. Both keys
make the choice unambiguous; a `math` section you write yourself that sets
`sheaf_at_encoding` to `true` is also respected. The one case read as the old
default is a full `math` section saved by an older release, which wrote `true`
for everyone; that is switched off once, with a warning in the log naming
the setting and how to turn it back on.

## Environment Variables

These override config file settings when set:

| Variable | Purpose |
|----------|---------|
| `SLM_DATA_DIR` | Data folder (default `~/.superlocalmemory/`). `SL_MEMORY_PATH` and `SLM_HOME` are older names for it |
| `SLM_PYTHON` | Python interpreter the `slm` command uses |
| `SLM_DAEMON_PORT` | Daemon port (default 8765) |
| `SLM_RANKING` | Adaptive ranking. Off unless set to `v1`, `v2` or `v2-ensemble` |
| `SLM_MCP_PROFILE` | MCP tool set: `core`, `code`, `full`, `power`, `mesh` or `whole`. See [MCP Tools Reference](mcp-tools.md#which-tools-a-client-sees) |
| `SLM_MCP_ALL_TOOLS` | `1` exposes all tools |
| `SLM_MCP_TOOLS` | Comma-separated allowlist of tool names |
| `SLM_MCP_MESH_TOOLS` | `1` or `true` exposes the mesh tools even if mesh is off in the config |
| `SLM_AGENT_ID` | Default agent attribution for MCP calls |
| `SLM_RATE_LIMIT_WRITE`, `SLM_RATE_LIMIT_READ`, `SLM_RATE_LIMIT_WINDOW` | Daemon HTTP rate limit (defaults 30 writes, 120 reads, 60 seconds) |
| `SLM_REMOTE_RERANK_DEADLINE_S` | Longer time limit for a remote rerank call |
| `SLM_LOOP_KILL` | Any non-empty value stops bounded loops |
| `SLM_CROSS_ENCODER_API_KEY` | Runtime-only bearer token for a remote rerank endpoint; overrides any owner-only config value |
| `SLM_SKIP_LAYA` | `1` (also `true`, `yes`, `on`) stops setup from installing the on-device answer check; `slm setup --no-laya` does the same. `SLM_INSTALL_LAYA=0` also opts out |
| `SLM_RECALL_NO_SUPPORT_GATE` | `1` turns the answer-support decision off, so recall no longer returns "That isn't in SuperLocalMemory." or the closest-memories tier ([Recall](recall.md#when-memory-cannot-answer)) |
| `SLM_RECALL_NO_FLOOR` | `1` turns the evidence floor off |
| `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `AZURE_OPENAI_API_KEY`, `OPENROUTER_API_KEY` | Provider keys for Mode C |

## Database Location

All data is stored locally in:

```
~/.superlocalmemory/memory.db    # SQLite database
~/.superlocalmemory/config.json  # Configuration
~/.superlocalmemory/backups/     # Backups, including the copy taken before a repair
~/.superlocalmemory/logs/        # daemon.log and daemon-error.log
```

To use a custom location:

```bash
export SLM_DATA_DIR="/path/to/your/data"
```

---

## Multi-Machine Mesh

| Variable | Default | Description |
|---|---|---|
| `SLM_MESH_PEER_URL` | unset | Full URL of remote SLM instance (e.g., `http://192.168.1.100:8765`) |
| `SLM_MESH_SHARED_SECRET` | unset | Shared bearer token — same on both machines. Required when `SLM_MESH_HOST` is not localhost. |
| `SLM_MESH_HOST` | `127.0.0.1` | IP to bind this machine's mesh listener |
| `SLM_MESH_WS_PORT` | `7900` | Port used for mDNS service announcement |
| `SLM_MESH_DISCOVERY` | `on` | Set to `off` to disable mDNS auto-discovery. Only used when the mesh talks to other computers |

See [Multi-Machine Setup](./multi-machine.md) for full setup guide.

---

## Optimize Configuration

The **Optimize** module reduces LLM cost with caching and compression. Configuration lives in a separate file at `~/.superlocalmemory/optimize.json` and hot-reloads within 2 seconds — no daemon restart required.

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

*SuperLocalMemory — Copyright 2026 Varun Pratap Bhardwaj. AGPL-3.0-or-later. Part of Qualixar.*
