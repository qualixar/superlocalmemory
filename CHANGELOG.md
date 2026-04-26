# Changelog

All notable changes to SuperLocalMemory V3 will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

### [Unreleased]
- **License:** Changed from Elastic-2.0 to AGPL-3.0-or-later to protect research IP

---

## [3.4.37] - 2026-04-26

**P0 RAM fix.** Total SLM footprint reduced from ~14 GB peak to ~2.3 GB peak
(84% reduction). Idle dropped from ~2.5 GB to ~1.0 GB. Users with 16 GB
laptops can now run SLM without uninstalling.

### Fixed
- **CoreML EP allocation** — Added `ORT_DISABLE_COREML=1` to
  `recall_worker.py`, `cli/commands.py` (warmup diagnose path), and the
  Popen environment dicts in `core/embeddings.py` and
  `retrieval/reranker.py`. Previously only `embedding_worker.py` and
  `reranker_worker.py` set this. On ARM64 Mac, ONNX Runtime's CoreML
  Execution Provider allocated 3-5 GB per missing guard.
- **Duplicate MemoryEngine** — The QueueConsumer (recall_queue.db drain)
  was routing through `WorkerPool` → `recall_worker` subprocess, which
  loaded a SECOND full MemoryEngine inside the daemon. Now routes through
  the daemon's in-process engine via the new `EngineRecallAdapter`.
  Eliminates ~800 MB of duplication.
- **Eager warmup** — Removed `WorkerPool.shared().warmup()` from daemon
  startup. The recall_worker subprocess no longer spawns at boot. It
  remains available as a fallback for dashboard/chat routes.

### Changed
- **RSS limits tightened:**
  - `embedding_worker` self-kill: 4000 MB → 1800 MB
  - `recall_worker` self-kill: 2500 MB → 1500 MB
  - Daemon watchdog `MAX_WORKER_MB`: 4096 MB → 1800 MB
  - `HealthMonitor.global_rss_budget_mb`: 4096 MB → 2500 MB
- **Watchdog interval:** 60s → 15s in both daemon watchdog and
  HealthMonitor `check_interval_sec`. Catches memory spikes faster.
- **Idle timeouts:**
  - `SLM_EMBED_IDLE_TIMEOUT`: 1800s (30 min) → 300s (5 min)
  - `SLM_RERANKER_IDLE_TIMEOUT`: 1800s → 300s
  - Reduces idle RAM held by ML model subprocesses.

### Added
- **`EngineRecallAdapter`** in `unified_daemon.py` — wraps the in-process
  MemoryEngine to satisfy `RecallPoolProtocol` for the QueueConsumer.
  Eliminates the recall_worker subprocess on the hot path.

---

## [3.4.36] - 2026-04-25

Persistent hook daemon: recall latency drops from ~2.2s to sub-second by
eliminating Python subprocess startup on every prompt.

### Added
- **`hooks/hook_daemon.py`** — Unix domain socket server that keeps a
  long-lived process for recall requests. Claude Code connects via socket
  instead of spawning a fresh Python interpreter per prompt. Eliminates
  ~300-500ms of subprocess overhead. Starts/stops with the SLM daemon.
- **Auto-restart watchdog:** `ensure_hook_daemon()` checks socket health
  and restarts the daemon if it died. Claude Code hooks call this before
  connecting, so a crashed daemon is transparent to the user.
- **Graceful fallback:** if the socket is unavailable, the hook
  automatically falls back to the v3.4.35 subprocess path. Claude Code
  performance is NEVER impacted by daemon failure.
- **9 new tests** for daemon lifecycle, socket protocol, ack detection,
  watchdog, fallback, and memory safety.

### Performance
- Ack prompts: ~5ms via socket (was 30ms via subprocess)
- Substantive recall: target sub-1s (was 2.2s p50 via subprocess)
- Hook daemon RSS: ~15-20MB (no engine, no ONNX, no PyTorch)

---

## [3.4.35] - 2026-04-25

Production auto-recall: every Claude Code prompt automatically retrieves the
top relevant memories via the unified queue, so the agent has continuous-
learning context without the user invoking recall manually.

### Added
- **`hooks/auto_recall_hook.py`** — production UserPromptSubmit handler.
  Reads stdin JSON from Claude Code, detects ack prompts (silent fast path),
  enqueues substantive prompts to `recall_queue.db`, polls for the result
  with mode-aware timeout (A=10s, B=25s, C=40s), and injects the top-K
  memories as Claude Code's `hookSpecificOutput.additionalContext` envelope.
  Wraps recalled content in untrusted-boundary markers so the LLM treats
  it as data, not instructions. Fail-open on any error.
- **`core/queue_consumer.py`** — daemon background thread that drains
  `recall_queue.db`. Claims jobs atomically, routes through `pool.recall()`
  (engine never loaded in MCP/hook processes), writes results back. Priority
  lanes (high=recall, low=consolidate). Periodic cleanup of completed rows.
- **`slm hook auto_recall`** CLI subcommand wires Claude Code to the hook.
- **50 new tests** — `test_queue_consumer.py` (11) + `test_auto_recall_hook.py`
  (39). Full TDD coverage including ack detection, fencing, dedup, fail-open.

### Changed
- **`core/recall_queue.py`** — `complete()` now wrapped in `BEGIN IMMEDIATE`
  for fencing-token atomicity under multi-process access. Dedup hash
  includes `namespace` to prevent cross-namespace result collisions.
- **`server/unified_daemon.py`** — starts QueueConsumer on boot, stops on
  shutdown.
- **`hooks/hook_handlers.py`** — dispatches `auto_recall` to the new hook.

### Performance
- p50 recall latency: 1.75s (40-prompt integration test, Mode B)
- p99 recall latency: 11.83s
- Hook process RSS: ~20 MB (no engine loading, no memory blast)
- Ack prompts: 30 ms (silent, no recall)

---

## [3.4.34] - 2026-04-25

Fix: user's mode choice can no longer be silently overwritten.

### Fixed
- **Mode protection in `SLMConfig.save()`.** Any `save()` call that would
  change the mode in `config.json` is now blocked unless the caller passes
  `mode_change=True`. This prevents accidental mode resets when code creates
  a fresh `SLMConfig()` (defaults to Mode A) and calls `save()` to persist
  an unrelated field change. A warning is logged when a silent mode change
  is blocked.
- **MCP `set_mode` preserves user settings.** Previously `set_mode` created
  a fresh `SLMConfig.for_mode()` that lost all user customizations (LLM
  provider, API keys, embedding config, active profile). Now carries forward
  all settings from the existing config, matching the dashboard behavior.
- All intentional mode-change paths (`slm mode`, MCP `set_mode`, dashboard
  PUT `/api/v3/mode`, setup wizard) pass `mode_change=True`.

---

## [3.4.33] - 2026-04-25

Fix: daemon leaked SQLite connections to learning.db via bandit threadlocals.

### Fixed
- **Bandit threadlocal connection leak.** `reward_proxy.settle_stale_plays`
  creates a `ContextualBandit` that opens a threadlocal connection via
  `_conn_for`. When called from `asyncio.to_thread` (bandit_loops.py,
  every 60 s), each thread-pool thread kept its connection open for the
  process lifetime. Over 24 h this accumulated 12+ leaked file descriptors
  and ~100 MB of wasted SQLite page-cache RAM. New
  `bandit.close_threadlocal_conn()` function, called in the
  `settle_stale_plays` finally block, ensures pool threads release their
  connections immediately.
- **Corrected embedding worker memory comment.** The `~200MB footprint`
  note was written for `all-MiniLM-L6-v2`; the default model
  `nomic-ai/nomic-embed-text-v1.5` uses ~1.1 GB via ONNX.

---

## [3.4.32] - 2026-04-24

Fix: concurrent remembers no longer block recalls on the shared embedder.

### Fixed
- **Daemon `/remember` is now async by default.** Writes to the pending
  queue in under 100 ms and returns a `pending_id`; a background thread
  drains the queue in the background. Previously, the synchronous
  `engine.store()` on the FastAPI event loop could block `/search` and
  `/health` for 30+ seconds while the single embedder worker processed a
  large write. Under concurrent load the daemon could appear hung.
- **Materializer yields to active recalls.** While any `/search` is in
  flight the drainer sleeps between items, so user-initiated recalls
  always get the embedder first.
- **MCP remember tool simplified.** Writes to `pending.db` and returns;
  the daemon's materializer completes the pipeline. Removes the
  redundant in-process `pool.store` background task that previously
  contended with `/search`.
- **`pool_store` returns `["pending:<id>"]`** when the daemon is async,
  keeping a stable identifier for callers without blocking on the
  embedder.

### Added
- `?wait=true` query parameter on `POST /remember` for callers that
  need synchronous behaviour and real `fact_ids` in the response.
- `superlocalmemory.core.recall_gate` module — shared counter that lets
  the materializer detect in-flight recalls and yield priority.

### Migration notes
- **No action required.** Existing clients continue to work; the
  response shape is compatible (`ok`, `count` still present). Scripts
  that depended on `fact_ids` to validate the write should switch to
  `pending_id` or pass `?wait=true` to opt in to the legacy behaviour.

---

## [3.4.31] - 2026-04-24

Dashboard truth, memory vs fact clarity, and self-cleaning pending queue.

### Changed
- **Dashboard now shows both memory counts honestly.** Parent memories
  (what you stored) and atomic facts (what retrieval indexes) appear as
  two distinct cards with their ratio. No more "Total Memories: 6,000"
  when you actually have 2,000 memories decomposed into 6,000 facts.
- **"Browse atomic facts"** relabeled for clarity — this view lists the
  indexed atomic units.
- **Visible search box** in the Memories tab — previously hidden behind
  the Recall Lab only. Search now debounces 280 ms on input.

### Added
- **`/api/memories/{id}/detail`** — full memory + all child atomic facts
  in one call. Powers the click-to-expand modal.
- **`/api/facts/{id}`** — single atomic fact detail with source memory
  content, entities, and canonical entities.
- **Pagination UI** — Prev/Next controls show "Showing 1–50 of 6,123".
  Previously hardcoded to 50 with no navigation.
- **CSV export** — new `format=csv` option on `/api/export` plus a
  dedicated "Export All (CSV)" menu item. JSON and JSONL still work.
- **Export progress toast** — "Preparing JSON export…" notification
  before the download starts.
- **`total_facts` + `facts_per_memory`** in `/api/stats` response.
- **Pending queue auto-cleanup** — the maintenance scheduler now sweeps
  the pending queue every cycle: completed rows > 7 days, failed rows
  over retry limit, and stuck rows > 7 days are removed; a 30-day hard
  cap prevents runaway growth on any status.

### Fixed
- **Test isolation** — `pending_store` now honors `SLM_DATA_DIR`. Four
  MCP remember tests were writing to the live `~/.superlocalmemory/`
  instead of `tmp_path`. Root conftest now forces `SLM_DATA_DIR=tmp_path`
  for every test unless explicitly opted out.
- **Fact click popup** — was calling `/api/v3/recall/trace` with a text
  substring (re-query by first 100 chars) and colliding with the memory
  row click handler. Now scoped to `.fact-result-item` only, hits the
  new `/api/facts/{fact_id}` endpoint.
- **Memory modal ID confusion** — the modal labeled `mem.id` as "ID"
  regardless of whether it was a memory_id or fact_id. Now displays
  both "Memory ID" and "Fact ID" when they differ.
- **Memory modal hydration** — fetches the full memory + fact list
  asynchronously when opened, so source content and entity data appear
  even for rows that arrived from the search endpoint.

---

## [3.4.30] - 2026-04-24

Multi-IDE shared worker, silent migration, and security hardening.

### Added
- **Multi-IDE RAM sharing.** MCP processes share a single recall worker
  via the daemon. Total RSS stays below 2 GB with four IDEs open.
- **Feedback and learning signals** flow from every IDE session to the
  daemon, not just the first.
- **Setup wizard** validates the data directory at install time and
  rejects iCloud, Dropbox, OneDrive, Box, Google Drive, and
  `Library/CloudStorage` paths that silently corrupt SQLite WAL.
- **One-time upgrade banner** after `pip install -U` / `npm install -g`
  points users to `slm doctor`.
- **`docs/errors.md`** — canonical error catalog with codes, recovery
  steps, exit codes, and HTTP status mappings.
- **CI matrix** now runs on `ubuntu-22.04`, `macos-14` (Apple Silicon),
  and `windows-latest` with `portalocker`.

### Changed
- **Silent, atomic data migration** on upgrade — no manual steps.
- **Migration serialized via file lock** so parallel pip + npm installs
  cannot race.
- **Concurrent-safe MCP engine singleton** with double-checked locking.
- Pool adapter returns frozen dataclasses instead of `SimpleNamespace`.

### Security
- File permissions tightened: marker files written at 0600, parent
  directories at 0700.
- Symlink-following blocked on version marker reads.
- Cloud-synced directory detection extended to `Library/CloudStorage`
  (macOS 13+).

### Fixed
- Silent error swallows in daemon shutdown, migration probe, and banner
  emission now log at WARNING.
- Fenced-out `complete()` writes (stale worker claims) emit a WARNING
  log instead of vanishing silently.
- Daemon-start migration guarded behind `is_ready` sentinel — skips
  when already applied.

---

## [3.4.23] - 2026-04-21

Critical hotfix on top of 3.4.22 for two end-user-facing regressions.

### Fixed
- **Daemon error log no longer balloons.** A ternary passed as the
  `logger.info` format string caused a `TypeError` on every startup in 24/7
  mode. Python's logging module then dumped the full FastAPI
  `merged_lifespan` stack to stderr; over a day the LaunchAgent log grew to
  tens of MB. The call is now pre-formatted. A defensive log-rotation pass
  at startup truncates any daemon log over 10 MB so users upgrading from
  3.4.22 get a clean slate on first boot.
- **Dashboard no longer hangs after a daemon upgrade.** Static JS/CSS/HTML
  was served without cache headers, so browsers served stale modules after
  `slm restart` and the dashboard showed an infinite spinner. All static
  responses now ship `Cache-Control: no-cache, must-revalidate`, and
  `index.html` embeds the server version; on mismatch the tab clears
  `localStorage` (preserving theme) and hard-reloads once.
- **Fetches can no longer hang forever.** A global `fetch` patch attaches a
  15-second `AbortController` timeout to every relative-URL request, so a
  dead socket surfaces as a rejection instead of leaving a spinner
  spinning. No callsite changes required.

### Added
- `GET /api/version` — returns the running daemon version; consumed by the
  dashboard version-fingerprint auto-reload.

---

## [3.4.22] - 2026-04-18

Hardening release — correctness, stability, and security fixes.

### Added
- `slm benchmark` plus escape-hatch commands (`disable`, `enable`,
  `clear-cache`, `reconfigure`).
- One-time upgrade banner on first boot after install.

### Changed
- Tighter defaults for the interactive installer.
- Licence: AGPL-3.0-or-later.
- Node.js prerequisite: ≥ 18.

### Security
- Hardened redaction, path validation, and token handling per internal
  audit. No end-user-visible behaviour change.

### Compatibility
- Fully backward compatible. `atomic_facts` is never modified by any
  migration. All upgrades are additive.

---

## [3.4.19] - 2026-04-17

### Fixed
- Recall cold-start eliminated. Embedding + reranker workers stay warm for 30 minutes by default instead of 2 minutes, so bursts of recalls no longer pay a 30-60 second model-load tax on every other query.

### New environment variables
- `SLM_EMBED_IDLE_TIMEOUT` — seconds to keep the embedding worker warm (default 1800). Set to 120 to restore pre-v3.4.19 behavior.
- `SLM_RERANKER_IDLE_TIMEOUT` — same, for the cross-encoder reranker (default 1800).

---

## [3.4.18] - 2026-04-17

### Fixed
- pip and npm installs now ship identical functionality. Semantic search and cross-encoder reranking work out of the box on pip (previously required `pip install superlocalmemory[search]`).
- First pip run auto-installs Claude Code hooks when Claude Code is detected, matching the npm postinstall experience.

---

## [3.4.17] - 2026-04-17

### Fixed
- Entity Explorer no longer stuck on "No entities found" after switching operating modes.
- Engine-backed routes (entity, ingest, recall, remember, list) auto-recover after mode changes — no daemon restart required.

### Added
- Mode change audit log at `~/.superlocalmemory/logs/mode-audit.log`.
- Mode C now requires an explicit API key via Settings to prevent accidental cloud-mode writes.

---

## Author

**Varun Pratap Bhardwaj**
*Solution Architect*

SuperLocalMemory V3 - Intelligent local memory system for AI coding assistants.

---

## [3.3.28] - 2026-04-07 — Stability Hotfix

### Fixed
- **Excessive memory usage during rapid file edits** — auto-observe now reuses a single background process instead of spawning one per edit. Rapid multi-file operations (parallel agents, branch switching, batch edits) no longer risk high memory usage.
- **Observation debounce** — rapid-fire observations are batched and deduplicated within a short window, reducing redundant work.
- **Memory-aware worker management** — new safety check skips heavy processing when system memory is low.

### New Environment Variables
| Variable | Default | Description |
|----------|---------|-------------|
| `SLM_OBSERVE_DEBOUNCE_SEC` | `3.0` | Observation batching window |
| `SLM_MIN_AVAILABLE_MEMORY_GB` | `2.0` | Min free RAM for background processing |

---

## [3.3.3] - 2026-04-01 — Langevin Awakening

### Fixed
- **Langevin dynamics now active** — positions were never initialized at store time, causing the entire Langevin lifecycle system to be inert (0 positioned facts). New facts now receive near-origin positions (Strategy A).
- **Backfill for existing facts** — maintenance now initializes unpositioned facts using metadata-aware equilibrium seeding (Strategy B) followed by 50-step burn-in (Strategy C). Old, rarely-accessed facts land in their correct lifecycle zones immediately.

### Improved
- Maintenance returns `langevin_backfilled` count for observability
- Health check now reports positioned facts accurately after backfill

---

## [3.3.0] - 2026-03-31 — The Living Brain

### New Features
- **Adaptive Memory Lifecycle** — memories naturally strengthen with use and fade when neglected. No manual cleanup needed.
- **Smart Compression** — embedding precision adapts to memory importance, achieving up to 32x storage savings on low-priority memories.
- **Cognitive Consolidation** — automatic pattern extraction from clusters of related memories. Your knowledge graph self-organizes.
- **Pattern Learning** — auto-learned soft prompts injected into agent context at session start. The system teaches itself what matters.
- **Hopfield Retrieval** — 6th retrieval channel for vague or partial query completion. Ask half a question, get the whole answer.
- **Process Health** — automatic detection and cleanup of orphaned SLM processes. No more zombie workers.

### New CLI Commands
- `slm decay` — run memory lifecycle review
- `slm quantize` — run smart compression cycle
- `slm consolidate --cognitive` — extract patterns from memory clusters
- `slm soft-prompts` — view auto-learned patterns
- `slm reap` — clean orphaned processes

### New MCP Tools
- `forget` — programmatic memory archival via lifecycle rules
- `quantize` — trigger smart compression on demand
- `consolidate_cognitive` — extract and store patterns from memory clusters
- `get_soft_prompts` — retrieve auto-learned patterns for context injection
- `reap_processes` — clean orphaned SLM processes
- `get_retention_stats` — memory lifecycle analytics

### Dashboard
- 7 new API endpoints for lifecycle stats, compression stats, patterns, and process health
- New dashboard tabs: Memory Lifecycle, Compression, Patterns

### Improvements
- Mode A/B memory usage reduced from ~4GB to ~40MB (100x reduction)
- Embedding migration on mode switch (auto-detects model change)
- Forgetting filter in retrieval pipeline (archived memories excluded from results)
- 6-channel retrieval (was 5)

### Migration
- Fully backward compatible with 3.2.x
- New tables created automatically on first run
- No manual migration needed

---

## [3.2.2] - 2026-03-30

### Added
- Performance improvements for retrieval pipeline
- New memory management capabilities with configurable lifecycle controls
- Enhanced dashboard with 3 additional monitoring tabs
- 9 new API endpoints for configuration and status
- 5 new MCP tools for proactive memory operations
- 5 new CLI commands for configuration management

### Changed
- Internal retrieval architecture optimized with additional search channel
- Schema extensions for improved data management (9 new tables)
- Memory surfacing engine with multi-signal scoring

### Performance
- Significant latency reduction in recall operations (vector-indexed retrieval)
- Idle-time memory optimization for large stores
- Reduced memory footprint for long-running sessions

---

## [3.2.1] - 2026-03-26

### Fixed
- **Windows `slm --version` / `slm -v`** — `.bat` and `.cmd` wrappers now intercept `--version`/`-v` directly (fast path, no Python needed) and set `PYTHONPATH` to the npm package's `src/` directory before launching Python. Previously, Windows users hitting `slm.bat` instead of the Node.js wrapper got `unrecognized arguments: --version` because Python resolved an older pip-installed version without the flag.
- **Unix bash wrapper** (`bin/slm`) — now sets `PYTHONPATH` and intercepts `--version`/`-v`, matching the Node.js wrapper's behavior. Previously relied on npm's shim always routing to `slm-npm`.
- **`postinstall.js`** — now runs `pip install .` to install the `superlocalmemory` Python package itself (not just dependencies). Prevents stale pip-installed versions from shadowing the npm-distributed source. Falls back to `--user` for PEP 668 environments.
- **`preuninstall.js`** — corrected version string from "V2" to "V3".
- **Windows Python detection** — added `py -3` (Python Launcher for Windows) as a fallback candidate in `slm.bat`.
- **Environment parity** — all three entry points (`slm-npm`, `slm`, `slm.bat`) now set identical PyTorch memory-prevention env vars (`PYTORCH_MPS_HIGH_WATERMARK_RATIO`, `TORCH_DEVICE`, etc.).

---

## [3.2.0] - 2026-03-26

### Added
- **`slm doctor` command** — comprehensive pre-flight check: Python version, all dependency groups, embedding worker functional test, Ollama connectivity, API key validation, disk space, database integrity. Supports `--json` for agent-native output.
- **`slm hooks install`** listed in CLI reference and README.
- Dashboard, learning (lightgbm), and performance (diskcache, orjson) dependencies now install automatically during `npm install`.

### Fixed
- **Warmup reliability** — increased subprocess timeout from 60s to 180s for first-time model download. Added step-by-step progress output and direct in-process import diagnostics when worker fails.
- **Mode B default model** — changed from `phi3:mini` to `llama3.2` to match `provider_presets()` and reduce first-time setup friction.
- **postinstall.js** — now installs all 5 dependency groups (core, search, dashboard, learning, performance) with clear status messages per group.
- **Error messages** — all embedding worker failures, engine fallbacks, and dashboard errors now suggest `slm doctor` for diagnosis.
- **pyproject.toml** — added `diskcache` and `orjson` to core dependencies; aligned optional dependency versions with core.

---

## [3.0.31] - 2026-03-21

### Fixed
- Profile switching and display uses correct identifiers
- Profile sync across CLI, Dashboard, and MCP — all entry points now see the same profiles
- Profile switching now persists correctly across restarts
- Resolve circular import in server module loading

---

## [2.8.6] - 2026-03-06

### Fixed
- Environment variable support across all CLI tools
- Multi-tool memory database sharing

### Contributors
- Paweł Przytuła (@pawelel) - Issue #7 and PR #8

---

## [2.8.3] - 2026-03-05

### Fixed
- Windows installation and cross-platform compatibility
- Database stability under concurrent usage
- Forward compatibility with latest Python versions

### Added
- Full Windows support with PowerShell scripts for all operations
- `slm attribution` command for license and creator information

### Improved
- Overall reliability and code quality
- Dependency management for reproducible installs

---

## [2.8.2] - 2026-03-04

### Fixed
- Windows compatibility for repository cloning (#7)
- Updated test assertions for v2.8 behavioral feature dimensions

---

## [2.8.0] - 2026-02-26

**Release Type:** Major Feature Release — "Memory That Manages Itself"

SuperLocalMemory now manages its own memory lifecycle, learns from action outcomes, and provides enterprise-grade compliance — all 100% locally on your machine.

### Added
- **Memory Lifecycle Management** — Memories automatically organize themselves over time based on usage patterns, keeping your memory system fast and relevant
- **Behavioral Learning** — The system learns what works by tracking action outcomes, extracting success patterns, and transferring knowledge across projects
- **Enterprise Compliance** — Full access control, immutable audit trails, and retention policy management for GDPR, HIPAA, and EU AI Act
- **6 New MCP Tools** — `report_outcome`, `get_lifecycle_status`, `set_retention_policy`, `compact_memories`, `get_behavioral_patterns`, `audit_trail`
- **Improved Search** — Lifecycle-aware recall that automatically promotes relevant memories and filters stale ones
- **Performance Optimized** — Real-time lifecycle management and access control

### Changed
- Enhanced ranking algorithm with additional signals for improved relevance
- Improved search ranking using multiple relevance factors
- Search results include lifecycle state information

### Fixed
- Configurable storage limits prevent unbounded memory growth

---

## [2.7.6] - 2026-02-22

### Improved
- Documentation organization and navigation

---

## [2.7.4] - 2026-02-16

### Added
- Per-profile learning — each profile learns its own preferences independently
- Thumbs up/down and pin feedback on memory cards
- Learning data management in Settings (backup + reset)
- "What We Learned" summary card in Learning tab

### Improved
- Smarter learning from your natural usage patterns
- Recall results improve automatically over time
- Privacy notice for all learning features
- All dashboard tabs refresh on profile switch

---

## [2.7.3] - 2026-02-16

### Improved
- Enhanced trust scoring accuracy
- Improved search result relevance across all access methods
- Better error handling for optional components

---

## [2.7.1] - 2026-02-16

### Added
- **Learning Dashboard Tab** — View your ranking phase, preferences, workflow patterns, and privacy controls
- **Learning API** — Endpoints for dashboard learning features
- **One-click Reset** — Reset all learning data directly from the dashboard

---

## [2.7.0] - 2026-02-16

**Release Type:** Major Feature Release — "Your AI Learns You"

SuperLocalMemory now learns your patterns, adapts to your workflow, and personalizes recall. All processing happens 100% locally — your behavioral data never leaves your machine.

### Added
- **Adaptive Learning System** — Detects your tech preferences, project context, and workflow patterns across all your projects
- **Personalized Recall** — Search results automatically re-ranked based on your learned preferences. Gets smarter over time.
- **Zero Cold-Start** — Personalization works from day 1 using your existing memory patterns
- **Multi-Channel Feedback** — Tell the system which memories were useful via MCP, CLI, or dashboard
- **Source Quality Scoring** — Learns which tools produce the most useful memories
- **Workflow Detection** — Recognizes your coding workflow sequences and adapts retrieval accordingly
- **Engagement Metrics** — Track memory system health locally with zero telemetry
- **Isolated Learning Data** — Behavioral data stored separately from memories. One-command erasure for full GDPR compliance.
- **3 New MCP Tools** — Feedback signal, pattern transparency, and user correction
- **2 New MCP Resources** — Learning status and engagement metrics
- **New CLI Commands** — Learning management, engagement tracking, pattern correction
- **New Skill** — View learned preferences in Claude Code and compatible tools
- **Auto Python Installation** — Installer now auto-detects and installs Python for new users

---

## [2.6.5] - 2026-02-16

### Added
- **Interactive Knowledge Graph** — Fully interactive visualization with zoom, pan, and click-to-explore
- **Mobile & Accessibility Support** — Touch gestures, keyboard navigation, and screen reader compatibility

---

## [2.6.0] - 2026-02-15

**Release Type:** Security Hardening & Scalability — "Battle-Tested"

### Added
- **Rate Limiting** — Protection against abuse with configurable thresholds
- **API Key Authentication** — Optional authentication for API access
- **CI Workflow** — Automated testing across multiple Python versions
- **Trust Enforcement** — Untrusted agents blocked from write and delete operations
- **Advanced Search Index** — Faster search at scale with graceful fallback
- **Hybrid Search** — Combined search across multiple retrieval methods
- **SSRF Protection** — Webhook URLs validated against malicious targets

### Improved
- Higher memory graph capacity with intelligent sampling
- Hardened profile isolation across all queries
- Bounded resource usage under high load
- Optimized index rebuilds for large databases
- Sanitized error messages — no internal details leaked
- Capped resource pools for stability

---

## [2.5.1] - 2026-02-13

**Release Type:** Framework Integration — "Plugged Into the Ecosystem"

### Added
- **LangChain Integration** — Persistent chat history for LangChain applications
- **LlamaIndex Integration** — Chat memory storage for LlamaIndex
- **Session Isolation** — Framework memories tagged separately from normal recall

---

## [2.5.0] - 2026-02-12

**Release Type:** Major Feature Release — "Your AI Memory Has a Heartbeat"

SuperLocalMemory transforms from passive storage to active coordination layer. Every memory operation now triggers real-time events.

### Added
- **Reliable Concurrent Access** — No more "database is locked" errors under multi-agent workloads
- **Real-Time Events** — Live event broadcasting across all connected tools
- **Subscriptions** — Durable and ephemeral event subscriptions with filters
- **Webhook Delivery** — HTTP notifications with automatic retry on failure
- **Agent Registry** — Track connected AI agents with protocol and activity monitoring
- **Memory Provenance** — Track who created or modified each memory, and from which tool
- **Trust Scoring** — Behavioral trust signals collected per agent
- **Dashboard: Live Events** — Real-time event stream with filters and stats
- **Dashboard: Agents** — Connected agents table with trust scores and protocol badges

### Improved
- Refactored core modules for reliability and performance
- Dashboard modernized with modular architecture

---

## [2.4.2] - 2026-02-11

### Fixed
- Profile isolation bug in dashboard — graph stats now filter by active profile

---

## [2.4.1] - 2026-02-11

### Added
- **Hierarchical Clustering** — Large knowledge clusters auto-subdivided for finer-grained topic discovery
- **Cluster Summaries** — Structured topic reports for every cluster in the knowledge graph

---

## [2.4.0] - 2026-02-11

**Release Type:** Profile System & Intelligence

### Added
- **Memory Profiles** — Single database, multiple profiles. Switch instantly from any IDE or CLI.
- **Auto-Backup** — Configurable automatic backups with retention policy
- **Confidence Scoring** — Statistical confidence tracking for learned patterns
- **Profile Management UI** — Create, switch, and delete profiles from the dashboard
- **Settings Tab** — Backup configuration, history, and profile management
- **Column Sorting** — Click headers to sort in Memories table

---

## [2.3.7] - 2026-02-09

### Added
- `--full` flag to show complete memory content without truncation
- Smart truncation for large memories

### Fixed
- CLI `get` command now retrieves memories correctly

---

## [2.3.5] - 2026-02-09

### Added
- **ChatGPT Connector** — Search and fetch memories from ChatGPT via MCP
- **Streamable HTTP Transport** — Additional transport option for MCP connections
- **Dashboard Enhancements** — Memory detail modal, dark mode, export, search score visualization

### Fixed
- Security improvement in dashboard event handling

---

## [2.3.0] - 2026-02-08

**Release Type:** Universal Integration

SuperLocalMemory now works across 16+ IDEs and CLI tools.

### Added
- **Auto-Configuration** — Automatic setup for Cursor, Windsurf, Claude Desktop, Continue.dev, Codex, Copilot, Gemini, JetBrains
- **Universal CLI** — `slm` command works in any terminal
- **Skills Installer** — One-command setup for supported editors
- **Tool Annotations** — Read-only, destructive, and open-world hints for all MCP tools

---

## [2.2.0] - 2026-02-07

**Release Type:** Feature Release — Advanced Search

### Added
- **Advanced Search** — Faster, more accurate search with multiple retrieval strategies
- **Query Optimization** — Spell correction, query expansion, and technical term preservation
- **Search Caching** — Frequently-used queries return near-instantly
- **Combined Search** — Results fused from multiple search methods for better relevance
- **Fast Vector Search** — Sub-10ms search at scale (optional)
- **Local Embeddings** — Semantic search with GPU acceleration (optional)
- **Modular Installation** — Install only what you need: core, UI, search, or everything

---

## [2.1.0-universal] - 2026-02-07

**Release Type:** Major Feature Release — Universal Integration

### Added
- **6 Universal Skills** — remember, recall, list-recent, status, build-graph, switch-profile
- **MCP Server** — Native IDE integration with tools, resources, and prompts
- **Attribution Protection** — Multi-layer protection ensuring proper credit
- **11+ IDE Support** — Cursor, Windsurf, Claude Desktop, Continue.dev, Cody, Aider, ChatGPT, Perplexity, Zed, OpenCode, Antigravity

---

## [2.0.0] - 2026-02-05

### Initial Release — Complete Rewrite

SuperLocalMemory V3 represents a complete architectural rewrite with intelligent knowledge graphs, pattern learning, and enhanced organization.

### Added
- **4-Layer Architecture** — Storage, Hierarchical Index, Knowledge Graph, Pattern Learning
- **Automatic Entity Extraction** — Discovers key topics and concepts from your memories
- **Intelligent Clustering** — Automatic thematic grouping of related memories
- **Pattern Learning** — Tracks your preferences across frameworks, languages, architecture, security, and coding style
- **Storage Optimization** — Progressive compression reduces storage by up to 96%
- **Profile Management** — Multi-profile support with isolated data

---

## Versioning

We use [Semantic Versioning](https://semver.org/):
- **MAJOR:** Breaking changes (e.g., 2.0.0 → 3.0.0)
- **MINOR:** New features (backward compatible, e.g., 2.0.0 → 2.1.0)
- **PATCH:** Bug fixes (backward compatible, e.g., 2.1.0 → 2.1.1)

**Current Version:** v3.3.0
**Website:** [superlocalmemory.com](https://superlocalmemory.com)
**npm:** `npm install -g superlocalmemory`

---

## License

SuperLocalMemory V3 is released under the [Elastic License 2.0](LICENSE).

---

**100% local. 100% private. 100% yours.**
