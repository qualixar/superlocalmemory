# CLI Reference

The `slm` command has 74 top-level commands, plus `slm hook`, which hosts call from their hook configuration and which is not part of the normal command parser. This page groups them by task, with one short table per group. The installed help always wins over this page: `slm --help` lists every command, `slm <command> --help` lists its flags, and `slm help` prints a grouped overview with the topics `modes`, `config` and `self-heal`. Everything here was checked against the code and its `--help` output.

## How the CLI behaves

Most commands talk to a background daemon and start it if it is not running. These commands never auto-start it, because they only read or write files: `setup`, `mode`, `provider`, `connect`, `upgrade-hosts`, `migrate`, `mcp`, `warmup`, `hooks`, `codex`, `config`, `evolve`, `db`, `disable`, `enable`, `clear-cache`, `reconfigure`, `benchmark`, `rotate-token`, `remote`, `evidence`, `brain`, `diagnostics`, `wrap`, `optimize`, `cache`, `compress`, `help-optimize`, `loop`, `help`, `serve`, `restart`, `gdpr`, `summary`, `backup`, `models`, `media` and `features`. Metadata options (`--help`, `--version`) create no data folder and start nothing.

The data folder is `~/.superlocalmemory` unless `SLM_DATA_DIR`, `SL_MEMORY_PATH` or `SLM_HOME` says otherwise, in that order. See [Installation](Installation#where-your-data-lives).

**Exit codes.** Most commands exit 0 on success, 1 when the command ran and failed, and 2 when it refused to run (an unknown kind, a rejected `--replaces`, a `forget` with no query, a repair aimed at the wrong folder). A few commands document more: `slm gdpr verify` exits 0 for an intact receipt, 1 for a tampered one and 2 when the receipt is not found. `slm remote check` exits 1 on any failure. `slm warmup` exits non-zero when recall is not ready. `slm media enable` exits 3 when the daemon is not running and 4 when the computer has too little memory for images and documents.

## Setup, configuration and status

| Command | What it does | Notable flags |
|---|---|---|
| `slm init` | One-command setup: runs the wizard if SLM is not configured, installs Claude Code hooks, connects detected IDEs and warms the embedding model | `--force` redo setup, `--auto` no terminal needed (Mode A plus hooks), `--gate` experimental PreToolUse gate |
| `slm setup` | The interactive first-time wizard (ten steps, see [Installation](Installation#the-setup-wizard)) | `--auto` use defaults, for scripts and CI; `--no-laya` do not install the on-device answer check (installed by default on Apple Silicon) |
| `slm mode [a, b, c]` | Show or set the operating mode; run `slm restart` to apply a change ([Modes Explained](Modes-Explained)) | `--json` |
| `slm provider [set] [PROVIDER]` | Show or set the language-model provider for Mode B and C. Providers: openai, anthropic, azure, ollama, openrouter, custom | `--endpoint URL`, `--key KEY`, `--model NAME`, `--mode b or c` (all for `custom`; `--mode` defaults to c) |
| `slm models` | Installed Ollama models, recommendations and the hosted catalogue. Read-only | `--json` |
| `slm embedder switch, status, rollback, cancel, forget-previous, calibrate, calibration-status, calibration-cancel` | Change the text embedding model in the background without stopping SLM, or calibrate the answer check on your store ([Embedding Models](Embedding-Models)). `upgrade` is still accepted but is not offered: text stays on its own model | `switch MODEL`: `--dimension`, `--provider` (sentence-transformers, ollama, openai or slm-media), `--endpoint`, `--task-prefixes`, `--no-wait`, `--force`; `calibrate`: `--model` (llama3.2:latest or gemma3:4b), `--samples`; all: `--json` |
| `slm config get KEY` and `slm config set KEY VALUE` | Read any `config.json` value in dot notation, or set one of a short allowed list | `--json` |
| `slm status` | Mode, provider, profile, data folder, database size, and whether the daemon is holding saves back | `--verbose` or `-v` adds migration log, daemon port and last version; `--json` |
| `slm health` | Health of the scoring, consistency and lifecycle layers ([Living Brain](Living-Brain)) | `--json` |
| `slm doctor` | Pre-flight check of dependencies, embedding worker, daemon, database and installed hosts | `--quick` fast checks only, `--deep` read every database page, `--fix` repair what can be repaired first, `--refile-hidden` list memories that are hidden but scored to keep (with `--yes`, re-file them), `--json` |
| `slm warmup` | Load the embedding model (about 500 MB the first time) and confirm recall is ready | `--timeout SECONDS` (default 120; 0 checks once) |
| `slm help [TOPIC]` | Grouped overview of all commands. Topics: modes, config, self-heal | none |
| `slm dashboard` | Open the web dashboard ([Dashboard Tour](Dashboard-Tour)) | `--port PORT` (default 8765) |

`slm doctor` always checks that the daemon is available, including with `--quick`: it reads the daemon's descriptor and makes at most one health request with a 0.5-second limit, and never starts the daemon. A missing descriptor is a warning that points at `slm serve start`; a recorded process that is dead, or an unhealthy or invalid daemon, is a failure that points at `slm restart`. The daemon check is part of the `--json` output too. Doctor exits 0 when every check passed or only warned and 1 when any check failed, and it prints the whole report before it exits. With Memory health on, it adds a **Memory description** row and a **Memory repairs** row ([Health and Repair](Health-and-Repair)). Two checks to know: **Failed saves** warns when saves were given up on after repeated attempts and fails at 50 or more (`slm enrich fix repair-saves` retries them), and **Answer check fit** warns when the text search runs on a model the answer check is not fitted for, with `slm embedder rollback` as the way back ([Embedding Models](Embedding-Models)).

`slm config set` accepts only these keys and refuses the rest with a `DISALLOWED_KEY` error that lists them: `evolution.enabled`, `evolution.backend`, `evolution.max_evolutions_per_cycle`, `evolution.mutation_model`, `evolution.verify_model`, `evolution.confirm_model`, `mesh_enabled`, `daemon_idle_timeout`, `entity_compilation_enabled`, `graph_backend`, `vector_backend`, `scope.default_scope`, `scope.recall_include_global` and `scope.recall_include_shared`. Everything else has its own command or lives in the dashboard Settings page ([Configuration](Configuration)).

## The daemon and the MCP server

| Command | What it does | Notable flags |
|---|---|---|
| `slm serve [start, stop, status, install, uninstall]` | Control the background daemon. `start` is the default. `install` adds a user-level service that starts it at login: a LaunchAgent on macOS, a systemd user service on Linux, a Task Scheduler task on Windows | none |
| `slm restart` | Kill orphan processes, clear stale state, start fresh and check health. Needed after changes that cannot apply at runtime, such as a new mode | `--dashboard` open the dashboard afterwards, `--json` |
| `slm mcp` | Start the MCP server on stdio. IDE configurations run it as a subprocess; you rarely run it by hand ([MCP Tools](MCP-Tools)) | none |
| `slm reap` | Find orphaned SLM processes. The default only lists them | `--force` kill orphans, `--all` kill every `slm mcp` process (after switching IDE), `--json` |
| `slm rotate-token` | Rotate the install token. Run `slm restart` afterwards | none |
| `slm token show` | Print the SuperLocalMemory key, readable only by you | none |

## Memory operations

| Command | What it does | Notable flags |
|---|---|---|
| `slm remember CONTENT` | Store a memory. The default receipt returns once the memory is searchable by its wording; enrichment continues in the background ([How Remember Works](How-Remember-Works)) | `--tags a,b`, `--kind KIND`, `--replaces ID`, `--sync` wait for completion, `--scope personal, shared or global`, `--shared-with PROFILES`, `--json` |
| `slm recall QUERY` (alias `slm search`) | Multi-channel retrieval ([How Recall Works](How-Recall-Works)) | See the recall flags below |
| `slm trace QUERY` | Recall with a per-channel score breakdown | `--limit N` (default 10), `--json` |
| `slm list` | Recent memories, newest first, with each one's kind and the id that `update` and `delete` take | `-n` or `--limit N` (default 20), `--kind`, `--tag` (repeatable), `--tags-match all or any`, `--json` |
| `slm update FACT_ID CONTENT` | Replace the text of one memory. The corrected fact is re-indexed everywhere. A review-gated change returns a correction case id | `--json` |
| `slm delete FACT_ID` | Delete one memory by exact id | `--yes` or `-y`, `--json` |
| `slm outcome success, partial or failure FACT_ID...` | Tell SLM how using recalled memories went; it is the command-line twin of the `report_outcome` tool ([Active Memory](Active-Memory)) | `--query-id ID` (from `slm recall --json`), `--note TEXT`, `--agent NAME`, `--report-id KEY` (a retry with it records nothing new), `--json` |
| `slm hide FACT_ID...` | Keep memories out of recall answers and session starts without deleting them; your decision, which no automatic rule changes | `--undo` (show them again), `--json` |
| `slm forget QUERY` | Delete every memory whose text contains the query (case-insensitive), after a confirmation | `--dry-run` preview only (with no query it previews everything), `--yes` or `-y`, `--json` |
| `slm observe [CONTENT]` | Offer content for automatic capture; SLM decides whether it holds a decision, bug fix or preference worth storing. Reads a pipe or an argument | none |

`--kind` takes one of the nine [memory kinds](Memory-Kinds): rule, decision, status, procedure, prospective, opinion, correction, episodic or semantic. `--replaces` names an earlier memory this one supersedes; the old one stops being returned, and `review_correction` undoes it.

**The `forget` naming collision.** The CLI command `slm forget QUERY` deletes memories whose text matches the query. The MCP tool `forget` does something different: it runs the Ebbinghaus decay cycle, which moves memories between lifecycle zones and is a dry run by default. It deletes nothing by query. Use `slm delete` or the MCP `delete_memory` tool to remove a single memory by id.

The `recall` flags narrow a search; filters that match nothing are reported rather than ignored.

| Flag | Meaning |
|---|---|
| `--limit N` | Maximum results (default 20) |
| `--window SPAN` | Event-time range: a relative span (24h, 7d, 30d, 1y) or `2026-07-01..2026-07-31` |
| `--project P`, `--project-strict`, `--prefer-project P` | Only that project's memories (not narrowed if none match, unless strict), or rank them higher without removing others |
| `--saved-by AGENT`, `--about NAME`, `--kind KIND` | Only memories saved by an agent, mentioning a name, or of a kind |
| `--tag LABEL` (repeatable), `--tags-match all or any` | Only memories with the tag or tags |
| `--as-of T`, `--known-as-of T`, `--valid-at T`, `--include-unknown` | Point-in-time recall (ISO-8601); `--include-unknown` adds memories whose temporal provenance is unknown |
| `--include-global`, `--no-global`, `--include-shared`, `--no-shared` | Include or exclude other profiles' global and shared memories ([Profiles](Profiles)) |
| `--depth 0, 1 or 2` | How much of each memory to return: 0 a glance (ids and one-line previews), 1 the full memory (default), 2 with its source memory, same-session neighbours, memories about the same entities and the correction that replaced it |
| `--max-tokens N` | Cap the returned results at about this many tokens; never exceeded |
| `--show-weak` | When nothing supports the question, show the closest weak matches labelled `[weak]` instead of nothing |
| `--fast` | Skip the internal agentic verification round |
| `--json` | Structured output |

The CLI shows related memories, labelled `[related]`, when nothing answers the question exactly, and prints "That isn't in SuperLocalMemory." when nothing is close. While SLM is still warming up it prints "SuperLocalMemory is still warming up — results may be incomplete. Try again in a few minutes." instead of that. With [Answer Check](Answer-Check) running, recall output also carries one line saying whether the top results answer the question ([How Recall Works](How-Recall-Works)).

## Kinds, summaries, saved views and corrections

| Command | What it does | Notable flags |
|---|---|---|
| `slm kinds status` | Counts per kind, the backend in use, and any classification run | `--json` |
| `slm kinds settings` | Show or change memory-kind settings | `--enable` or `--disable`, `--backend auto, rules, laya, jev, llm or off`, `--jev-consent yes or no`, `--standing-rules on or off` |
| `slm kinds backfill start, pause, resume, cancel, revert, status` | Classify existing memories as a background, undoable run (`revert RUN_ID`) | `--json` |
| `slm kinds set FACT_ID KIND` | Set and confirm one memory's kind | `--json` |
| `slm kinds review` and `slm kinds confirm` | List suggestions waiting for confirmation, and confirm 1 to 200 at once (`FACT_ID` or `FACT_ID=KIND`) | `review`: `--kind`, `--limit` |
| `slm summary session ID, sessions, day [DATE], project [PATH]` | Readable, traceable summaries; extractive unless a model is configured. No daemon needed | `--profile P`, `--json` |
| `slm view list, create NAME QUERY, run NAME, show NAME, rename NAME NEW, delete NAME` | Saved views: named recall queries you can re-run. Deleting a view touches no memory | `create`: `--kind`, `--window`, `--as-of`, `--limit` (1 to 50) |
| `slm review-correction CASE_ID apply, reject or rollback EXPECTED_VERSION` | Resolve a review-gated correction case; the version is a compare-and-swap guard | `--event-valid-until T` (apply only), `--json` |
| `slm corrections overtaken` and `slm corrections restore-overtaken CASE_ID` | List correction cases that your own later delete, replace or edit closed, and put one back | `--limit` (1 to 500), `--profile`, `--json` |

See [Memory Kinds](Memory-Kinds) for what the nine kinds mean and how suggestions are confirmed.

## Pictures, documents, folders and other sources

| Command | What it does | Notable flags |
|---|---|---|
| `slm features` | What is on and what you can turn on: images and documents, folder sources, bot messages | `--json` |
| `slm media enable, disable, status, gc, repair` | Turn images and documents on or off, check set-up, find leftovers, and make pictures findable by what they show ([Pictures and Documents](Pictures-and-Documents)) | `enable`: `--yes`; `disable`: `--remove-files`; `gc`: `--apply`; `repair`: `--dry-run`; all: `--json` |
| `slm sources add PATH, list, report ID, rescan ID, remove ID, forget-empty ID` | Connect folders and Obsidian vaults. Sources are read, never written | `add`: `--kind folder or obsidian`, `--yes`; `remove`: `--purge` also erases the memories it gave, `--yes`; `forget-empty`: `--yes` |
| `slm adapters list, enable, disable, start, stop, status [NAME]` | Manage ingestion adapters: gmail, calendar, transcript ([Ingestion Adapters](Ingestion-Adapters)) | none |
| `slm ingest` | Import external observations into SLM learning | `--source ecc or jsonl`, `--file FILE`, `--dry-run`, `--json` |

## Profiles

| Command | What it does | Notable flags |
|---|---|---|
| `slm profile list, switch NAME, create NAME` | List memory profiles, change the active one, or create one. Deleting a profile is a dashboard action, not a CLI command ([Profiles](Profiles)) | `--json` |

## Data, integrity, repair, backup and restore

| Command | What it does | Notable flags |
|---|---|---|
| `slm db migrate` | Apply additive, forward-only schema migrations. They also apply automatically at startup | `--status` inspect, `--dry-run` preview |
| `slm db integrity` | Read-only store check: pages, relations, fidelity, projections, repair status. Safe while SLM runs ([Health and Repair](Health-and-Repair)) | `--pages` read every page (slow), `--json` |
| `slm db repair` | Fix orphan rows, stale or orphan vectors, erased-text leftovers and settled obligations, with receipts. Preview by default | `--apply`, `--undo RUN_ID`, `--root FOLDER` (required to change anything), `--batch-size`, `--pause-ms`, `--max-seconds`, `--json` |
| `slm db repair-entities` | Undo wrong entity links left by the older fuzzy name merge. Preview by default; applying takes a backup first and writes an undo record | `--apply`, `--undo RUN_ID`, `--root FOLDER` (required to change), `--json` |
| `slm db fidelity` | List facts that no longer say what their memory said (a number, a date, a dropped "never") | `--profile`, `--limit`, `--withhold FACT_ID` and `--release FACT_ID` (SLM must be stopped), `--json` |
| `slm db regraph` | Re-derive the graph copy of your memories from the store | `--check` report drift only, `--profile` |
| `slm db reembed` | Backfill embeddings for facts that never got one | `--all-profiles`, `--limit N`, `--json` |
| `slm db compact` | Drop old LanceDB vector-store versions. The background job already keeps only an hour of history (`SLM_VECTOR_HISTORY_DAYS` changes it); use this with SLM stopped and `--offline` to shrink a store that already grew | `--offline` (daemon stopped; also removes unverified files) |
| `slm db scale status, adopt, prepare, verify, promote, rollback` | Stage and promote the optional CozoDB and LanceDB projections; SQLite stays canonical | `--stage-id ID` (verify, promote), `--backup-id ID` (rollback) |
| `slm db restore-points` | List the copies of your memories you can go back to | `--json` |
| `slm db restore POINT_ID` | Go back to a restore point, applied when SLM next starts ([Host Upgrades](Host-Upgrades)) | `--yes`, `--cancel`, `--no-reimport`, `--json` |
| `slm db prepare-downgrade` | Get your memories ready to run on the previous SLM version | `--yes`, `--cancel`, `--json` |
| `slm migrate` | A one-shot migrator for data from a very old install; a current install does not need it. Not the same as `slm db migrate` | `--rollback` only while the migrator's own backup still exists |
| `slm evidence export DEST, verify BUNDLE, import BUNDLE, rebuild` | Versioned, checksummed JSONL evidence bundles. `import` and `rebuild` are dry runs unless `--execute` | `export`: `--profile`; `import`: `--profile`, `--replace` or `--merge` (add to a profile that has memories; existing rows are kept), `--rollback-dir`, `--execute`; all: `--json` |
| `slm backup status, recovery-key, decrypt FILE` | Cloud-backup encryption: state, the recovery key, and turning a downloaded backup into a plain `.db` file ([Cloud Backup](Cloud-Backup)) | `recovery-key --import` reads a key from stdin; `decrypt`: `-o OUT`, `--recovery-key-stdin`, `--force`; all: `--json` |
| `slm backup restore-safety-copy` | Put back the safety copy that a failed dashboard restore kept. The service must be stopped | `--id POINT` restore a different recovery backup, `--json` |
| `slm backup clear-recovery-needed` | Let the service start again after you repaired the stores yourself. Deletes the pre-restore copy the marker kept | `--json` |
| `slm clear-cache` | Wipe regenerable caches. `memory.db` and `learning.db` are kept | none |
| `slm enrich status, pause RUN, resume RUN, undo [RUN], fix REPAIR` | Memory health: the automatic description of older memories, the repairs that wait for your say-so (`entity-links`, `refile-hidden`, `repair-saves`) and undo ([Health and Repair](Health-and-Repair)) | `undo` and `fix`: `--yes` (needed without a terminal); `fix`: `--include-undone`; all: `--json` |

Restore points are a command-line and API feature; the dashboard has no restore-point screen. The dashboard restores a different thing, a recovery backup, from Backup & Cloud ([Cloud Backup](Cloud-Backup)).

## Governance, GDPR and teams

| Command | What it does | Notable flags |
|---|---|---|
| `slm gdpr status` | Compliance posture: receipts, audit events, known gaps. Read-only | `--profile`, `--json` |
| `slm gdpr export` | Subject-access and portability export for one profile | `--profile NAME` (required), `--output FILE` (default stdout), `--json` |
| `slm gdpr erase` | Irreversibly erase a profile. Needs both `--profile` and `--yes`; preview with `--dry-run` | `--profile`, `--yes`, `--dry-run`, `--json` |
| `slm gdpr verify` | Check the HMAC of an erasure receipt (exit 0 intact, 1 tampered, 2 not found) | `--receipt-id ID`, `--profile`, `--json` |
| `slm team status` | Whether login is required and how many users exist | `--json` |
| `slm team policy` | Require login for this workspace, or stop requiring it | `--require-login on or off` |

`slm gdpr` offers the controls a GDPR programme needs; whether a deployment is compliant depends on how it is configured and operated. See [Governance](Governance) and [Security, Company Mode and Teams](Security-Company-Mode-and-Teams). `slm team` runs on the SLM computer and acts as the break-glass path for company mode.

## Mesh and remote access

| Command | What it does | Notable flags |
|---|---|---|
| `slm mesh status` and `slm mesh peers` | Local agent-mesh broker health and statistics, or the active peer sessions ([Multi-Agent Memory](Multi-Agent-Memory), [Bots That Talk](Bots-That-Talk)) | none |
| `slm remote tls init` | Create a certificate authority and a server certificate for remote access | `--name HOST` and `--ip ADDR` (repeatable), `--days N` (at most 397), `--force` |
| `slm remote enable` and `slm remote disable` | Turn the TLS listener on or off; takes effect on restart | `enable --listen HOST:PORT` |
| `slm remote keys add NAME` | Create a key, shown once, bound to exactly one profile | `--read-only` recall only, `--profile P` (default: the active profile) |
| `slm remote keys list, allow KEY mesh or media, disallow KEY mesh or media, revoke KEY` | List keys (never their secrets), grant or remove the optional mesh and media abilities, or revoke a key on its next request | `list`: `--json` |
| `slm remote check` | Check remote access; exits 1 on any failure | `--json` |

`slm remote` serves other computers on your network and is separate from [Web Access and Connected Apps](Web-Access-and-Connected-Apps), which connects AI apps on the internet from the dashboard. A key's holder can read the full text of every memory in its profile, so create keys sparingly.

## Optimize

| Command | What it does | Notable flags |
|---|---|---|
| `slm optimize status, on, off, savings` | Show or switch the Optimize cache and compression settings, and report token and cost savings ([Optimize](Optimize)) | `savings`: `--since DAYS` (default 7), `--provider anthropic, openai or gemini`; all: `--json` |
| `slm cache status, clear, invalidate, ttl, semantic` | Cache state, clearing, tag invalidation, TTLs, and the semantic cache switch | `--tenant T` (default `default`), `invalidate --tag TAG`, `ttl --set SECONDS --semantic SECONDS`, `semantic on or off`, `--json` |
| `slm compress status, mode, prose, code, ccr, align` | Compression state and mode (`safe` or `aggressive`); `prose on or off`. `code`, `ccr` and `align` still parse but only print a notice | `--json` |
| `slm proxy` | Start the optimization proxy | `--port` (default 8765), `--provider anthropic, openai or gemini`, `--no-compress`, `--semantic`, `--json` |
| `slm wrap AGENT [ARGS]` | Point an agent at the proxy and launch it | `--list` registered agents, `--dry-run`, `--persistent` |
| `slm help-optimize [TOPIC]` | Full Optimize reference. Topics: cache, compress, optimize, proxy, agents, safety | `--no-pager` |

## Hooks and IDE integration

| Command | What it does | Notable flags |
|---|---|---|
| `slm connect [IDE]` | Write SLM's MCP entry into an IDE's config ([IDE Setup](IDE-Setup)) | `--list`, `--here` project scope, `--dry-run`, `--transport stdio, http or http-mcp-remote`, `--port` (default 8765), `--verify`, `--profile NAME` (writes `SLM_MCP_PROFILE`), `--cross-platform`, `--disable ADAPTER`, `--json` |
| `slm upgrade-hosts` | Preview, then refresh, installed host integrations after an SLM upgrade. Read-only unless `--apply` ([Host Upgrades](Host-Upgrades)) | `--host NAME` (repeatable), `--all-detected`, `--apply` |
| `slm hooks install, remove, status` | Manage additive auto-capture hooks for Claude Code or Codex. `status` is the default | `--agent claude or codex`, `--gate`, `--dry-run` |
| `slm codex install, remove, status` | Manage the Codex skills, subagents and lifecycle hooks | `--dry-run` |
| `slm hook EVENT` | Run one hook handler. Hosts call it; you do not | none |

`slm hook` takes one event name and bypasses the normal parser so it starts fast. The events are `mandate`, `start`, `gate`, `init-done`, `checkpoint`, `stop`, `user_prompt_submit`, `post_tool_async`, `post_tool_outcome`, `user_prompt_rehash`, `stop_outcome`, `auto_recall`, `topic_shift`, `before_web`, and the Codex events `codex-start`, `codex-prompt` and `codex-stop`. An unknown event prints an error and exits 1.

## Sessions and lifecycle

| Command | What it does | Notable flags |
|---|---|---|
| `slm session open` | Warm a session's context | `--project-path`, `--query`, `--max-results` (default 10) |
| `slm session close` | Close a session and create its temporal summaries | `--session-id` (default: the most recent) |
| `slm session-context [QUERY]` | Print relevant memories for hooks. Uses a fast SQLite path | `--max-age-days N` (default 30; 0 turns the age filter off), `--full`, `--json` |
| `slm decay` | Ebbinghaus forgetting cycle. Previews by default | `--execute` apply, `--dry-run`, `--profile`, `--json` |
| `slm quantize` | Embedding-precision cycle. Previews by default | `--execute`, `--dry-run`, `--profile`, `--repair-sketches`, `--json` |
| `slm consolidate` | The consolidation pipeline | `--cognitive` also run cognitive consolidation, `--dry-run`, `--profile`, `--json` |
| `slm soft-prompts` | List the auto-learned patterns that become soft prompts | `--profile`, `--json` |

`slm quantize` previews by default. `--repair-sketches` repairs only the compressed search sketches (the QJL ones) that `slm doctor` can recommend repairing: it rebuilds legacy or incompatible sketches from the original vectors that are still kept, and changes neither the stored vectors, nor the base compressed bytes, nor bit widths, nor lifecycle state. It starts no embedding model, a repeat skips sketches that are already compatible, and memories without a usable original vector are reported as needing to be embedded again. It reuses the existing rotation; if the rotation matrix is missing or incompatible, restore it from a backup before applying. The full `--execute` cycle also adjusts precision according to retention, and before it applies anything it prints counts for sketch repairs, downgrades, upgrades and memories moving to the forgotten state (the vectors are kept). The counts can change if writes happen meanwhile. Both forms accept `--profile` and `--json`; with `--json` the preview goes to the error stream and the final JSON carries it under `data.plan`.

The cycles and what they do are explained in [Living Brain](Living-Brain).

## Bounded loops, evolution and the brain

| Command | What it does | Notable flags |
|---|---|---|
| `slm loop demo` | Run the keyless convergence demo: a stub proposer and a gate that fails twice and passes on lap 3. A good post-upgrade check ([Bounded Loops](Bounded-Loops)) | `--iterations N` (default 10), `--json` |
| `slm loop history` | List recorded runs for a loop name | `--name NAME` (default `convergence-demo`), `--json` |
| `slm loop show RUN_ID` | Show every lap of one run | `--json` |
| `slm evolve` | Post-session skill evolution. The Stop hook normally runs it ([Skill Evolution](Skill-Evolution)) | `--session ID`, `--profile ID` |
| `slm brain [status]` | Read-only, profile-scoped Living Brain evidence summary | `--json` |

Running a loop against a real gate is done with the MCP tools `slm_loop_run`, `slm_loop_history` and `slm_loop_show`; the CLI has only `demo`, `history` and `show`.

## Diagnostics

| Command | What it does | Notable flags |
|---|---|---|
| `slm diagnostics export DEST` | Write a deterministic, content-free JSON report of local operational aggregates, for support | `--json` |
| `slm diagnostics reliability` | Ask whether wired mechanisms really work: has each learner moved off its prior, has each guarded path ever run | `--min-observations N` |
| `slm ops list, resolve OPERATION_ID, status` | List failed, stuck or degraded operations; retry, force-reconcile or cancel one (administrator action); quick failure count | `list`: `--profile`; `resolve`: `--action retry, force_reconcile or cancel`; all: `--json` |
| `slm benchmark` | Run the evo-memory benchmark against an isolated temporary database. It never touches your data | `--json` |

## Escape hatches

| Command | What it does | Notable flags |
|---|---|---|
| `slm disable` | Switch SLM off globally: writes `~/.superlocalmemory/.disabled` and stops the daemon | `--reason TEXT` |
| `slm enable` | Remove the `.disabled` marker and print the command that starts the daemon | none |
| `slm reconfigure` | Re-run the interactive post-install configurator (profile and install-time knobs) | extra flags pass through |

`slm clear-cache`, `slm reap` and `slm restart` are also escape hatches and are listed above. The environment variable `SLM_DISABLE=1` has the same effect as `slm disable` for one shell.

## Agent-native JSON output

Commands that list `--json` in their help print one compact JSON object on a single line. The envelope is the same everywhere: `success`, `command`, `version`, then `data` on success or `error` on failure, with optional `metadata` and `next_actions` that name follow-up commands.

```json
{
  "success": true,
  "command": "recall",
  "version": "<installed-version>",
  "data": {
    "results": [
      {
        "fact_id": "abc123",
        "content": "Database uses PostgreSQL 16",
        "relevance_score": 0.87,
        "ranking_score": 0.0132,
        "memory_confidence": 0.7,
        "rank_position": 1
      }
    ],
    "count": 1,
    "score_contract_version": "2",
    "calibration_status": "uncalibrated",
    "answer_confidence": null,
    "answer_check_status": "off"
  },
  "next_actions": [
    {"command": "slm list --json", "description": "List recent memories"}
  ]
}
```

Recall keeps ranking relevance separate from stored-memory confidence ([Retrieval Score Contract](Retrieval-Score-Contract)). The full recall response also carries `query_id`, `no_confident_match`, `abstained`, `abstention_reason`, `answerability`, `channel_status`, `incomplete_channels`, `project_scope`, `tag_scope`, `reranker_status` and the answer-check fields described in [Answer Check](Answer-Check). When `no_confident_match` is true or `answer_confidence` is low, the CLI help advises an agent to rewrite the query into one to three more specific sub-queries and recall again before concluding nothing was found.

```bash
slm recall "auth" --json | jq '.data.results[0].content'
slm list --json | jq '.data.results[].fact_id'
slm status --json | jq '.data.mode'
```

In CI, `slm status --json | jq -e '.success'` is a quick health gate, and `slm remember "Deployed $SHA" --json` records a deployment.

## CLI or MCP

| Need | Use | Example |
|---|---|---|
| IDE integration | MCP, in the profile that fits ([MCP Tools](MCP-Tools)) | `slm connect cursor --profile code` |
| Shell scripts and CI | CLI with `--json` | `slm recall "auth" --json` |
| Agent frameworks | CLI with `--json`, or the adapters in [Framework Adapters](Framework-Adapters) | `slm remember "..." --json` |
| Humans | CLI, readable output | `slm recall "auth"` |

Code: `cli/main.py` (parser and dispatch), `cli/commands.py`, `cli/json_output.py`, `cli/kinds_cmd.py`, `cli/summary_cmd.py`, `cli/view_cmd.py`, `cli/media_cmd.py`, `cli/sources_cmd.py`, `cli/remote_commands.py`, `cli/embedder_cmd.py`, `cli/integrity_cmd.py`, `cli/gdpr_cmd.py`, `cli/service_installer.py`, `hooks/hook_handlers.py`.
