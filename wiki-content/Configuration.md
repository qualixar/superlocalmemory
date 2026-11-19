# Configuration

SLM keeps all of its settings in plain files inside one data folder, and almost every setting has a command, a Settings group in the dashboard, or both, so you rarely edit JSON by hand. This page explains where the folder is, which file holds what, which Settings group changes which behaviour, and every environment variable that was confirmed in the source, with its default and where it is read.

## The data folder

The data folder is `~/.superlocalmemory/` unless you move it. The environment variables below are checked in this order and the first one that is set wins, followed by a base directory passed in explicitly, then a `base_dir` hint in an old `config.json`, then the default. The resolver only reads; it never moves data.

1. `SLM_DATA_DIR`
2. `SL_MEMORY_PATH` (older name)
3. `SLM_HOME` (older name)

Identity files (`.install_token`, `.signer_key`, `api_key`, `credentials.json`) are treated as durable across a move of the folder. If the folder you select and the default folder both already hold data, startup is blocked with a conflict error, and SLM will not merge or move either one for you. The reason is that two daemons serving two databases is how stores get damaged, so SLM makes you choose.

| File or folder | What it holds |
|---|---|
| `memory.db` | Memories, facts, entities, graph, scenes, access control, audit, evolution log, corrections. |
| `mesh.db` | The agent mesh: peers, messages, shared state, locks and peer settings. Activity is kept `mesh.retention_days` days (default 7); shared state and peer settings are kept. |
| `learning.db` | Learning signals, ranking state, receipts, Answer Check history, saved views. |
| `media.db`, `media/` | Pictures, documents, folder sources and the original files. |
| `config.json` | The active `SLMConfig`: mode, profile, LLM, embedding, retrieval, scope, forgetting, evolution and the rest. |
| `mode_a.json`, `mode_b.json`, `mode_c.json`, `current_mode` | A saved copy of the settings for each mode and the letter of the active one. See [Modes Explained](Modes-Explained). |
| `profiles.json` | The profile list and active profile, kept in step with the database. See [Profiles](Profiles). |
| `answer_check.json`, `memory_kinds.json` | Settings kept outside the per-mode copies so a consent cannot come back after a mode switch. |
| `optimize.json` | [Optimize](Optimize) settings, hot-reloaded every 2 seconds. |
| `code_graph_config.json` | Code-graph switches. |
| `features.json` | Whether images, documents and folder sources are turned on, and the 4.2.30 settings: one state per flag (`on`, `shadow` or `off`) and the kill switch. See [The 4.2.30 settings](#the-4230-settings). |
| `disclosures.json` | The measurement notice, the version you acknowledged, and each on or off change of it. |
| `consents.json` | Your agreement to send pairs of memories to the online judge. Not a feature flag, and no environment variable sets it. |
| `wiki.json`, `wiki/` | Where the living wiki writes its Markdown files, and the files themselves. |
| `adapters.json` | Ingestion adapter configuration. See [Ingestion Adapters](Ingestion-Adapters). |
| `backup_config.json`, `cloud-backup-encryption.json`, `backups/` | Backup schedule, encryption state and the local copies. See [Cloud Backup](Cloud-Backup). |
| `remote_keys.json` | Web Access keys (stored as digests). |
| `store-check.json` | Result of the memory-store check. |
| `config.toml` | Optional installer-written file. SLM reads only its `[deployment]` section. |
| `.disabled` | Global off switch, created by `slm disable`. |

## How the layers combine

Settings reach `config.json` from the Settings pane, from commands such as `slm mode`, `slm provider`, `slm embedder` and `slm config set`, or from you editing the file. `config.json` is written atomically and forced to owner-only permissions (0600) because it can hold a provider key. The environment variables in the tables below are read where they are used and are never written back into `config.json`, so a process-only override cannot leak into the file on a later save. Prefer the environment for secrets such as `SLM_CROSS_ENCODER_API_KEY` or the provider keys, because that avoids a plaintext copy on disk.

If `config.json` is unreadable or corrupt, SLM falls back to the Mode A default and logs a warning rather than refusing every command.

`slm config get <dot.key>` reads any value. `slm config set <dot.key> <value>` is deliberately limited to this list: `evolution.enabled`, `evolution.backend`, `evolution.max_evolutions_per_cycle`, `evolution.mutation_model`, `evolution.verify_model`, `evolution.confirm_model`, `mesh_enabled`, `daemon_idle_timeout`, `entity_compilation_enabled`, `graph_backend`, `vector_backend`, `scope.default_scope`, `scope.recall_include_global`, `scope.recall_include_shared`. `scale_engine_state` is excluded on purpose: it records what was done to the store, so writing it by hand could make the daemon skip or repeat a projection.

## The Settings pane

The Settings pane has a search box over every setting and 18 groups. "Save all changes" at the bottom clicks the Save button of the groups that have one, and each group says whether a daemon restart is needed.

| Group | What it changes |
|---|---|
| Operating Mode & LLM Provider | Mode A, B or C; provider (None, Ollama, OpenRouter, OpenAI, Anthropic), model, API key, base URL; a Test button. |
| Answer check | On this Mac (Laya), Online with Jev, or Off; setup, repair, test, remove. Under Online with Jev, "Background judges" shows the notice and the I agree and Stop buttons for the judges that ask Jev whether a newer memory replaces an older one. See [Answer Check](Answer-Check). |
| Help SLM measure what works | One On or Off switch for the withholding and swapping described below. |
| Turn off everything new in 4.2.30 | The kill switch, and a button that brings back the memories 4.2.30 retired after counting them and asking. |
| Embeddings & Retrieval | Default local model or a custom OpenAI-compatible endpoint (model, dimension, endpoint, key, Test); the background re-index panel; and, only for a store whose text an older release moved to the picture model, the memory-engine card with Roll back. See [Embedding Models](Embedding-Models). |
| Storage Backends | Graph backend `auto`, `sqlite` or `cozo`; vector backend `auto`, `sqlite-vec` or `lancedb`; base directory (read-only). Saving reports whether a restart is needed. |
| Memory Scope (auto-saves) | Default write scope (personal, shared, global) and whether recall includes shared and global memories. Both recall switches are off by default. |
| Recall & Memory Behaviour (auto-saves, no restart) | Recall depth (`retrieval.top_k`, 1 to 200, shown as 20), the reranker switch (about 200 MB of RAM and about 200 ms), memory injection into prompts, the core-memory block and its size (0 to 50, default 5). |
| Auto-Capture | Capture from agent sessions; capture decisions; capture bug fixes. |
| Auto-Recall | Inject relevant memory into agent context; recall on session start. |
| Auto-Invoke (FOK Threshold) | Run recall automatically above a relevance threshold (0 always, 1 never); ACT-R mode. |
| Forgetting Engine | Enable; protect core memories; archive and forget thresholds; run interval in minutes; the trust-based fade factor (kappa) with a decay preview. |
| Skill Evolution | Off by default. Enable, and choose a backend (`auto`, `claude`, `ollama`, `anthropic`, `openai`). It makes background model calls. See [Skill Evolution](Skill-Evolution). |
| Backup | Auto-backup on or off, daily or weekly, copies kept, Backup Now, link to Cloud Backup. |
| Mesh Network | `mesh.enabled`. |
| Trust & Source Quality | Weight recall by trust; prefer first-party agents; minimum trust to promote a memory. |
| Rate Limits | Write limit, read limit and window in seconds. Applied immediately; the local dashboard gets a multiple of the limits. |
| Daemon (port changes require restart) | API port, legacy port, whether the legacy port is open, idle timeout (0 = run forever). |

Not in Settings: the remote reranker (`config.json` only), retention policy and company mode (Governance pane), memory-kind settings (`slm kinds settings`), Optimize (its own pane), Web Access (`slm remote`), the code graph, turning on images and documents and the optional picture and document models (the Documents & Images pane), importing (the Bring your memories from another app card on the home pane), and the trust, truth, judge and wiki flags (`slm flags`). The limits you set under Rate Limits are saved to `config.json` and re-applied at daemon start.

## The 4.2.30 settings

Release 4.2.30 adds a set of switches that are not in `config.json`. They live in `features.json` in the data folder and are changed with `slm flags set <flag> on|shadow|off`, with `slm measurement`, or from the dashboard. `slm flags` lists every flag and its state and needs no daemon. An environment variable `SLM_FEATURE_<FLAG>` (each dot becomes `_`, upper case; for example `SLM_FEATURE_OUTCOMES_HOLDOUT=off`) overrides the file. `shadow` means the subsystem computes and records in its own tables, changes no memory and sends nothing off the computer, with one exception: in `delivery.need_gate` shadow, memories are still injected on the prompts its fixed rules let through. Subsystems read their flag at their next run, and `trust.signing` needs a daemon restart.

Four settings are on by default and the rest are off.

### What SLM measures

SLM measures whether memory helps your agents, on this computer and as ids and numbers only. `slm measurement status` shows the notice and each measure in plain words, and the same notice is printed on the first terminal run after an upgrade and at npm install.

| Flag | Default | What it does |
|---|---|---|
| `outcomes.receipts` | on | Writes a receipt each time SLM hands memories to an agent: memory ids, rank, score and the logged probability of each choice. Never text. |
| `outcomes.holdout` | on | In about 1 in 10 session starts, leaves out one eligible memory or switches off one retrieval channel for that fetch (MCP session start only), chosen by a logged random draw. Never pinned rules, standing rules, core blocks, explicit recall, cues or lease warnings. |
| `outcomes.swaps` | on | In about 1 in 20 of an agent's explicit recalls, two neighbouring results trade places. Nothing is left out or added. |
| `outcomes.usage` | on | Compares an agent's last reply with the memories it was given to see which were used. Ids and scores only. |

`slm measurement off` turns off `outcomes.holdout`, `outcomes.swaps` and `outcomes.usage` together, and `slm measurement off --receipts` turns the receipts off as well. The dashboard's Settings card "Help SLM measure what works" has one switch for the same thing. It is not a plugin setting: the withholding and swapping happen in the daemon and the session-start command, which a plugin's own options cannot reach. Inside an agent, the skill tells the agent to run `slm measurement off` when you ask, never on its own. `slm health --memory` (the Memory Health report) shows what these measurements found.

### Memory on every prompt, and reminders on cues

Both are on by default. `delivery.need_gate` registers per-prompt recall: the UserPromptSubmit hook (`slm hook auto_recall`) asks memory before the agent answers, but only when the prompt needs it. Fixed rules skip acknowledgements, greetings, arithmetic, translation requests and slash commands, and the rest are ranked by a simple score. The skip threshold is set from your own logged prompts so that at most 10% of the prompts that needed memory are skipped, and until enough prompts have a known outcome the score skips nothing. A memory the conversation already got is not shown again, and the hook adds nothing if it cannot answer within its time limit. `slm health --memory` shows how many prompts were skipped.

`delivery.cues` brings a memory back when the agent touches something it was saved for. A memory saved with cues (`remember(cues=["path:src/**/schema.py", "tool:Bash:git push", "error:database is locked", "repo:myproject", "date:2026-12-01"])` or `slm remember --cue path:...`) comes back right before or after the matching step, at most once per conversation. `slm cues list` shows them.

`delivery.harm_gate` only reports in this release: `slm health --memory` lists memories that measurably make the agent worse when included. It never removes a memory from what an agent sees.

### The kill switch

`slm flags kill on`, `SLM_4230_OFF=1` in the daemon's environment, `POST /api/flags/kill`, or the Settings card "Turn off everything new in 4.2.30" returns the store to the way it read before 4.2.30, without a downgrade. It stops every 4.2.30 background pass and makes reads ignore slots, source leases, disputes and taint, so recall results carry no dispute, source, origin or `why` notes. It does not undo retirements: `slm flags rollback-retirements [--dry-run]` says how many memories would come back and then brings them back as one recorded decision that can be undone.

### Trust: origin, the firewall, the secret screen and signing

All on by default since 4.2.30; set any one to `off` with `slm flags set <flag> off`.

| Flag | What it does |
|---|---|
| `trust.taint` | Records how each memory arrived and classes it, best first: `user_direct`, `user_confirmed`, `system`, `agent_inferred`, `tool_output`, `mesh_peer`, `web`, `imported_file`, `model_generated`. A memory saved before your store took 4.2.30 is `legacy` and is trusted as it was before, decided by its date; one saved later with no origin record is `unknown` and is not trusted. A memory's taint is the lowest class among it and everything it was built from. `shadow` records and shows the class. `on` also keeps text below `system` from becoming a pin, a rule, a core block or a session-start instruction by itself, and stops a lower-class memory from replacing a higher one. It becomes an instruction through one explicit, recorded promotion by any caller (`slm trust promote`, or the dashboard card Held back from instructions), which `slm trust withdraw` can take back. Everything from before the upgrade keeps working as it did, and a background pass records the origin of each old memory. |
| `trust.secret_screen` | On, every door that saves text replaces anything shaped like a credential with a typed placeholder such as `[REDACTED:GITHUB:ab12]`. The original is kept nowhere, and a save is refused if the screen cannot run. It recognises key formats, not sentences such as "the password is hunter2". |
| `trust.signing` | The daemon signs each memory with a key read once from your operating system's keychain and makes each new audit-chain link an HMAC under the same key. It detects a memory changed by something that has your files but not your login session; it does not stop a process running as you. No keychain means signing is unavailable, never a key file. On macOS the keychain may ask once to allow access, for example after Python is upgraded. `slm signing status` and `slm doctor --truth` report problems by id. |

`slm why FACT_ID` explains one memory: where it came from, whether the firewall holds it, its signature, its disputes and when agents were shown it.

### Disputes, leases and the judges

All on by default. `truth.slots` lets memories state the current value of one thing (`remember(slot="subject/relation", slot_value="...")`). A different value later opens a dispute, and recall results carry both sides with a next action. `truth.leases` records the fingerprint of the file a memory was saved from (`remember(source_file=...)`) and marks results whose file changed or went missing. `judge.ladder` sends typed questions to rules, the on-device judge and, only with your consent (`slm consent give judge_ladder_hosted`), the online Jev judge. A judge acts on a memory only after blind audit labels support it at a stated error rate; until then it only watches. `slm disputes` and `slm audit` settle and label from a terminal, and the dashboard has the same as plain-word cards.

### Proposals, the living wiki and the dream cycle

All on by default. `compiler.dream` lets the background cycle queue pages and exact-duplicate merges for review at most every six hours; it asks no model and applies nothing. `compiler.wiki` (`on`, or `shadow` to see what would be written) lets `slm wiki build` write approved pages as Markdown and `slm wiki sync` read your edits back as proposals. The folder is `SLM_WIKI_DIR`, or `dir` in `wiki.json`, or `wiki/` in the data folder. `compiler.auto_apply` applies only a set whose basis is a recorded certificate, so in practice a person or an agent decides each set. Every sentence of a page is a word-for-word piece of a memory or a fixed template. The kill switch reads both flags as off.

### Importing and the picture models

`portability.import` is on: `slm import` and the dashboard card can preview, run and undo the import of a ChatGPT, Claude, Mem0, Letta, OKF or Markdown file; an upload is capped at 64 MB. `media.captions` (a 0.5 GB download), `media.document_pack` (1.9 GB) and `media.regions` (no download) are on. On a computer with 16 GB of memory or more, SLM sets up images and documents and then both models by itself while it is idle, one download at a time and never under memory pressure; anything you turned off stays off. Below 16 GB it is one click (`slm media enable`, `slm media pack enable captions|documents|regions`, or the dashboard), with a note that it may be slow. `SLM_NO_AUTO_DOWNLOAD=1` stops the automatic download on a computer. See [Pictures and Documents](Pictures-and-Documents) and the [import guide](https://github.com/qualixar/superlocalmemory/blob/main/docs/importing.md).

## Environment variables

Every variable below was found by searching the source. The default is stated only where the code states it. "Source" is where the variable is read.

### Where SLM keeps things and who is allowed to write

| Variable | Effect | Default | Source |
|---|---|---|---|
| `SLM_DATA_DIR`, `SL_MEMORY_PATH`, `SLM_HOME` | Data folder, first one set wins. | `~/.superlocalmemory` | `infra/data_root.py` |
| `SLM_AGENT_ID` | Agent name attached to MCP calls on stdio. | `mcp_client` | `mcp/agent_context.py` |
| `SLM_USER_SESSION` | A dashboard user session the CLI sends to the daemon in `X-SLM-User-Session`, for governed workspaces. | unset | `cli/daemon.py` |
| `SLM_PII_REDACTION` | A truthy value turns on PII redaction when saving (also the `pii_redaction` config key and the deployment policy). | off | `memory_core/save_path.py` |
| `SLM_REQUIRE_CREDENTIALS` | `1` makes even loopback writes present a daemon capability, install token or API key. | off | `server/write_identity.py` |
| `SLM_REQUIRE_API_KEY_LOOPBACK` | When set and an `api_key` file exists, loopback writes must also send a matching `X-SLM-API-Key`. | off | `infra/auth_middleware.py` |
| `SLM_DISABLE` | Any value other than `0`, `false`, `no` or `off` makes SLM do nothing, like the `.disabled` file. | unset | `core/slm_disabled.py` |
| `SLM_4230_OFF` | The 4.2.30 kill switch: turns every 4.2.30 setting off. See [The kill switch](#the-kill-switch). | unset | `core/feature_flags.py` |
| `SLM_FEATURE_<FLAG>` | Overrides one 4.2.30 flag (`on`, `shadow`, `off`); each dot becomes `_`, for example `SLM_FEATURE_OUTCOMES_HOLDOUT`. | the value in `features.json` | `core/feature_flags.py` |
| `SLM_WIKI_DIR` | Folder the living wiki writes to. | `dir` in `wiki.json`, else `wiki/` in the data folder | `compiler/wiki.py` |
| `SLM_NON_INTERACTIVE` | Makes the setup wizard treat the session as non-interactive. | unset | `cli/setup_wizard.py` |
| `SLM_PYTHON` | Python interpreter the npm installer uses; it must be 3.12 to 3.14. | searched automatically | `scripts/postinstall.js` |

### MCP

| Variable | Effect | Default | Source |
|---|---|---|---|
| `SLM_MCP_PROFILE` | Tool profile: `core`, `code`, `full`, `power`, `mesh`, `whole`. See [Profiles](Profiles). | the default set, equal to `full` | `mcp/server.py` |
| `SLM_MCP_TOOLS` | Comma-separated allowlist of tool names. | unset | `mcp/server.py` |
| `SLM_MCP_ALL_TOOLS` | `1` registers every tool. | unset | `mcp/server.py` |
| `SLM_MCP_MESH_TOOLS` | `1` or `true` exposes the mesh tools even if mesh is off in config. | follows `mesh_enabled` (on) | `mcp/server.py` |
| `SLM_MCP_STATEFUL` | `1` opts out of stateless Streamable HTTP and uses session ids. | stateless | `core/remote_mode.py` |
| `SLM_DISABLE_WARMUP_SIDE_EFFECTS` | `1` skips the dependency check and warm-up side effects at import. | unset | `__init__.py` |

### Daemon, memory and request limits

| Variable | Effect | Default | Source |
|---|---|---|---|
| `SLM_DAEMON_PORT` | Daemon port. | 8765 | `server/unified_daemon.py` |
| `SLM_DAEMON_HOST` (or `SLM_HOST`) | Address the daemon binds. Anything beyond loopback logs a warning unless `SLM_REQUIRE_CREDENTIALS=1`. | `127.0.0.1` | `server/unified_daemon.py` |
| `SLM_DAEMON_IDLE_TIMEOUT` | Seconds idle before the daemon stops; 0 never. Falls back to `daemon_idle_timeout`. | 0 | `server/unified_daemon.py` |
| `SLM_DISABLE_LEGACY_PORT` | `1` or `true` turns off the 8767 redirect. | on | `server/legacy_port.py` |
| `SLM_RSS_BUDGET_MB` | Total memory budget for the health monitor; 0 computes 40 percent of RAM. | 0 | `server/unified_daemon.py` |
| `SLM_MAX_WORKER_MB` | Per-worker size limit checked by the memory watchdog. | 2500 | `server/unified_daemon.py` |
| `SLM_EMBED_WORKER_RSS_LIMIT_MB` | Resident-size cap for the embedding worker. | 2500 | `core/embedding_worker.py` |
| `SLM_RATE_LIMIT_WRITE`, `SLM_RATE_LIMIT_READ`, `SLM_RATE_LIMIT_WINDOW` | Per-client request limits and window in seconds. Loopback gets `max(300, 10 x write)` writes and `max(2000, 20 x read)` reads. | 30, 120, 60 | `core/remote_mode.py` |
| `SLM_DB_BUSY_TIMEOUT_MS` | How long a database call waits for a writer. | 10000 | `storage/database.py` |
| `SLM_HOOK_RECALL_WORKERS` | How many agent sessions' per-prompt recalls the daemon serves at the same time, 1 to 4. One serves sessions strictly in turn. | 2 | `core/queue_consumer.py` |
| `SLM_LOOP_KILL` | Any non-empty value stops bounded loops. | unset | `loops/engine.py` |

### Web Access and remote use

| Variable | Effect | Default | Source |
|---|---|---|---|
| `SLM_REMOTE` | Opts the daemon into LAN and distributed mode. | off | `core/remote_mode.py` |
| `SLM_MCP_ALLOWED_HOSTS` | Comma-separated trusted clients: `*`, IP, CIDR or prefix wildcard. | unset | `core/remote_mode.py` |
| `SLM_ALLOWED_HOSTS` | Host names the daemon accepts in the `Host` header, as a DNS-rebinding guard; `*.suffix` allowed. | localhost and IP literals | `server/host_guard.py` |
| `SLM_REMOTE_LISTEN` | Address for the TLS remote listener; overrides the `listen` value in `remote/remote.json`. | unset | `server/remote_listener.py` |
| `SLM_REMOTE_TLS_CERT`, `SLM_REMOTE_TLS_KEY` | Certificate and key for that listener. | unset | `server/remote_listener.py` |
| `SLM_REMOTE_ALLOW_PLAINTEXT` | `1` allows remote callers over plain HTTP; SLM logs that keys and memory are then readable on the network. | off | `server/remote_access.py` |

### Retrieval, ranking and caching

| Variable | Effect | Default | Source |
|---|---|---|---|
| `SLM_CROSS_ENCODER_API_KEY` | Bearer token for a remote rerank endpoint; overrides the config value. | unset | `retrieval/remote_reranker.py` |
| `SLM_REMOTE_RERANK_DEADLINE_S` | Time limit for one remote rerank call, retries included. | 1.0 | `retrieval/remote_reranker.py` |
| `SLM_RERANKER_ENABLED` | Process-only override of `use_cross_encoder`; `0`, `false`, `no`, `off` disable. | the config value | `core/config.py` |
| `SLM_RANKING` | Adaptive ranking: `v1`, `v2` or `v2-ensemble`. | off | `core/recall_pipeline.py` |
| `SLM_RECALL_NO_SUPPORT_GATE` | `1`, `true` or `yes` turns the support check off and restores the older recall behaviour; the setting `retrieval.answer_support_gate` set to false does the same. | on | `retrieval/support_gate.py` (`switched_off`) |
| `SLM_SKIP_LAYA` | `1`, `true`, `yes` or `on` stops setup and upgrades installing the on-device answer check; `slm setup --no-laya` does the same for one run. | off | `cli/laya_offer.py` (`opted_out`) |
| `SLM_BANDIT_DISABLED` | `1` skips bandit re-ranking and its background loops. | off | `core/recall_pipeline.py` |
| `SLM_TRUST_READ_GATE` | `1` enables the trust check on reads. | off | `trust/gate.py` |
| `SLM_CACHE_BACKEND` | `tiered` or `sqlite` for the derived cache. | `tiered` | `cache/factory.py` |
| `SLM_CACHE_DB` | Path of the active-brain cache database. | `active_brain_cache.db` in the data folder | `core/context_cache.py` |
| `SLM_DERIVE_CACHE_MAX_MB` | Size cap of the derived cache. | 2048 | `cache/sqlite_store.py` |
| `SLM_AUTO_PROMOTE_MIN_EDGES` | Graph edges before the scale engine auto-promotes. | 100000 | `core/backend_orchestrator.py` |
| `SLM_VECTOR_HISTORY_DAYS` | How many days of old vector-store versions the background compaction keeps. Every saved memory writes a version, and every semantic search walks the versions, so a long history slows search. | 1/24 (one hour) | `core/maintenance_scheduler.py` |
| `SLM_REINDEX_BATCH`, `SLM_REINDEX_PAUSE_S`, `SLM_REINDEX_DUTY_REST` | Batch size, pause between batches in seconds, and rest as a fraction of batch time during a model switch. | 32, 0.02, 1.0 | `core/embedding_reindex_steps.py` |
| `SLM_EVOLUTION_BACKEND` | `claude-cli` runs skill evolution through the Claude CLI. | Anthropic API | `evolution/llm_dispatch.py` |
| `SLM_EVO_AUTO_APPROVE` | `1` or `true` auto-approves skill evolutions. `EvolutionConfig` has no field for it, so the variable decides. | off | `evolution/skill_evolver.py` |

`SLM_EVOLUTION_ENABLED` also exists, but the daemon sets it for its hooks from `evolution.enabled`, so set the config key instead.

### Pictures and documents

| Variable | Effect | Default | Source |
|---|---|---|---|
| `SLM_MEDIA_ALLOW_LOW_RAM` | `1` treats a computer below the 15 GiB check as large enough for the automatic set-up. | off | `runtimes/media_env.py` |
| `SLM_NO_AUTO_DOWNLOAD` | `1` stops the automatic download of the picture and document models on this computer. | off | `runtimes/media_auto.py` |
| `SLM_MEDIA_WORKER_RSS_LIMIT_MB` | Overrides the picture worker's size cap. | per model (4500 MB for EmbeddingGemma 2) | `runtimes/media_models.py` |
| `SLM_MEDIA_SPACE_MODE` | `paired` or `separate` picture search space. | `separate` | `runtimes/space_plan.py` |
| `SLM_DOC_MAX_MB` | Largest PDF accepted, in MB. | 100 | `documents/submit.py` |
| `SLM_DOC_MAX_PAGES` | Most pages in one PDF. | 500 | `documents/runner.py` |
| `SLM_MEDIA_URL_HOSTS` | Extra hosts a connected web app may fetch attached files from. | none | `core/media_fetch.py` |

### Mesh

| Variable | Effect | Default | Source |
|---|---|---|---|
| `SLM_MESH_HOST` | Address the mesh listener binds. | `127.0.0.1` | `mesh/broker.py` |
| `SLM_MESH_SHARED_SECRET` | Shared bearer token; required when the host is not local. | unset | `mesh/broker.py` |
| `SLM_MESH_WS_PORT` | Port used for mDNS announcement. | 7900 | `mesh/broker.py` |
| `SLM_MESH_DISCOVERY` | `off` disables mDNS discovery, which only applies off-host. | on | `mesh/broker.py` |
| `SLM_MESH_PEER_URL` | Full URL of a remote SLM instance. | unset | `mesh/broker.py` |
| `SLM_MESH_TLS` | `1`, `on`, `true` or `yes` uses TLS for discovered peers. | off | `mesh/remote_sync.py` |

Provider keys are read from `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `AZURE_OPENAI_API_KEY` and `OPENROUTER_API_KEY` when set (`core/config.py` provider table, `llm/backbone.py`).

**Rate limits.** The daemon starts its limiters at 30 writes and 120 reads per 60 seconds (`SLM_RATE_LIMIT_WRITE`, `SLM_RATE_LIMIT_READ`, `SLM_RATE_LIMIT_WINDOW`). The Settings Rate Limits group can show 100 and 300 until you save a value, because the page reads its own defaults; save a value once and both agree.

**Recall depth.** A recall returns up to 20 results by default (`CANONICAL_RECALL_LIMIT`). Callers can ask for a different number per call, and since 4.2.30 the number asked for is always respected: a question worded like an aggregation no longer returns more than the `limit` given.

## The deployment preset

`config.toml` may contain a `[deployment]` section that picks a preset. Personal is the default (no login, no PII redaction, no retention enforcement, audit on). Enterprise turns login, PII redaction and retention on. A deployment section that is present but invalid resolves to the enterprise preset, so a typo can never silently relax controls. See [Security, Company Mode and Teams](Security-Company-Mode-and-Teams).

## Non-English text

There is no language setting. Replace the embedder and the reranker instead, as described in [Modes Explained](Modes-Explained#language). The reranker keys live under `retrieval` in `config.json`:

```json
{
  "retrieval": {
    "use_cross_encoder": true,
    "cross_encoder_backend": "openai",
    "cross_encoder_endpoint": "https://models.example.test/v1/rerank",
    "cross_encoder_model": "BAAI/bge-reranker-v2-m3",
    "cross_encoder_timeout_seconds": 15.0
  }
}
```

The endpoint must be HTTPS off the machine (plain HTTP is allowed for loopback and, while `retrieval.trust_plain_http_lan` is true, numeric private-LAN addresses), may not contain user info, a query string or a fragment, and redirects are not followed. If it is unreachable, recall returns fusion-ranked results without reranking; it never silently falls back to the English model. Remote reranking sends the query and each candidate's text to that endpoint after a best-effort secret and PII filter, so only use a service you trust.

Code: `core/feature_flags.py`, `core/kill_switch.py`, `outcomes/switch.py`, `core/config.py` (`SLMConfig`, `ScopeConfig`, `RetrievalConfig`, `load_deployment_config`), `infra/data_root.py`, `cli/commands.py` (`cmd_config`), `ui/js/od-settings.js`, `core/remote_mode.py`, `infra/rate_limiter.py`, `retrieval/remote_reranker_config.py`, `core/answer_check_state.py`, `core/memory_kind_config.py`, `mcp/server.py`.
