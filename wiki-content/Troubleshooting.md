# Troubleshooting

This page is organised by what you see, not by component. Start with two commands: `slm doctor` checks Python, dependencies, the embedding worker, disk space, the database, the projection queue, images and documents, and the component registry, and `slm status --json` gives a machine-readable snapshot to paste into a bug report. `slm doctor --fix` first repairs what it can (re-downloads missing models, installs sqlite-vec) and then reports; `--quick` runs only the fast checks and `--deep` reads every database page. The data folder is `~/.superlocalmemory/` unless `SLM_DATA_DIR` says otherwise, and daemon logs are in its `logs/` folder (`daemon.log`, `daemon-error.log`).

## The daemon will not start, or commands say it is unavailable

Almost everything that writes goes through one resident daemon. When it cannot be used, a CLI command fails with the code `DAEMON_UNAVAILABLE` (retryable) and a `reason` that names the cause. With `--json` the same `code`, `message`, `hint` and `retryable` fields come back in the error object; MCP tools return the same code.

| Reason | What it means | Do this |
|---|---|---|
| `no_daemon` | No daemon is registered for this data folder | `slm serve start` |
| `daemon_starting` | It is still booting | Retry in a few seconds |
| `daemon_process_exited` | The recorded process is gone | `slm serve start` |
| `pid_reused_by_another_process` | Another process now holds the old process id | `slm restart` |
| `daemon_identity_mismatch` | The process did not match the identity on file; a stepped clock can cause this (common under WSL2) | `slm restart` |
| `descriptor_unusable` | The daemon descriptor file is unreadable or belongs to another user or data folder | `slm restart` |
| `daemon_unreachable` | The process is alive but did not answer `/health` in time | Read `logs/daemon.log`; `slm restart` if it stays stuck |
| `port_owned_by_another_daemon` | Another SLM instance holds the port | Stop it, or set `SLM_DAEMON_PORT` to a free port |
| `request_rejected` | The daemon is healthy but dropped this request | Read `logs/daemon.log` for the failing request |

If `slm serve start` returns but nothing works, check whether SLM was switched off: `slm disable` writes a `.disabled` file in the data folder, and the environment variable `SLM_DISABLE=1` does the same for one shell. `slm enable` removes the file. If the machine has stray SLM processes from closed editors, `slm reap` kills orphans whose parent died (`--force` or `--all` for more), and `slm restart` is the heavy option: kill orphans, clean state, start, verify health.

A 429 from the daemon (`{"error": "Too many requests."}` with a `Retry-After` header) means a client exceeded its rate limit. For clients other than the local dashboard the default is 30 writes and 120 reads per 60 seconds; the local machine gets a far higher ceiling (at least 300 writes and 2,000 reads). Raise them with `SLM_RATE_LIMIT_WRITE`, `SLM_RATE_LIMIT_READ` and `SLM_RATE_LIMIT_WINDOW`, or in Settings > Rate Limits.

A save can also be refused with nothing stored when the daemon's intake queue is full. That refusal is marked retryable, so waiting a second and trying again is safe.

## Recall returns nothing, or the wrong things

Memories belong to a profile. If you saved under `work` and you are in `default`, nothing appears: `slm profile list`, then `slm profile switch work` ([Profiles](Profiles)). Next look for filters: `--project`, `--kind`, `--tag`, `--window`, `--as-of`, `--known-as-of` and `--valid-at` all narrow the search, and an empty result carries a note saying what the filter did. Bad filter values are refused rather than ignored, with `INVALID_KIND` (not one of the nine [memory kinds](Memory-Kinds)) or `INVALID_TIME_FILTER` (use `7d`, `2026-07-01..2026-07-31`, or an ISO 8601 time); in both cases nothing was searched. An unknown `profile_id` is refused as `unknown_profile`, never treated as empty.

If filters are not the cause, the semantic channel may be down. Run `slm warmup`, which loads the embedding model and reports whether recall is ready, and `slm doctor` for the "Embedding worker" line. Recall also drops results whose semantic similarity is below an evidence floor of 0.60, so a very loose query can return fewer results than you expect rather than weak guesses. To see why a query behaves as it does, use `slm trace "query"` or the dashboard Memories > Recall Lab, which shows each channel as ok, no hits, off, failed, timed out or no embedding ([How Recall Works](How-Recall-Works)). After bulk imports or when entity links look wrong, `slm db regraph --check` shows how far the graph has drifted and `slm db regraph` re-derives it; `slm db reembed` backfills facts that never got an embedding.

If recall answers "SuperLocalMemory is answering other questions right now. Ask again in about N seconds", every recall slot is serving other agents (HTTP 503 with `Retry-After`, `code: RECALL_BUSY` over MCP, exit code 75 from `slm recall`). It says nothing about whether the memory exists and nothing was lost: ask again. A question waits up to about twelve seconds for a slot before it says this. If it keeps happening with only one or two agents, run `slm doctor` and look at the machine's load.

If recall is slow on a large store, two things are worth checking. The vector store keeps one version per saved memory, and only the last hour of them is kept by default (`SLM_VECTOR_HISTORY_DAYS`); a store that grew before that, with thousands of old versions, can be shrunk at once with SLM stopped: `slm db compact --offline`. While the Mode B language model is working on extraction or summaries, recall is slower than on an idle machine ([How Recall Works](How-Recall-Works)).

If `slm doctor` shows "Answer check fit" as a warning, your text search runs on a model the answer check is not fitted for, and recall may say a memory is not there when it is. An older release moved the text there. Press **Roll back** on the Settings card, or run `slm embedder rollback`; your memories are kept ([Embedding Models](Embedding-Models)).

If recall says "That isn't in SuperLocalMemory." or "I don't have this exactly", that is the support check doing its job, not an error ([How Recall Works](How-Recall-Works)). Before you conclude the memory is missing, read the note: "SuperLocalMemory is still warming up — results may be incomplete. Try again in a few minutes." means a model was still loading, so wait and ask again. To see the closest matches anyway, use `slm recall "..." --show-weak` or the **Show closest matches** box in the dashboard's Memories search. About the on-device check itself, see [Answer Check](Answer-Check). Quick fixes there: "Needs a Mac with Apple Silicon" means the on-device option is unavailable, so choose Online with Jev or leave it off. A key test result of "The key was not accepted." means re-check or regenerate the key, and "The account has no credit left." means the provider account needs credit. If several SLM processes run at once, only one loads the on-device model and the others report recalls as unchecked. Uninstalling SLM leaves the on-device model in `~/.superlocalmemory/runtimes/laya`; choose Remove in Settings first or delete the folder.

## The memory store looks damaged

Start read-only: `slm db integrity` reports pages, relations, fidelity, projections and repair status without changing anything, and is safe while SLM runs; add `--pages` to read every page (slow on a large store). The dashboard checks once after each upgrade and shows the result in Health > Memory store, with **Check again** and **Repair now**. Repair takes a full backup copy first and changes nothing if that copy fails. From a terminal, `slm db repair --root <data folder>` previews what it would do; add `--apply` to do it, and `--undo RUN_ID` puts back what a run changed. Codes you may see: `NO_STORE` (no `memory.db` in the data folder, so check `SLM_DATA_DIR`), `WRONG_ROOT` (the `--root` does not match the folder this SLM serves; nothing changed), `BUSY` (a repair is already running), `NOT_AUTHORIZED` (your role on that profile does not allow it).

A bare full-table delete on the vector table fails with `database disk image is malformed`. That message is misleading: the database is not corrupt, and integrity checks keep reporting ok. Do not replace the database because of it, and never edit the `fact_embeddings*` tables by hand. Real corruption is very rare with SQLite WAL mode; if `slm db integrity --pages` says the file is unsound, stop the daemon (`slm serve stop`), restore a complete, verified backup of the whole data folder taken with the daemon stopped, check it with `slm status --json`, then start again. There is no supported schema downgrade except restoring a verified pre-upgrade backup. For failed or stuck background work (dead-letter items, stuck ingestion, writer stalls) use `slm ops list`, `slm ops resolve` and `slm ops status`, or Governance > Admin: Ops Health. Details: [Health and Repair](Health-and-Repair).

## Pictures and documents

**A picture or page says it is warming.** The picture model starts on demand. A save that arrives while it loads returns `warming` with "images are starting up; try again in a minute" (the worker says "The image model is starting. Try again in a moment."). A local caller waits up to 90 seconds before that answer and a connected web app 20 seconds, so a second try a minute later normally works. Nothing is lost: the picture is cleaned up and you resubmit.

**Images and documents did not set up by themselves.** On a computer with 16 GB of memory or more (the code checks 15 GiB) SLM does it a few minutes after it starts, one download at a time, when the computer is idle and has the disk space; `slm media status` says what is happening and, after a failed download, why (it tries again later). Below 16 GB nothing downloads by itself and it is never refused: tap the button and read the warning "This computer has N GB of memory; pictures and documents may be slow...". If you turned them off, they stay off until you turn them on. `slm doctor` has an "Images & documents" line.

**The image model errors.** The fixed messages are "The image model took too long and was stopped (timed out).", "Not enough free memory to load the image model right now.", "The image model could not be started.", "The image model could not be loaded." and "The image model stopped unexpectedly." Close memory-heavy apps and retry, and run `slm media status` and `slm media repair`. A picture saved but not searchable by what it shows says "run: slm media repair". A recall-by-content problem for pictures is a vector problem, so repair is the first step; `slm media gc` fills in memory links left pending.

**A PDF job fails.** The job error is one of `encrypted` (password needed), `unreadable`, `too_many_pages` (over 500 pages by default), `page_timeout` (30 seconds on one page), `time_limit` (1,200 seconds for the job), `memory_limit` (parser over 1,600 MB), `pdf_reader_missing`, `page_failed` or `failed`. Files over 100 MB are refused (`SLM_DOC_MAX_MB`). See [Pictures and Documents](Pictures-and-Documents).

## Connected web apps

Web access needs this computer on and online. These codes are produced by the hosted gateway rather than by code in this repository, so the meanings below come from the product's error reference.

| Code | Meaning | What to do |
|---|---|---|
| `connector_asleep` | No heartbeat from your computer for over 45 seconds | Wake it and retry. A sleep never appears as an empty recall |
| `connector_offline`, `connector_unavailable`, `connector_closed`, `connector_error`, `connector_replaced`, `stale_connector` | The link to the gateway is down or was just replaced | Check the network, `slm status`, and the Connected apps page |
| `relay_timeout`, `origin_timeout` | A call passed its 25-second deadline | Retry, one call at a time |
| `connector_busy` (older guides: `relay_busy`) | Too many calls in flight (limit 8), HTTP 429 | Wait a few seconds |
| `DAILY_LIMIT_REACHED` | The free daily allowance is used; resets at midnight UTC | Local use is unaffected |
| `TOOL_DENIED`, `INSUFFICIENT_SCOPE` | The app lacks that permission, for example saving | Remove the app and add it again with the permission ticked |
| `REVOKED`, `connection_revoked`, `ENTITLEMENT_REQUIRED` | The app was removed or Web access ended | Turn Web access on again |
| `remote_tool_not_allowed` | The tool is host-only and a remote caller asked for it | Use it from this computer |

If the Connected apps page says the service is not available, run `slm status` then `slm restart`. If sign-in does not finish, use **Continue sign-in** or **Restart sign-in**, which keeps the permissions you chose. If an app saves nothing, saving is a separate permission; the app's row shows **Save** only when it was granted. More in [Web Access and Connected Apps](Web-Access-and-Connected-Apps).

## Installation and editor connections

**`slm: command not found`.** The npm global bin folder is not on your PATH. Add `$(npm prefix -g)/bin` to it, or run `npx superlocalmemory status`. SLM needs Python 3.12 to 3.14 and Node 18 or later. For permission errors during an npm install, point npm's prefix at a folder you own, or install inside a Python virtual environment.

**An editor does not show SLM tools.** Run `slm connect <editor>` to rewrite its MCP block, restart the editor completely rather than reloading the window, and run `slm doctor`. `slm connect --list` shows every target and where its config file would be; see [IDE Setup](IDE-Setup). Editors do not conflict with each other; they share one daemon, which is the single writer. For stale host assets after an upgrade, `slm upgrade-hosts` previews what would be refreshed ([Host Upgrades](Host-Upgrades)).

**Mode C errors.** Authentication errors mean the provider key is missing: `slm provider` shows the configuration and `slm provider set <provider>` resets it, or export the provider's key variable. Timeouts behind a corporate proxy need `HTTPS_PROXY`. If cloud calls are too slow, `slm mode b` (a model on this machine) or `slm mode a` (no language model) avoid the network ([Modes Explained](Modes-Explained)).

**Optimize proxy.** If a client gets 404 on `/v1/messages` after you enabled the proxy, the routes mount only when the daemon starts, so run `slm restart`. A `ZlibError` in Claude Code usually means `ANTHROPIC_BASE_URL` is set both in your shell and in `settings.json`; unset the shell one ([Optimize](Optimize)). Ingestion adapters that do nothing are covered in [Ingestion Adapters](Ingestion-Adapters).

## Resource use and crashes

Worker processes are watched. If the combined memory of SLM's workers goes over a budget (by default 40 percent of RAM with a 2,500 MB floor), the heaviest worker is killed, and workers silent for 60 seconds are killed too. Tune with `SLM_RSS_BUDGET_MB`, `SLM_MAX_WORKER_MB` and `SLM_EMBED_WORKER_RSS_LIMIT_MB`. The database being too large is handled by `slm decay` (preview) and `slm decay --execute`, and `slm db compact` for LanceDB versions.

On macOS, NumPy, SciPy and PyTorch use Apple's Accelerate framework, which has reported defects in matrix factorizations. SLM's save, recall, maintenance and embedding paths make no factorization calls, and the random rotation used for embedding compression avoids Accelerate's QR. If SLM quits unexpectedly and the crash report names `libLAPACK` or `Accelerate`, open an issue with that report and `slm status --json`.

## Exit codes and error envelopes

| Exit code | Meaning |
|---|---|
| 0 | Success |
| 1 | The command ran and failed |
| 2 | The command refused to run: unknown kind, rejected `--replaces`, or a repair aimed at the wrong folder |

`slm gdpr verify` uses 0 for an intact receipt, 1 for a tampered one and 2 when the receipt is not found. Newer MCP tools report `ok: false` and older ones `success: false` with an `error` string; handle both. Daemon HTTP errors use ordinary status codes, and a body that fails validation gets a 422 that says where and why without echoing your value.

If none of this helps, open an issue at the project's GitHub repository with the output of `slm status --json` and the relevant lines of `logs/daemon.log`.

Code: `src/superlocalmemory/cli/daemon_diagnosis.py`, `cli/commands.py` (doctor), `cli/integrity_cmd.py`, `core/slm_disabled.py`, `core/health_monitor.py`, `runtimes/media_env.py`, `runtimes/worker_client.py`, `media/ingest.py`, `server/unified_daemon.py` (rate limits), `docs/errors.md`.
