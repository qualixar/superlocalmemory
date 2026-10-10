# CLI Reference

Complete reference for the `slm` command-line interface. `slm <command> --help`
always shows the options your installed version accepts, and `slm help` prints a
grouped overview with the topics `modes`, `config` and `self-heal`.

Most commands talk to a background daemon and start it if it is not running.

---

## Setup & Configuration

### `slm setup`

Run the interactive setup wizard. Package installation itself is
non-interactive and does not install hooks, edit IDE configuration, start a
daemon, or download a model. `slm setup` is the explicit activation boundary.

```bash
slm setup
slm setup --auto     # non-interactive, defaults, for scripts and CI
```

The wizard checks your system, then asks for the operating mode, whether to
build a code knowledge graph, which models to download (embedding model about
500 MB, reranker about 130 MB, an optional compression model about 560 MB, and
on Apple Silicon the optional [Answer check](answer-check.md) model about
1.1 GB), whether the daemon stays on all the time or shuts down when idle,
mesh, ingestion adapters, entity compilation and skill evolution. Skill
evolution stays off unless you enable it. It ends with a verification step and
then asks for consent before it installs the Claude Code plugin and hooks,
wires any other IDE it detects, and turns on auto-start after login (`slm serve
install`). Every one of those integration steps is skipped when the wizard runs
without a terminal.

### `slm init`

One-command setup: runs the wizard if SLM is not configured, installs the
Claude Code hooks, connects detected IDEs and warms the embedding model.

```bash
slm init
slm init --force   # redo setup even if already configured
slm init --auto    # no terminal needed: Mode A plus hooks, for scripts and CI
slm init --gate    # also enable the experimental PreToolUse gate
```

`--gate` blocks tools until `session_init` has run. It is off by default.

### `slm mode [a|b|c]`

Get or set the operating mode.

```bash
slm mode          # Show current mode
slm mode a        # Local Guardian: no language model, nothing leaves the device
slm mode b        # A model on this machine (Ollama by default, any local server works)
slm mode c        # Your own endpoint, or a cloud provider (key required for cloud)
slm mode b --json
```

Switching modes keeps your embedding, retrieval, forgetting and other settings.
Run `slm restart` to apply the new mode.

### `slm provider [set]`

Get or set the LLM provider for Mode B/C.

```bash
slm provider          # Show current provider
slm provider set      # Interactive provider selector
slm provider set openai   # Set a named provider directly
```

The named providers are `openai`, `anthropic`, `azure`, `ollama`, `openrouter`
and `custom`. `custom` points Mode B or Mode C at your own OpenAI-compatible
endpoint (llama.cpp, vLLM, LM Studio, and others) and needs no key:

```bash
slm provider set custom --endpoint http://192.168.1.50:8041/v1 --mode b
slm provider set custom --endpoint https://my-llm.example.com/v1 --key sk-... --mode c
```

`--mode` is `b` or `c` (default `c`); `--key` and `--model` are optional. The
save is followed by a connection test, printed immediately. It is the same
probe the dashboard runs, so the CLI and dashboard never disagree about whether
an endpoint is reachable. HTTPS is required for public hosts and bare
hostnames; a numeric private-LAN address may use plain HTTP only when
`retrieval.trust_plain_http_lan` is true (the default); loopback is always
allowed.

### `slm connect [ide]`

Configure IDE integrations.

```bash
slm connect                 # Auto-detect and configure all IDEs
slm connect cursor          # Configure Cursor specifically
slm connect claude-code     # Prints the Claude Code plugin install commands
slm connect --list          # Client names supported by this release
slm connect cursor --dry-run
```

| Option | Description |
|--------|-------------|
| `--list` | List all supported IDEs |
| `--dry-run` | Show what would be written without changing anything |
| `--here` | Write config for the current project instead of globally |
| `--transport stdio\|http\|http-mcp-remote` | MCP transport written into the IDE config (default `stdio`). `http` needs the daemon; `http-mcp-remote` is a stdio bridge for clients that only speak stdio |
| `--port N` | Daemon port for the http transports (default 8765) |
| `--verify` | After writing, probe the daemon to confirm an `http` transport is reachable |
| `--profile NAME` | Write `SLM_MCP_PROFILE` into the MCP server block (see [MCP Profiles](profiles.md)) |
| `--cross-platform`, `--disable NAME` | Use the cross-platform adapter orchestrator, or disable one adapter by name |
| `--json` | Structured output |

### `slm upgrade-hosts`

After upgrading SLM, preview which existing host integrations would be
refreshed. The default is read-only; `--apply` writes. `--host NAME` (repeatable)
picks hosts, and `--all-detected` targets only hosts that already contain an SLM
integration. See [Host Integration Upgrades](host-upgrades.md).

---

## Memory Operations

### `slm remember "content" [options]`

Store a memory.

```bash
slm remember "API rate limit is 100 req/min on staging"
slm remember "Use camelCase for JS, snake_case for Python" --tags "style,convention"
slm remember "Always run migrations before deploy" --kind rule
slm remember "Staging moved to Postgres 17" --replaces <fact_id>
slm remember "Maria owns the auth service" --scope shared --shared-with team-a
slm remember "Wait for all enrichment" --sync --json
```

| Option | Description |
|--------|-------------|
| `--tags "a,b"` | Comma-separated tags |
| `--kind KIND` | What sort of memory this is: `rule`, `decision`, `status`, `procedure`, `prospective`, `opinion`, `correction`, `episodic` or `semantic`. Rules and decisions saved this way load at the start of later sessions. See [Memory kinds](memory-kinds.md) |
| `--replaces ID` | Id of an earlier memory this one replaces. The old one stops being returned; undo with `slm review-correction` |
| `--scope` | `personal`, `shared` or `global`. Unset uses the configured default, which is `personal` |
| `--shared-with` | Comma-separated profile ids for shared scope |
| `--sync` | Wait until enrichment finishes instead of returning when the memory is queryable |
| `--json` | Emit the operation receipt and materialization state |

Without `--sync`, the daemon returns once the memory is `queryable` by its
words; meaning-based search catches up in a few seconds. If the daemon is busy
the receipt says `accepted`: the memory is saved durably and indexed within
seconds. If the daemon cannot be reached the command fails with
`DAEMON_UNAVAILABLE`, a diagnosis of why and the next command to run; nothing
is queued silently.

### `slm recall "query" [options]`

Search your memories. Returns ranked evidence under [Score Contract
v2](retrieval-score-contract.md). `slm search` is an alias with the same options.
See [Recall](recall.md) for how the filters combine.

```bash
slm recall "rate limit"
slm recall "who owns auth" --limit 5
slm recall "deploy steps" --project acme-billing --kind procedure
slm recall "staging database" --tag infra --tag postgres --tags-match any
slm recall "what shipped" --window 7d
slm recall "staging database" --known-as-of 2026-01-01T00:00:00+00:00
```

| Option | Default | Description |
|--------|---------|-------------|
| `--limit N` | 20 | Maximum results |
| `--project NAME` | | Only memories saved under this project (name or path). If none of the memories found were, results are not narrowed and recall says so |
| `--project-strict` | off | With `--project`: only that project's memories, even if none |
| `--prefer-project NAME` | | Rank that project's memories higher; removes nothing |
| `--saved-by AGENT` | | Only memories saved by this agent |
| `--about NAME` | | Only memories that mention this person, project or tool |
| `--kind KIND` | | Only one [memory kind](memory-kinds.md) |
| `--tag TAG` | | Only memories with this tag; repeat for more |
| `--tags-match all\|any` | `all` | With several `--tag`: every tag, or at least one |
| `--window SPAN` | | Event-time range: `24h`, `7d`, `30d`, `1y` or `2026-07-01..2026-07-31` |
| `--as-of TIME` | | Recall pinned to an ISO 8601 snapshot |
| `--known-as-of TIME` | | Only facts SLM knew by that time |
| `--valid-at TIME` | | Only facts valid at that time |
| `--include-unknown` | off | Include facts saved before 4.0.2, which have no recorded time, in strict time travel |
| `--include-global` / `--no-global` | off | Include or exclude global-scope facts |
| `--include-shared` / `--no-shared` | off | Include or exclude facts shared with this profile |
| `--fast` | | Guarantee the fast path |
| `--json` | | Structured output |

A time filter that cannot be read stops the command before any search and says
what format it expected.

When [Answer check](answer-check.md) is turned on, output gets one extra line,
for example `Answer check: none of these memories answers the question
(confidence 0.18). Say you don't have it, or ask — don't present these as the
answer.` Results themselves are unchanged; the line is added, never a filter.
`slm trace` prints the same line. When a check is set up but did not judge this
recall (it was busy or still warming), the output says so. After a
recall that could not search everywhere, it prints `Incomplete search: ...`
naming the channels that were skipped. An empty result prints `No confident
match.` or `No matching memories found.`

### `slm forget "query" [options]`

Delete memories matching a fuzzy query. `query` is optional with `--dry-run`.
There is no `slm forget --id`; use `slm delete <fact_id>` for an exact id. This
is a different operation from the MCP `forget` tool, which runs the forgetting
decay cycle.

```bash
slm forget "old staging credentials"             # Confirm before deletion
slm forget "old staging credentials" --dry-run   # Preview only
slm forget "old staging credentials" --yes       # Skip confirmation
slm forget --dry-run                             # Preview every memory
```

| Option | Description |
|--------|-------------|
| `--dry-run` | Preview matches without deleting |
| `--yes`, `-y` | Skip the confirmation prompt |
| `--json` | Structured preview or result |

### `slm delete <fact_id>`

Delete a specific memory by its exact fact id. Get ids from `slm list` or
`slm recall --json`.

```bash
slm delete abc123
slm delete abc123 --yes    # Skip confirmation
slm delete abc123 --json
```

### `slm update <fact_id> <content>`

Propose a correction to one memory. It creates an immutable successor but does
not change current recall until a reviewer applies the case; the predecessor
stays available to time-aware recall. See [Reviewed
corrections](reviewed-corrections.md).

```bash
slm update abc123 "API rate limit is now 200 req/min on staging"
slm update abc123 "Updated content" --json
```

### `slm review-correction <case_id> <apply|reject|rollback> <expected_version>`

Review a proposed correction, using its version for compare-and-swap safety.
`apply` makes the successor current; `rollback` restores the predecessor's
prior state. `--event-valid-until` (apply only) takes a reviewer-approved
real-world boundary.

```bash
slm review-correction case123 apply 0 --json
slm review-correction case123 apply 0 --event-valid-until 2026-08-16T00:00:00Z
```

### `slm corrections overtaken` and `slm corrections restore-overtaken`

A pending correction is closed automatically when you delete, replace or edit
the memory it was about. `slm corrections overtaken [--limit N] [--profile P]`
lists those closed cases, newest first, and `slm corrections restore-overtaken
<case_id>` puts one back.

### `slm list [options]`

List recent memories, newest first, with each one's kind and the id that
`slm update` and `slm delete` take.

```bash
slm list                      # Last 20 memories
slm list --limit 50
slm list --kind decision
slm list --tag infra --tag postgres --tags-match any
```

| Option | Description |
|--------|-------------|
| `--limit N`, `-n N` | Number of entries (default 20) |
| `--kind KIND` | Only one memory kind |
| `--tag TAG`, `--tags-match all\|any` | Exact tag filter |
| `--json` | Structured output |

### `slm trace "query"`

Recall with a per-channel score breakdown (`--limit`, default 10).

```bash
slm trace "database port"
```

The output names the channels that contributed to each result. The candidate
producers are dense semantic, BM25 lexical, temporal, Hopfield associative and
spreading activation; entity-graph information can adjust a post-fusion score
but is not a separate producer.

### `slm health`

Diagnostics for the mathematical layers (similarity, consistency and lifecycle
dynamics), the embedding model and the database. `--json` for structured output.

### `slm models`

Lists the Ollama models installed on this computer, the local models recommended
for its memory, and the hosted catalogue, from the same catalogue as the setup
wizard and the dashboard. `--json` for structured output.

---

## Kinds, summaries and saved views

### `slm kinds ...`

Memory kinds label what sort of thing a memory is. See [Memory
kinds](memory-kinds.md).

```bash
slm kinds status                          # kinds per memory, the backend in use, runs
slm kinds settings --standing-rules on    # show or change settings
slm kinds set <fact_id> decision          # confirm one memory's kind
slm kinds review --kind rule --limit 20   # suggestions awaiting confirmation
slm kinds confirm <fact_id> <fact_id>=rule  # a bare id accepts the stored suggestion
slm kinds backfill start                  # classify existing memories (undoable)
slm kinds backfill status
```

`slm kinds settings` takes `--enable` or `--disable`, `--backend
auto|rules|laya|jev|llm|off`, `--jev-consent yes|no` (allows Jev to type
memories, which sends memory text online) and `--standing-rules on|off`.
`backfill` has `start [--mode untyped|refresh] [--yes]`, `pause`, `resume`,
`cancel` and `revert`, each taking a run id, and `status`. `--yes` confirms a run
that sends memory text online. `refresh` redoes suggestions but never a kind you
confirmed.

### `slm summary ...`

Readable summaries that cite the memory ids they were built from.

```bash
slm summary sessions              # recent sessions you can summarise
slm summary session <session_id>
slm summary day                   # today; also a YYYY-MM-DD date or yesterday
slm summary project               # the current directory; or pass a path
```

Each takes `--json` and `--profile`.

### `slm view ...`

Saved views are named recall queries you can re-run.

```bash
slm view create "Work log" "what did I ship" --window 7d
slm view list
slm view run "Work log"        # same as: slm view show "Work log"
slm view rename "Work log" "Shipped"
slm view delete "Shipped"
```

`create` takes `--kind`, `--window`, `--as-of` and `--limit` (1-50). Deleting a
view removes the saved query only.

---

## Pictures, documents and folders

### `slm features`

What is on and what you can turn on: images and documents, folder sources and bot messages.

### `slm media enable|disable|status|gc`

```bash
slm media enable              # shows the download size (about 1.5 GB) and a disk check, then asks; --yes skips the question
slm media status              # what is on and how set-up is going
slm media disable             # off again; memories are kept (--remove-files also deletes the downloaded models)
slm media gc                  # report picture records without a memory and files without a record; removes nothing
slm media gc --apply          # remove them (owner or admin)
```

Images and documents need a computer with at least 16 GB of memory. On a smaller computer `enable` refuses with a plain reason and exits with code 4; your text memories keep working. Set-up runs in the background inside the SLM service, and a restart (`slm restart`, or the dashboard button) starts the picture worker. Turning it on never changes your existing memories or their embeddings.

### `slm sources ...`

```bash
slm sources add ~/Notes --kind obsidian   # or --kind folder; shows what would be read and asks first
slm sources list
slm sources report <id>
slm sources rescan <id>
slm sources remove <id> [--purge]         # --purge also erases the memories it gave
slm sources forget-empty <id>             # forget the files of a folder you emptied on purpose
```

Sources are read-only: SLM mirrors Markdown, text, canvas files, PDFs and PNG, JPEG and WebP images into memory and never writes to the folder. SLM's own data folder can never be added. Pictures and PDFs in a folder need images and documents turned on.

## Migration

### `slm migrate`

Migrate a V2 database to the current schema (one-shot; no `--dry-run`).

```bash
slm migrate                # Run migration
slm migrate --rollback     # Legacy V2 migrator rollback only
```

The migrator spans file copies, SQLite commits and a symlink, so it is not
transactional end to end. Verify a complete offline backup (`slm serve stop`
first, include WAL/SHM files and `lance/` if present) before migrating. See
[Migration from V2](migration-from-v2.md). `slm migrate --rollback` is only for
that legacy migrator; `slm db migrate` below is forward-only.

---

## Profile Management

### `slm profile [command]`

Manage memory profiles (separate memory contexts on one computer; not a security
boundary, see [Profiles](profiles.md)).

```bash
slm profile list                  # List all profiles
slm profile switch work           # Switch the active profile
slm profile create client-acme    # Create a new profile
slm profile list --json
```

---

## System & Maintenance

### `slm status`

Mode, provider, profile, data folder, database path and size, and whether the
daemon is holding saves back.

```bash
slm status
slm status --verbose   # also: migration log, daemon port, disabled marker, last booted version
slm status --json
```

`--json` returns `mode`, `provider`, `profile`, `base_dir`, `db_path`,
`db_size_mb`, `fact_count`, `entity_count`, `edge_count`, `profile_generation`,
`version` and `projection_queue_depth`; with the daemon running it adds
`saves_waiting` and `unreadable_saves`.

### `slm doctor`

Pre-flight check: dependencies, embedding worker, daemon connectivity and
configuration. Run it after any install or upgrade.

```bash
slm doctor
slm doctor --quick   # dependencies and configuration only
slm doctor --deep    # read every database page (slow on a large store)
slm doctor --fix     # re-download missing models and install sqlite-vec first
slm doctor --json
```

### `slm warmup`

Load the embedding model (about 500 MB to download the first time) and confirm
recall is ready. Exits non-zero if it is not. `--timeout SECONDS` sets how long
to wait for a starting daemon (default 120; 0 checks once).

### `slm dashboard`

Open the local web dashboard. `--port N` (default 8765).

```bash
slm dashboard
slm dashboard --port 9000
```

### `slm serve [start|stop|status|install|uninstall]`

Control the background daemon. `start` is the default. `install` and
`uninstall` add or remove the operating-system service that starts it at login.

```bash
slm serve start
slm serve status
slm serve stop
```

### `slm restart`

Restart the daemon: kills orphans, clears stale state, starts fresh and checks
health. Needed after changes that cannot apply at runtime, such as a new mode.
`--dashboard` opens the dashboard afterwards; `--json` for structured output.

### `slm mcp`

Start the MCP server on stdio, for IDE configurations that run it as a
subprocess. For HTTP transport the daemon exposes `/mcp/` itself. See [MCP
Tools Reference](mcp-tools.md).

### `slm mesh status|peers`

Inspect the local agent mesh: broker health and statistics, or the active peer
sessions.

### `slm rotate-token`

Rotate the SLM install token. Run `slm restart` afterwards.

### `slm disable [--reason "..."]` and `slm enable`

`slm disable` writes a `.disabled` marker and stops the daemon; commands that
need the daemon then print an informational message until `slm enable` removes
the marker.

### `slm clear-cache`

Wipe regenerable caches. `memory.db` and `learning.db` are preserved.

### `slm reconfigure`

Re-run the interactive post-install configurator to change the performance
profile or other install-time options.

### `slm reap [--force] [--all]`

Find orphaned SLM processes. The default only lists them; `--force` kills them
and `--all` kills every `slm mcp` process, for use after switching IDE.
`--json` for structured output.

---

## Global Options

| Option | Description |
|--------|-------------|
| `--help` | Show help for a command |
| `--version` | Show the SLM version |

Metadata options do not create a data folder or start anything. Other options
belong to specific commands; do not assume one accepted by a command is global.

## Agent-Native JSON Output

Commands that advertise `--json` print an envelope. Recall results use Score
Contract v2:

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
    "answer_check_status": "off",
    "reranker_status": "not_configured",
    "local_reranker_status": ""
  },
  "next_actions": [
    {"command": "slm list --json", "description": "List recent memories"}
  ]
}
```

The full response also carries `query_id`, `no_confident_match`, `abstained`,
`answerability`, `channel_status`, `incomplete_channels`, `project_scope` and
`tag_scope`; see [Recall](recall.md#reading-the-json).
`answer_check_status` reports what happened to the [answer
check](answer-check.md): `judged`, `off`, `skipped`, `busy`, `warming` or
`unavailable`. `reranker_status` names the step that produced the final order
(`jev_listwise` when the online check's optional reorder chose it).
`local_reranker_status` keeps the local reranker's own status in that case. A
script that calls the daemon's HTTP API directly can add `?answer_check=skip`
or `?answer_check=no_reorder` to `GET /recall` to change what runs for one
request; any other value is refused with HTTP 400.

### Usage with jq

```bash
slm recall "auth" --json | jq '.data.results[0].content'
slm list --json | jq '.data.results[].fact_id'
slm status --json | jq '.data.mode'
```

### In CI/CD (GitHub Actions)

```yaml
- name: Store deployment info
  run: slm remember "Deployed ${{ github.sha }} to production" --json

- name: Check memory health
  run: slm status --json | jq -e '.success'
```

---

## Optimize Commands

The Optimize layer reduces LLM cost with caching and compression. See [Three
surfaces](optimize-overview.md) and [Optimize CLI](optimize-cli.md).

### `slm optimize status|on|off|savings`

```bash
slm optimize status                # Show all settings
slm optimize on                    # Enable cache + compress
slm optimize off                   # Disable (proxy passes through)
slm optimize savings --since 30    # Token/cost report; default 7 days
slm optimize savings --provider anthropic
```

### `slm cache status|clear|invalidate|ttl|semantic`

Exact cache is the stable path. The semantic cache is experimental.

```bash
slm cache status                   # Entry count, DB size, TTLs, hit rate
slm cache clear                    # Delete all entries for the tenant
slm cache invalidate --tag "key"   # Delete entries by tag
slm cache ttl --set 86400          # Exact-cache TTL in seconds
slm cache ttl --semantic 3600      # Semantic-cache TTL
slm cache semantic on|off
```

Every cache command accepts `--tenant` (default `default`) and `--json`.
`slm cache invalidate --tag mcp-kv` removes entries written by the
`slm_cache_set` tool.

### `slm compress status|mode|prose`

```bash
slm compress status                # Mode and per-channel state
slm compress mode safe|aggressive
slm compress prose on|off          # Prose compression (aggressive mode only)
```

`slm compress code`, `ccr` and `align` still parse but no longer do anything;
they print a notice pointing to `slm compress prose on`.

### `slm proxy [options]`

Start the optimization proxy.

```bash
slm proxy                          # Port 8765, Anthropic surface
slm proxy --port 8080 --provider openai
slm proxy --no-compress            # Cache only
slm proxy --semantic               # Enable the semantic cache for this run
```

`--provider` is `anthropic` (default), `openai` or `gemini`.

### `slm wrap <agent> [options]`

Start the proxy, set the environment and launch an agent.

```bash
slm wrap claude
slm wrap aider -- --model gpt-4
slm wrap --list                    # Registered agents
slm wrap --persistent              # Write the settings instead of launching
slm wrap --dry-run
```

### `slm help-optimize [topic]`

Developer reference with per-agent recipes. Topics: `cache`, `compress`,
`optimize`, `proxy`, `agents`, `safety`; `--no-pager` prints to stdout.

---

## Hooks and IDE Integration

### `slm hooks install|remove|status`

Manage additive auto-capture hooks for Claude Code or Codex. `status` is the
default.

```bash
slm hooks install                 # Claude Code (default)
slm hooks install --agent codex
slm hooks remove
slm hooks status --agent codex
slm hooks install --dry-run
```

`--gate` (Claude Code) enables the experimental PreToolUse gate. After
installing Codex hooks, review and trust them in Codex with `/hooks`.

### `slm codex install|remove|status`

Manage the SLM add-ons for Codex: skills, subagents and lifecycle hooks.
`--dry-run` validates without writing.

```bash
slm codex install
slm codex status
slm codex remove
```

`slm codex install` is additive: it does not replace other agents' hooks or
rewrite `~/.codex/config.toml`. MCP wiring is a separate step: `slm connect
codex`.

---

## Sessions and Lifecycle

### `slm session open|close`

```bash
slm session open --project-path /path/to/project   # Warm context for a project
slm session open --query "auth service work" --max-results 10
slm session close
slm session close --session-id abc123
```

### `slm session-context [query]`

Print session context for hooks: relevant memories for the query or the current
project. `--max-age-days N` (default 30; 0 disables) hides older memories unless
their score is at least 0.7. `--full` uses the full engine path, which is slower
and needs Ollama; the default is a fast SQLite path. `--json` for structured
output.

### `slm observe [content]`

Submit content for automatic capture. SLM decides whether it holds a decision,
bug fix or preference worth storing.

```bash
echo "Decided to use WebSocket over SSE" | slm observe
slm observe "API rate limit is 100 req/min on staging"
```

### `slm brain status`

Read-only summary of the local, profile-scoped Living Brain evidence. `--json`
for structured output.

### Lifecycle cycles

`slm decay`, `slm quantize` and `slm consolidate` preview by default.

```bash
slm decay                   # Preview Ebbinghaus zone transitions
slm decay --execute         # Apply them
slm quantize --execute      # Apply embedding precision changes
slm consolidate --cognitive # Include CCQ cognitive consolidation
slm consolidate --dry-run
slm soft-prompts            # List auto-learned soft prompts
```

`decay`, `quantize`, `consolidate` and `soft-prompts` accept `--profile` and
`--json`.

---

## Data and Evidence

### `slm evidence export|verify|import|rebuild`

Versioned, checksummed JSONL bundles of memory evidence.

```bash
slm evidence export /path/to/bundle.jsonl --profile default
slm evidence verify /path/to/bundle.jsonl
slm evidence import /path/to/bundle.jsonl --execute
slm evidence import /path/to/bundle.jsonl --replace --execute
slm evidence rebuild --execute   # Rebuild derived lexical state
```

| Subcommand | Description |
|-----------|-------------|
| `export <dest>` | Write a deterministic checksummed JSONL bundle |
| `verify <bundle>` | Verify checksums and source reconciliation |
| `import <bundle>` | Import relational truth; a dry run unless `--execute`. Replacing existing memories (`--replace`) also requires `--rollback-dir` |
| `rebuild` | Rebuild derived lexical state; a dry run unless `--execute` |

### `slm diagnostics export <dest>` and `slm diagnostics reliability`

`export` writes a content-free JSON report of local operational aggregates (no
memory content, no secrets) for manual inspection or support. `reliability
[--min-observations N]` asks whether wired mechanisms actually work: whether each
Bayesian learner has moved off its prior and whether each schema-guarded path has
ever run against this store.

### `slm backup status|recovery-key|decrypt`

Cloud backups are encrypted. `status` shows the encryption state;
`recovery-key` shows the recovery key, or with `--import` reads one from stdin
for a new computer; `decrypt <file> [-o OUT] [--recovery-key-stdin] [--force]`
turns a downloaded backup into a plain `.db` file. See [Cloud
backup](cloud-backup.md).

### `slm gdpr status|export|erase|verify`

```bash
slm gdpr status
slm gdpr export --profile work --output work-export.json
slm gdpr erase --profile work --dry-run
slm gdpr erase --profile work --yes     # irreversible
slm gdpr verify --receipt-id <id>
```

`erase` removes a whole profile and cannot be undone; it needs both `--profile`
and `--yes`. `verify` checks an erasure receipt and exits 0 when it is intact,
1 when tampered, 2 when not found. See [Compliance](compliance.md).

### `slm benchmark`

Run the evo-memory benchmark against an isolated temporary database. It never
reads or writes your data. `--json` for structured output.

---

## Configuration and Adapters

### `slm config get|set <key> [value]`

Read any value in `config.json` with dot notation, or set one of a short list of
keys.

```bash
slm config get evolution.enabled
slm config set evolution.enabled true
slm config set scope.recall_include_shared true
```

`set` accepts only these keys and refuses anything else with a
`DISALLOWED_KEY` error that lists them:

| Key | Description |
|-----|-------------|
| `evolution.enabled`, `evolution.backend`, `evolution.max_evolutions_per_cycle`, `evolution.mutation_model`, `evolution.verify_model`, `evolution.confirm_model` | Skill evolution |
| `mesh_enabled` | Turn the agent mesh on or off |
| `daemon_idle_timeout` | Idle seconds before the daemon shuts down |
| `entity_compilation_enabled` | Per-entity knowledge summaries |
| `graph_backend`, `vector_backend` | Scale Engine backends (see `slm db scale`) |
| `scope.default_scope`, `scope.recall_include_global`, `scope.recall_include_shared` | Where new memories go and whether recall includes other profiles' memories (see [Shared memory](shared-memory.md)) |

Other settings are changed with their own command (`slm mode`, `slm provider`,
`slm embedder`, `slm optimize`) or in the dashboard. See
[Configuration](configuration.md).

### `slm adapters [subcommand]`

Manage ingestion adapters (Gmail, Calendar, Transcript).

```bash
slm adapters list
slm adapters enable gmail
slm adapters start gmail
slm adapters status gmail
slm adapters stop gmail
slm adapters disable gmail
```

### `slm ingest [--source ecc|jsonl]`

Import external observations into SLM's learning system.

```bash
slm ingest --source ecc            # Claude Code sessions
slm ingest --source jsonl --file /path/to/data.jsonl
slm ingest --source ecc --dry-run
```

`jsonl` expects objects with `content` and an optional `timestamp`.

### `slm evolve [--session <id>] [--profile <id>]`

Post-session skill evolution. The Stop hook normally runs it; invoke it by hand
to process a specific session.

---

## Remote access over TLS

`slm remote` lets AI tools on other computers use this SLM with a TLS
certificate and per-profile keys. See [Distributed
deployment](distributed-deployment.md#remote-access-over-tls). This is separate
from [Web access](remote-access/README.md), which connects AI apps on the
internet from the dashboard.

```bash
slm remote tls init --name my-laptop.local --ip 192.168.1.20
slm remote enable --listen HOST:PORT
slm remote keys add hermes-laptop --read-only --profile work
slm remote keys list
slm remote keys revoke hermes-laptop
slm remote check
slm remote disable
```

A key is shown once when created. `--read-only` allows recall only. Enabling or
disabling takes effect on restart.

---

## Bounded Loops

### `slm loop demo [--iterations N] [--json]`

Run the keyless convergence demo. A stub proposer runs laps against a
deterministic gate that fails twice and passes on lap 3. Every lap is written
to SLM memory under the tag `loop:convergence-demo` and shows in the dashboard.

```bash
slm loop demo
slm loop demo --iterations 5
slm loop demo --json
```

It confirms the engine, the ledger and the gate mechanism end to end, so it is a
good check after an upgrade.

### `slm loop history [--name NAME] [--json]`

List recorded runs for a loop name from the ledger. `--name` defaults to
`convergence-demo`.

### `slm loop show <run_id> [--json]`

Show every lap of one run, in order. `run_id` is printed by `slm loop demo` and
returned by the `slm_loop_run` MCP tool.

The CLI has `demo`, `history` and `show` only; running a loop against a gate is
done with the MCP tools `slm_loop_run`, `slm_loop_history` and `slm_loop_show`.
See [MCP Tools Reference → Bounded-loop tools](mcp-tools.md#bounded-loop-tools).

---

## Embedding models and store health

### `slm embedder switch|status|cancel|rollback|forget-previous`

Change the embedding model without stopping SLM. A switch re-indexes every
memory in the background inside the running service; recall and remember keep
using the current model until the new one is ready, then both change in one
step. The previous vectors are kept until the next switch, so `rollback` can
return to them; `forget-previous` frees them. A hosted model's key is set in the
dashboard or config, never on the command line.

```bash
slm embedder switch nomic-ai/nomic-embed-text-v1.5
slm embedder switch text-embedding-3-small --provider openai --dimension 1536
slm embedder status
slm embedder cancel
slm embedder rollback
```

`switch` also takes `--endpoint` (an OpenAI-compatible URL), `--no-wait`
(return once queued) and `--json`. `--provider` is `sentence-transformers`,
`ollama` or `openai`; the default is the current one.

### `slm db integrity [--pages] [--json]`

Reports the store's health in five separate sections: page integrity (only with
`--pages`, which reads every page and is slow on a large store), relational
integrity (foreign-key findings and leftover rows), source fidelity (facts
withheld from answers or no longer saying what their memory said), projection
readiness (work still owed to the keyword, vector and date indexes) and any
repair running now or last run. It is read-only, prints counts only, and is safe
while SLM runs.

### `slm db repair [--apply] [--undo RUN_ID] --root ROOT`

Without `--apply`, previews what it would fix:

- leftover rows from deleted memories;
- erased words that remain in an index;
- index updates that never finished;
- memories whose searchable fact was removed;
- vector indexes that no longer match the memories (vector parity). A
  meaning-search vector that is not its memory's own embedding is rewritten from
  that embedding, and a LanceDB row whose memory is gone, deleted or withheld is
  removed. LanceDB is checked only on a store that uses it.

`--apply` makes those changes, in short batches (`--batch-size`, default 100;
`--pause-ms`, default 50; `--max-seconds` stops early so you can run it again),
and prints receipts of fact ids and counts, never memory text. `--undo RUN_ID`
puts back what a run changed, except the vector and erased-text steps, which keep
no copy on purpose. A repair never brings back anything that was erased, deleted
or withheld, and it never inserts a vector for such a memory.

```bash
slm db repair --root ~/.superlocalmemory            # preview
slm db repair --root ~/.superlocalmemory --apply
slm db repair --root ~/.superlocalmemory --undo <run_id>
```

`--apply` and `--undo` refuse to run unless `--root` names the data folder the
command resolves to, so a wrong folder is stopped before anything is written.
With SLM running, the repair runs inside it so it stays the only writer.

**The store check after an upgrade.** You do not have to run any of this by hand.
A few minutes after SLM first starts on a new version, it checks the store once
in the background. The check only reads and counts, and saves its result in
`store-check.json` in the data folder. The dashboard's **Health** page shows it
under **Memory store**, in plain words, with **Repair now**. Repair now first
saves a full backup copy of your memory and changes nothing if that copy fails;
it then runs the same repair as `slm db repair --apply` and checks again. It
removes no memory you have.

### `slm db fidelity [--profile P] [--limit N] [--json]`

Lists facts that no longer say what their memory said (a number turned into a
date, a dropped "never"). It is read-only and safe while SLM runs. With SLM
stopped, `--withhold FACT_ID` takes one fact out of answers (undo with
`--release FACT_ID`). Or correct it with `slm update` and `slm review-correction`.

### `slm db regraph [--check] [--profile P]`

Re-derive the graph copy of your memories from the store. `--check` only reports
how far the copy has drifted.

### `slm db reembed [--all-profiles] [--limit N] [--json]`

Backfill embeddings for facts that never got one.

### `slm db compact [--offline]`

Drop old LanceDB vector-store versions. `--offline` needs the daemon stopped and
also removes unverified files; use it for a leaked store.

### `slm db restore-points`, `slm db restore`, `slm db prepare-downgrade`

The dashboard's **Updates & restore** card from a terminal. `slm db
restore-points` lists the copies of your memories you can go back to. `slm db
restore <point_id>` goes back to one, applied when SLM next starts; `--cancel`
cancels one that has not run, `--no-reimport` skips re-adding memories saved
after the point, and `--yes` skips the prompt. `slm db prepare-downgrade` gets
your memories ready to run on the previous SLM version (`--cancel` undoes it).
See [Host upgrades](host-upgrades.md) and [Restore points](restore-points.md).

## Database Maintenance

### `slm db migrate [--status|--dry-run]`

Run or inspect additive database schema migrations. They apply automatically at
startup and are forward-only (no `--rollback`).

```bash
slm db migrate --status    # Show migration status (no writes)
slm db migrate --dry-run   # Preview what would change (no writes)
slm db migrate             # Apply pending migrations
```

### `slm ops list|resolve|status`

Inspect and resolve failed, stuck or degraded operations.

```bash
slm ops list --profile work --json
slm ops status --json
slm ops resolve <operation_id> --action retry
slm ops resolve <operation_id> --action force_reconcile
slm ops resolve <operation_id> --action cancel
```

`resolve` is an administrator action. Take the id from `slm ops list` and pick
the action deliberately; it can re-drive or cancel durable work.

### `slm db scale status|adopt|prepare|verify|promote|rollback`

Manage the optional Scale Engine (CozoDB graph and LanceDB vector projections).

```bash
slm db scale status
slm db scale prepare
slm db scale verify --stage-id <id>
slm db scale promote --stage-id <id>
slm db scale rollback --backup-id <id>
slm db scale adopt                 # Adopt a detected earlier projection
```

SQLite with sqlite-vec stays canonical. Projections are parity-gated; a failed
verify leaves recall on SQLite and keeps the rejected manifest for inspection.
See [Scale engine](scale-engine.md).
