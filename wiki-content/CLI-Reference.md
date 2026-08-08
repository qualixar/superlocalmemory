# CLI Reference — V4.0.0

The installed CLI is the command source of truth. Use `slm --help` and
`slm <command> --help`; commands that advertise `--json` provide structured
output. This page describes **V4.0.0**; the installed `--help` wins if prose drifts.

## Setup & Status

| Command | Description |
|---------|-------------|
| `slm setup` | Run the interactive setup wizard (mode selection, provider config) |
| `slm status` | Show system status (mode, database path, DB size) |
| `slm mode` | Show current operating mode |
| `slm mode a\|b\|c` | Switch operating mode |
| `slm provider` | Show current LLM provider |
| `slm provider set` | Configure LLM provider (Mode B/C) |
| `slm health` | Show math layer health (Fisher-Rao, Sheaf, Langevin stats) |
| `slm warmup` | Pre-download embedding model (~500MB, one-time) |
| `slm doctor [--fix] [--quick] [--json]` | Pre-flight checks (deps, embedding worker, daemon) |
| `slm restart` | Kill orphans, clean state, start fresh, verify health |

## Memory Operations

### Store

```bash
slm remember "Fixed the auth bug — JWT expiry was set to 1 hour instead of 24"
slm remember "API rate limit is 100/min" --tags "api,config"
slm remember "Important fact" --json    # Agent-native JSON output
slm remember "Shared decision" --scope shared --shared-with team-a
slm remember "Wait for enrichment" --sync --json
```

Store a memory. The default daemon path commits raw evidence plus a queryable SQLite relational/FTS projection, then enriches the same durable operation in the background. V4 seals a hash-verifiable completion manifest (COMPLETE or explicitly DEGRADED).

Options:
- `--tags "tag1,tag2"` — Add tags
- `--json` — Output structured JSON (for agents, scripts, CI/CD)
- `--sync` — Wait for all declared derivation and projector stages
- `--scope personal|shared|global` — Set visibility
- `--shared-with "profile-a,profile-b"` — Name readers for shared scope

JSON output includes `operation_id`, `materialization_state`, and fact IDs. If
the daemon cannot start, raw evidence enters the legacy offline spool; replay
submits it through M018 before marking that spool row done. Stuck ops surface via `slm ops list`.

### Recall

```bash
slm recall "JWT token configuration"
slm recall "auth setup" --limit 5 --json
```

Retrieve memories using the **five candidate producers** healthy in the configured
mode (semantic, BM25 lexical, temporal, Hopfield associative, spreading activation), followed by RRF fusion, optional reranking, and graph-based score
enhancement (entity graph is an enhancement, not a 6th producer). Results follow [Score Contract v2](Retrieval-Score-Contract).

Options:
- `--limit N` — Number of results (default: 20)
- `--json` — Output structured JSON

### List

```bash
slm list                    # Last 20 memories (shows IDs for delete/update)
slm list -n 50 --json       # JSON output with fact IDs
```

List recent memories chronologically. Shows fact IDs needed for `delete` and `update` operations.

Options:
- `--limit N` / `-n N` — Number of entries (default: 20)
- `--json` — Output structured JSON

### Trace

```bash
slm trace "JWT token configuration"
slm trace "database port" --json
```

Same as recall, but shows per-channel score breakdown. Current candidate
producers are dense semantic, BM25 lexical, temporal, Hopfield associative, and
spreading activation (5 producers). Entity-graph information can enhance a post-fusion score
but is not a separate candidate producer.

Options:
- `--json` — Output structured JSON with channel_scores per result

### Forget

```bash
slm forget "JWT token configuration"
slm forget "old staging config" --yes         # Skip confirmation
slm forget "old stuff" --json                 # Preview matches (no delete)
slm forget "old stuff" --json --yes           # Delete and return JSON
```

Delete memories matching a query. Shows matching memories and asks for confirmation before deleting.

Options:
- `--yes` / `-y` — Skip confirmation prompt
- `--json` — Output structured JSON (without `--yes`: preview only; with `--yes`: delete and confirm)

### Delete

```bash
slm delete <fact_id>                # Delete by exact ID (with confirmation)
slm delete <fact_id> --yes          # Skip confirmation
slm delete <fact_id> --json --yes   # Delete and return JSON
```

Delete a specific memory by exact fact ID. Use `slm list` to find fact IDs.

Options:
- `--yes` / `-y` — Skip confirmation prompt
- `--json` — Output structured JSON

### Update

```bash
slm update <fact_id> "corrected content"
slm update <fact_id> "new text" --json
```

Update the content of a specific memory. Use `slm list` to find fact IDs. V4 re-indexes the corrected fact everywhere (semantic + keyword).

Options:
- `--json` — Output structured JSON

## IDE Integration

```bash
slm connect        # Auto-detect and configure all installed IDEs
slm connect --list # Show which IDEs are configured
slm mcp            # Start MCP server (stdio transport — used by IDEs)
# HTTP transport also available: http://127.0.0.1:8765/mcp/
```

The `slm mcp` command is what your IDE calls internally for stdio. You typically don't run it directly — your IDE's MCP config handles it:

```json
{
  "mcpServers": {
    "superlocalmemory": {
      "command": "slm",
      "args": ["mcp"]
    }
  }
}
```

## Profiles

```bash
slm profile list              # List all profiles
slm profile create <name>     # Create a new profile
slm profile switch <name>     # Switch active profile (RBAC member-gated)
```

Personal facts are profile-isolated by default. Shared and global recall are
opt-in and remain subject to the configured scope policy; do not use profiles
as a substitute for operating-system or tenant isolation. See [[RBAC and Teams]] and [[GDPR Compliance]].

## Migration (two distinct commands)

### `slm migrate` — V2 → V3 data migration

```bash
slm migrate                   # Upgrade V2 database to V3 (V2Migrator)
slm migrate --rollback        # Undo migration while the created backup still exists
```

For existing V2 (2.8.6 or earlier) installations. This is not a single atomic transaction: it spans file copies (`~/.claude-memory/memory.db` → `~/.superlocalmemory/memory-v2-backup.db` and `~/.superlocalmemory/memory.db`), SQLite commits that add V3 tables/columns and re-index for 5-producer retrieval, and a rename/symlink (`~/.claude-memory` → `~/.superlocalmemory`) — failures are reported, not silently rolled back. A backup is created (`~/.superlocalmemory/memory-v2-backup.db` and `~/.claude-memory-v2-original` when applicable); operators must verify the result (`slm status`, `slm health`, `slm status --json | jq '.data.fact_count'`, and recall checks) before relying on the new store. Rollback via `slm migrate --rollback` is only possible while the created backup still exists — verify (`ls ~/.superlocalmemory/memory-v2-backup.db` and check for `~/.claude-memory-v2-original`) before use; code has no automatic 30-day deletion or guaranteed window. Nothing is done if no V2 installation is detected.

### `slm db migrate` — V4 additive schema maintenance

```bash
slm db migrate --status       # Inspect forward/deferred migrations
slm db migrate                # Apply pending additive migrations (forward only)
slm db migrate --dry-run      # Preview (no writes)
```

Wraps LLD-07 additive migrations — **forward apply only** with `status` and `dry-run` inspection. There is no `slm db migrate --rollback`; `src/superlocalmemory/cli/db_migrate.py` supports only `status`, `dry-run`, and forward `apply`. Refuses to run against a DB written by a newer build; a migration whose dependency did not complete is held back. V4.0.0 includes M038 (eager, applied at startup) and M039 (deferred, applied once engine-owned tables exist) — no manual `slm db migrate` is normally required. Schema downgrade is unsupported; to revert a V4 upgrade, restore a verified pre-upgrade backup of the complete data root (see backup guidance below — stop the daemon first and copy the data-root store set with WAL/SHM). This is **not** the V2→V3 migrator above.

### `slm db scale` — parity-gated projections

```bash
slm db scale status                              # Show Scale Engine state
slm db scale prepare                             # Stage a new projection
slm db scale verify --stage-id <id>             # Verify parity with canonical SQLite
slm db scale promote --stage-id <id>            # Promote verified projection
slm db scale rollback --backup-id <id>          # Roll back to a prior projection
slm db scale adopt                               # Adopt a detected pre-v3.7 projection
```

## Operations & remediation (V4)

```bash
slm ops list --profile <name> --json   # List failed/stuck/degraded ops (admin)
slm ops resolve <operation_id> --action retry|force_reconcile|cancel
slm ops status --json                  # Quick failure count + writer stall overview
```

MCP equivalents (`power`/`whole` profile): `list_failed_operations`, `resolve_operation` (see [MCP Tools](MCP-Tools)). Also visible in the dashboard Operations / Health panel.

## Bounded loops (V4)

```bash
slm loop demo --iterations 10 --json   # Keyless convergence demo (deterministic stub)
slm loop history --name <loop> --json  # List recorded runs for a loop
slm loop show <run_id> --json          # Show every lap of one run
```

MCP: `slm_loop_run` / `slm_loop_history` / `slm_loop_show` (`code`/`full`/`power`/`whole`). See [Bounded Loops](Bounded-Loops).

## Dashboard

```bash
slm dashboard                 # Open web dashboard at http://localhost:8765
slm dashboard --port 9000     # Use a custom port
```

Local dashboard workspaces include Dashboard, Brain, Knowledge Graph, Memories, Health, Governance (Access & Users / Data Privacy / Audit / Lifecycle & Trust), Operations, Entity Explorer, Skill Evolution, Mesh Peers, MCP & Tools, Cloud Backup, Settings, and Optimize. Workspace/tab counts are illustrative — verify the installed build; do not treat a tab count as a contract.

## Other notable commands

```bash
slm evidence export <dest> --profile default --json   # Checksummed JSONL bundle (GDPR Art. 15/20)
slm evidence verify <bundle> --json
slm evidence import <bundle> --profile default --replace --json
slm diagnostics export <dest> --json                  # Bounded operational aggregates
slm cache status|clear|invalidate|ttl|semantic --json
slm compress status|mode|code|prose|ccr --json
slm optimize status|on|off|savings --since 7 --json
slm proxy --port 8765 --provider anthropic            # Optimization proxy
```

Run `slm --help` and `slm <command> --help` for the full surface — this page is an orientation map, not the complete parser.

## Examples

```bash
# Store a decision with tags
slm remember "Chose PostgreSQL over MongoDB for the user service. Reason: ACID transactions needed for billing." --tags "architecture,database"

# Recall with channel breakdown
slm trace "database decision for user service"

# Check system status
slm status

# Check math layer health
slm health

# Switch to full power mode
slm mode c

# Open the dashboard
slm dashboard
```

## Agent-Native JSON Output

Commands that advertise `--json` provide structured output. Recall fields keep
ranking relevance separate from stored-memory confidence:

```json
{
  "success": true,
  "command": "recall",
  "version": "4.0.0",
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
    "answer_confidence": null
  },
  "next_actions": [
    {"command": "slm list --json", "description": "List recent memories"}
  ]
}
```

Structured-output support is explicit per command and can expand between
releases.

**Usage with jq:**

```bash
slm recall "auth" --json | jq '.data.results[0].content'
slm list --json | jq '.data.results[].fact_id'
slm status --json | jq '.data.mode'
```

**In CI/CD (GitHub Actions):**

```yaml
- name: Store deployment info
  run: slm remember "Deployed ${{ github.sha }}" --json

- name: Verify memory health
  run: slm status --json | jq -e '.success'
```

## Dual Interface: MCP + CLI

SuperLocalMemory exposes both MCP and CLI surfaces:

| Need | Use | Example |
|------|-----|---------|
| IDE integration | MCP (87 whole / 42 full default) | Run `slm connect --list`, then configure a listed client |
| Shell scripts | CLI + `--json` | `slm recall "auth" --json \| jq '.data.results'` |
| CI/CD pipelines | CLI + `--json` | `slm remember "deployed v2.1" --json` |
| Agent frameworks | CLI + `--json` + adapters | [[Framework Adapters]] — 9 adapters |
| Human use | CLI | `slm recall "auth"` (readable output) |

## Common Command List

This is an orientation list, not the complete installed surface. Run `slm
--help` for the installed release.

| # | Command | --json | What It Does |
|:-:|---------|:------:|-------------|
| 1 | `slm setup` | | Interactive first-time wizard |
| 2 | `slm mode [a\|b\|c]` | Yes | Get or set operating mode |
| 3 | `slm provider [set]` | | Get or set LLM provider |
| 4 | `slm connect [--list]` | Yes | Configure IDE integrations |
| 5 | `slm migrate [--rollback]` | | **V2→V3 data migration** — rollback only while backup still exists; verify before use (`slm db migrate` is different — see above) |
| 5b | `slm db migrate [--status \| --dry-run]` | | Additive schema maintenance (V4, forward only; no rollback) |
| 6 | `slm remember "..."` | Yes | Store a memory |
| 7 | `slm recall "..." [--limit N]` | Yes | Search memories (5 producers) |
| 8 | `slm list [-n N]` | Yes | List recent memories (shows IDs) |
| 9 | `slm forget "..." [--yes]` | Yes | Delete matching memories |
| 10 | `slm delete <id> [--yes]` | Yes | Delete specific memory by ID |
| 11 | `slm update <id> "..."` | Yes | Update a specific memory |
| 12 | `slm status` | Yes | System status |
| 13 | `slm health` | Yes | Math layer health |
| 14 | `slm trace "..."` | Yes | Recall with channel breakdown |
| 15 | `slm mcp` | | Start MCP server (stdio, used by IDE) |
| 16 | `slm warmup` | | Pre-download embedding model |
| 17 | `slm dashboard [--port N]` | | Launch web dashboard |
| 18 | `slm profile list\|create\|switch` | Yes | Profile management |
| 19 | `slm ops list\|resolve\|status` | Yes | **V4** operational remediation |
| 20 | `slm loop demo\|history\|show` | Yes | **V4** bounded loops |

---
*Part of [Qualixar](https://qualixar.com) | Created by [Varun Pratap Bhardwaj](https://varunpratap.com)*
