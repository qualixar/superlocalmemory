# CLI Reference

The installed CLI is the command source of truth. Use `slm --help` and
`slm <command> --help`; commands that advertise `--json` provide structured
output.

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

## Memory Operations

### Store

```bash
slm remember "Fixed the auth bug — JWT expiry was set to 1 hour instead of 24"
slm remember "API rate limit is 100/min" --tags "api,config"
slm remember "Important fact" --json    # Agent-native JSON output
slm remember "Shared decision" --scope shared --shared-with team-a
slm remember "Wait for enrichment" --sync --json
```

Store a memory. The default daemon path commits raw evidence plus a queryable SQLite relational/FTS projection, then enriches the same durable operation in the background.

Options:
- `--tags "tag1,tag2"` — Add tags
- `--json` — Output structured JSON (for agents, scripts, CI/CD)
- `--sync` — Wait for all declared derivation and projector stages
- `--scope personal|shared|global` — Set visibility
- `--shared-with "profile-a,profile-b"` — Name readers for shared scope

JSON output includes `operation_id`, `materialization_state`, and fact IDs. If
the daemon cannot start, raw evidence enters the legacy offline spool; replay
submits it through M018 before marking that spool row done.

### Recall

```bash
slm recall "JWT token configuration"
slm recall "auth setup" --limit 5 --json
```

Retrieve memories using the candidate producers healthy in the configured
mode, followed by fusion, optional reranking, and graph-based score
enhancement. Results follow [Score Contract v2](Retrieval-Score-Contract).

Options:
- `--limit N` — Number of results (default: 10)
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
spreading activation. Entity-graph information can enhance a post-fusion score
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

Update the content of a specific memory. Use `slm list` to find fact IDs.

Options:
- `--json` — Output structured JSON

## IDE Integration

```bash
slm connect        # Auto-detect and configure all installed IDEs
slm connect --list # Show which IDEs are configured
slm mcp            # Start MCP server (stdio transport — used by IDEs)
```

The `slm mcp` command is what your IDE calls internally. You typically don't run it directly — your IDE's MCP config handles it:

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
slm profile switch <name>     # Switch active profile
```

Personal facts are profile-isolated by default. Shared and global recall are
opt-in and remain subject to the configured scope policy; do not use profiles
as a substitute for operating-system or tenant isolation.

## Migration

```bash
slm migrate                   # Upgrade V2 database to V3
slm migrate --rollback        # Undo migration
```

## Dashboard

```bash
slm dashboard                 # Open web dashboard at http://localhost:8765
slm dashboard --port 9000     # Use a custom port
```

17-tab dashboard: memory browser, knowledge graph, recall lab, trust scores, math health, compliance, learning, IDE connections, settings, and more.

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
| IDE integration | MCP | Run `slm connect --list`, then configure a listed client |
| Shell scripts | CLI + `--json` | `slm recall "auth" --json \| jq '.data.results'` |
| CI/CD pipelines | CLI + `--json` | `slm remember "deployed v2.1" --json` |
| Agent frameworks | CLI + `--json` | OpenClaw, Codex, Goose, nanobot |
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
| 5 | `slm migrate [--rollback]` | | V2 to V3 migration |
| 6 | `slm remember "..."` | Yes | Store a memory |
| 7 | `slm recall "..." [--limit N]` | Yes | Search memories |
| 8 | `slm list [-n N]` | Yes | List recent memories (shows IDs) |
| 9 | `slm forget "..." [--yes]` | Yes | Delete matching memories |
| 10 | `slm delete <id> [--yes]` | Yes | Delete specific memory by ID |
| 11 | `slm update <id> "..."` | Yes | Update a specific memory |
| 12 | `slm status` | Yes | System status |
| 13 | `slm health` | Yes | Math layer health |
| 14 | `slm trace "..."` | Yes | Recall with channel breakdown |
| 15 | `slm mcp` | | Start MCP server (for IDE) |
| 16 | `slm warmup` | | Pre-download embedding model |
| 17 | `slm dashboard [--port N]` | | Launch web dashboard |
| 18 | `slm profile list\|create\|switch` | Yes | Profile management |

---
*Part of [Qualixar](https://qualixar.com) | Created by [Varun Pratap Bhardwaj](https://varunpratap.com)*
