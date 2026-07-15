# CLI Reference
> SuperLocalMemory V3 Documentation
> https://superlocalmemory.com | Part of Qualixar

Complete reference for the `slm` command-line interface.

---

## Setup & Configuration

### `slm setup`

Run the interactive setup wizard. Package installation itself is
non-interactive and does not install hooks, edit IDE configuration, start a
daemon, or download a model. `slm setup` is the explicit activation boundary.

```bash
slm setup
```

### `slm mode [a|b|c]`

Get or set the operating mode.

```bash
slm mode          # Show current mode
slm mode a        # Zero-cloud (no LLM, no API key)
slm mode b        # Local LLM via Ollama
slm mode c        # Cloud LLM (requires API key)
```

### `slm provider [set]`

Get or set the LLM provider for Mode B/C.

```bash
slm provider          # Show current provider
slm provider set      # Interactive provider selector
slm provider set openai   # Set provider directly
```

### `slm connect [ide]`

Configure IDE integrations.

```bash
slm connect           # Auto-detect and configure all IDEs
slm connect cursor    # Configure Cursor specifically
slm connect claude    # Configure Claude Code specifically
```

Run `slm connect --list` for the client names supported by the installed
release. A documented configuration is not a claim that every client has
passed the frozen V3.7 cross-client matrix.

## Memory Operations

### `slm remember "content" [options]`

Store a memory.

```bash
slm remember "API rate limit is 100 req/min on staging"
slm remember "Use camelCase for JS, snake_case for Python" --tags "style,convention"
slm remember "Maria owns the auth service" --scope shared --shared-with team-a
slm remember "Wait for all enrichment" --sync --json
```

| Option | Description |
|--------|-------------|
| `--tags "a,b"` | Comma-separated tags for categorization |
| `--sync` | Wait until declared derivation and projector stages complete |
| `--scope` | `personal`, `shared`, or `global` visibility |
| `--shared-with` | Comma-separated profile IDs for shared scope |
| `--json` | Emit the operation receipt and materialization state |

Without `--sync`, the daemon returns after the memory is `queryable`; canonical
enrichment continues in the background. If the daemon cannot start, the CLI
stores raw evidence in the legacy offline spool. That spool is a compatibility
input: replay submits the same source/idempotency identity through M018 before
marking it done.

### `slm recall "query" [options]`

Search your memories. Returns ranked evidence under [Score Contract
v2](retrieval-score-contract.md).

```bash
slm recall "rate limit"
slm recall "who owns auth" --limit 5
slm recall "database config" --profile work
```

| Option | Default | Description |
|--------|---------|-------------|
| `--limit N` | 10 | Maximum results to return |
| `--include-global` | off | Include global-scope facts when authorized |
| `--include-shared` | off | Include facts shared with the active profile |

### `slm search "query" [options]`

Alias for `slm recall`. Same behavior, same options.

### `slm forget "query" [options]`

Delete memories matching a query.

```bash
slm forget "old staging credentials"             # Confirm before deletion
slm forget "old staging credentials" --dry-run   # Preview only
slm forget "old staging credentials" --yes       # Skip confirmation
```

| Option | Description |
|--------|-------------|
| `--dry-run` | Preview matches without deleting |
| `--yes`, `-y` | Skip the confirmation prompt |
| `--json` | Emit structured preview or mutation output |

Use `slm delete <fact_id>` for precise deletion by ID.

### `slm list [options]`

List recent memories.

```bash
slm list              # Last 20 memories
slm list --limit 50   # Last 50 memories
```

## V3 Features

### `slm trace "query"`

Recall with a channel-by-channel breakdown. Shows how each retrieval channel contributed to the results.

```bash
slm trace "database port"
```

Output exposes the channels that contributed to each result. The current
candidate producers are dense semantic, BM25 lexical, temporal, Hopfield
associative, and spreading activation. Entity-graph information can enhance a
post-fusion score but is not a separate candidate producer.

### `slm health`

Show diagnostics for the mathematical layers.

```bash
slm health
```

Reports status of:
- Fisher-Rao similarity layer
- Sheaf consistency layer
- Langevin lifecycle dynamics
- Embedding model status
- Database integrity

## Migration

### `slm migrate [options]`

Migrate a V2 database to V3 format.

```bash
slm migrate                # Run migration
slm migrate --rollback     # Roll back when a valid migration backup exists
```

| Option | Description |
|--------|-------------|
| `--rollback` | Revert to V2 format (backup must exist) |

Use `slm db migrate --dry-run` to inspect additive database migrations.

## Profile Management

### `slm profile [command]`

Manage memory profiles (isolated memory contexts).

```bash
slm profile list                  # List all profiles
slm profile switch work           # Switch to "work" profile
slm profile create client-acme    # Create a new profile
slm profile list --json           # Structured profile inventory
```

## System & Maintenance

### `slm status`

Show system status: mode, profile, memory count, database location, health.

```bash
slm status
```

## Global Options

Global metadata options are activation-free. Command-specific options appear in
`slm <command> --help`; do not assume that an option accepted by one command is
global.

| Option | Description |
|--------|-------------|
| `--help` | Show help for a command |
| `--version` | Show SLM version |

## Agent-Native JSON Output

Commands that advertise `--json` emit structured output. Recall results use
Score Contract v2:

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

Check `slm <command> --help` before scripting a surface; structured-output
support is explicit per command and may expand between releases.

### Usage with jq

```bash
# Get first result content
slm recall "auth" --json | jq '.data.results[0].content'

# Get all fact IDs
slm list --json | jq '.data.results[].fact_id'

# Check current mode
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

## Examples

### Daily workflow

```bash
# Morning: check what you remembered yesterday
slm list --limit 10

# During work: store a decision
slm remember "Decided to use WebSocket instead of SSE for real-time updates" --tags "architecture"

# Later: recall the decision
slm recall "real-time communication approach"

# End of day: check system health
slm status
```

### Project setup

```bash
# Create a profile for a new project
slm profile create mobile-app
slm profile switch mobile-app

# Store project context
slm remember "React Native 0.76 with Expo SDK 52"
slm remember "Backend is FastAPI on AWS ECS"
slm remember "CI/CD via GitHub Actions, deploys on merge to main"
```

---

## Optimize Commands (v3.6)

SLM v3.6 adds the **Optimize** layer — Cache + Compress + Align for LLM cost reduction.

### `slm optimize status|on|off|savings`

Master control for the Optimize module.

```bash
slm optimize status                # Show all settings
slm optimize on                    # Enable cache + compress
slm optimize off                   # Disable (proxy passes through)
slm optimize savings               # Token/cost report (last 7 days)
slm optimize savings --since 30    # Last 30 days
slm optimize savings --provider anthropic  # Per-provider filter
slm optimize savings --json
```

### `slm cache status|clear|invalidate|ttl|semantic`

Cache sub-control. Exact cache is the stable path. Semantic cache remains
experimental and must not be treated as enabled production behavior without a
release-linked precision, invalidation, and tenant-isolation report.

```bash
slm cache status                   # Entry count, DB size, TTLs, hit rate
slm cache clear                    # Delete all entries (default tenant)
slm cache invalidate --tag "key"   # Delete entries by tag
slm cache ttl --set 86400          # Set exact TTL (24h default)
slm cache ttl --semantic 3600      # Set semantic TTL
slm cache semantic on|off          # Enable/disable semantic cache
```

### `slm compress status|mode|code|prose|ccr|align`

Compression sub-control — per-channel toggles.

```bash
slm compress status                # Mode + per-channel state
slm compress mode safe|aggressive  # Set aggressiveness
slm compress code on|off           # Code/JSON compression
slm compress prose on|off          # Prose compression (opt-in)
slm compress ccr on|off            # Reversible context retrieval
slm compress align on|off          # Prefix stabilization
```

### `slm proxy [options]`

Start the optimization proxy (port 8765 by default).

```bash
slm proxy                          # Default port 8765
slm proxy --port 8080              # Custom port
slm proxy --provider anthropic     # Provider surface
slm proxy --no-compress            # Cache only
slm proxy --semantic               # Enable semantic cache
```

### `slm wrap <agent> [options]`

Proxy-activate an agent — starts proxy + sets environment + launches agent.

```bash
slm wrap claude                    # Claude Code (recommended)
slm wrap cursor                    # Cursor
slm wrap aider -- --model gpt-4    # Aider
slm wrap --list                    # List registered agents
slm wrap --persistent              # Permanent config write
slm wrap --dry-run                 # Preview
```

### `slm help-optimize [topic]`

Full developer reference with per-agent recipes.

```bash
slm help-optimize                  # Full reference
slm help-optimize cache            # Cache reference
slm help-optimize compress         # Compress + safety warning
slm help-optimize agents           # Per-agent setup
slm help-optimize safety           # Safety warning only
```

Full details: [docs/optimize-cli.md](./optimize-cli.md)

---

*SuperLocalMemory V3 — Copyright 2026 Varun Pratap Bhardwaj. AGPL-3.0-or-later. Part of Qualixar.*
