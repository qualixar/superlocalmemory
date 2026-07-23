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
```

| Option | Default | Description |
|--------|---------|-------------|
| `--limit N` | 20 | Maximum results to return |
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

### `slm delete <fact_id>`

Delete a specific memory by its exact fact ID. Use `slm list --json` or `slm recall --json` to obtain fact IDs.

```bash
slm delete abc123
slm delete abc123 --yes    # Skip confirmation
slm delete abc123 --json
```

### `slm update <fact_id> <content>`

Edit the content of a specific memory by its exact fact ID. The fact must belong to the active profile.

```bash
slm update abc123 "API rate limit is now 200 req/min on staging"
slm update abc123 "Updated content" --json
```

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

## Optimize Commands

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

### `slm compress status|mode|prose`

Compression sub-control — active subcommands.

```bash
slm compress status                # Mode + per-channel state
slm compress mode safe|aggressive  # Set aggressiveness
slm compress prose on|off          # Prose compression (opt-in, aggressive mode only)
```

> **Removed in v3.6.10:** `slm compress code`, `slm compress ccr`, and `slm compress align` no longer perform meaningful work and print a migration notice when invoked. Use `slm compress prose on` for prose-level compression.

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

## Daemon and System

### `slm init`

One-command first-time setup: mode selection, hook installation, IDE configuration, and optional warmup. Equivalent to running `slm setup` then `slm hooks install` then `slm connect` then `slm warmup` interactively.

```bash
slm init           # Interactive wizard covering all setup steps
slm init --mode a  # Non-interactive: Mode A, no hooks, no IDE config
```

| Option | Description |
|--------|-------------|
| `--mode a/b/c` | Set operating mode non-interactively |
| `--no-hooks` | Skip hook installation |
| `--no-ide` | Skip IDE configuration |

### `slm doctor`

Pre-flight check: verifies dependencies, embedding worker, daemon connectivity, and configuration. Run after any install or upgrade.

```bash
slm doctor
slm doctor --json   # Structured output
```

### `slm warmup`

Pre-download the embedding model (~500MB). Prevents the first-use download lag.

```bash
slm warmup
```

### `slm dashboard`

Open the local web dashboard. Requires the daemon to be running.

```bash
slm dashboard           # Opens at http://localhost:8765
slm dashboard --port 9000
```

### `slm serve [start|stop]`

Start or stop the background daemon. The daemon enables instant CLI response and is required by the dashboard, MCP server, and most commands.

```bash
slm serve start    # Start daemon (auto-started on most commands)
slm serve stop     # Stop daemon
```

### `slm restart`

Restart the daemon — required after configuration changes that cannot take effect at runtime.

```bash
slm restart
slm restart --json
```

### `slm mcp`

Start the MCP server in stdio transport mode. Used by IDE configurations that require a subprocess-based MCP server.

```bash
slm mcp
```

For HTTP transport (recommended), the daemon exposes `/mcp/` automatically. See [docs/mcp-tools.md](./mcp-tools.md).

### `slm rotate-token`

Rotate the SLM install token. Run `slm restart` afterward to pick up the new token.

```bash
slm rotate-token
```

### `slm disable [--reason "..."]`

Globally disable SLM. Writes a `.disabled` marker and stops the daemon. All subsequent commands that require the daemon return an informational message until `slm enable` is run.

```bash
slm disable
slm disable --reason "maintenance window"
```

### `slm enable`

Remove the `.disabled` marker. Prints the command to start the daemon.

```bash
slm enable
```

### `slm clear-cache`

Wipe regenerable caches. `memory.db` and `learning.db` are preserved — only derived caches and optimization state are cleared.

```bash
slm clear-cache
```

### `slm reconfigure`

Re-run the interactive post-install configurator. Use to change the performance profile, operating mode, or deployment tier without a full reinstall.

```bash
slm reconfigure
```

### `slm reap [--force] [--all]`

Find and optionally kill orphaned SLM processes.

```bash
slm reap            # Dry-run: list orphans
slm reap --force    # Kill orphaned processes
slm reap --all      # Kill ALL slm mcp processes (use after IDE switch)
slm reap --json
```

---

## Hooks and IDE Integration

### `slm hooks install|uninstall|status`

Manage auto-capture hooks for Claude Code and Codex.

```bash
slm hooks install              # Install for Claude Code (default)
slm hooks install --agent codex  # Install for Codex
slm hooks uninstall
slm hooks status
```

| Option | Description |
|--------|-------------|
| `--agent codex` | Target Codex instead of Claude Code |
| `--global` | Install in global config (all projects) |
| `--project` | Install in project-local config |
| `--dry-run` | Preview changes without writing |

### `slm codex install|remove|status`

Manage SLM add-ons for Codex: skills, subagents, and lifecycle hooks.

```bash
slm codex install       # Add SLM-owned skills, agents, hooks to Codex
slm codex remove        # Remove SLM-owned files from Codex
slm codex status        # Verify installation
slm codex install --dry-run
```

`slm codex install` is additive — it does not replace other agents' hooks or rewrite `~/.codex/config.toml`. MCP wiring is a separate step: `slm connect codex`.

### `slm connect [ide]`

Auto-configure an IDE integration. Detects the installed IDE and writes the appropriate MCP config snippet.

```bash
slm connect                # Auto-detect and configure all IDEs
slm connect claude-code    # Claude Code specifically
slm connect cursor         # Cursor
slm connect codex          # Codex MCP wiring
slm connect --list         # Show all supported IDEs
```

---

## Sessions and Lifecycle

### `slm session open|close`

Manage named sessions for context grouping and temporal summaries.

```bash
slm session open --project-path /path/to/project  # Warm context for a project
slm session open --query "auth service work"       # Explicit query
slm session close                                  # Close and summarize
slm session close --session-id abc123
```

### `slm session-context [query]`

Print session context for use by hooks. Returns relevant memories for the current project path or an explicit query.

```bash
slm session-context                         # Auto-derive from cwd
slm session-context "auth service"          # Explicit query
slm session-context --max-results 5 --json
```

### `slm observe [content]`

Submit content for automatic capture evaluation. The system decides whether the content contains a decision, bug fix, or preference worth storing.

```bash
echo "Decided to use WebSocket over SSE" | slm observe
slm observe "API rate limit is 100 req/min on staging"
```

---

## V3.3 Lifecycle Commands

### `slm decay [--execute]`

Run the Ebbinghaus forgetting decay cycle. Default is dry-run.

```bash
slm decay                   # Preview transitions
slm decay --execute         # Apply zone transitions
slm decay --profile work --execute
slm decay --json
```

### `slm quantize [--execute]`

Run the EAP embedding quantization cycle. Default is dry-run.

```bash
slm quantize                # Preview changes
slm quantize --execute      # Apply precision changes
slm quantize --json
```

### `slm consolidate [--cognitive]`

Run the memory consolidation pipeline.

```bash
slm consolidate                      # Standard consolidation
slm consolidate --cognitive          # Include CCQ cognitive consolidation
slm consolidate --dry-run            # Preview
slm consolidate --profile work
slm consolidate --json
```

### `slm soft-prompts`

List active soft prompts — patterns the system has automatically learned from usage.

```bash
slm soft-prompts
slm soft-prompts --profile work --json
```

---

## Data and Evidence

### `slm evidence export|verify|import|rebuild`

Export, verify, import, or rebuild versioned memory evidence bundles. Evidence bundles are deterministic, checksummed JSONL archives.

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
| `import <bundle>` | Import relational truth (dry-run unless `--execute`) |
| `rebuild` | Rebuild derived lexical state (dry-run unless `--execute`) |

### `slm diagnostics export <dest>`

Export bounded local operational aggregates. Produces a content-free JSON report (no memory content, no secrets) for manual inspection or support.

```bash
slm diagnostics export /path/to/report.json
```

### `slm benchmark`

Run the evo-memory benchmark against an isolated temporary database. Never reads or writes user data.

```bash
slm benchmark
slm benchmark --json
```

---

## Configuration and Adapters

### `slm config get|set <key> [value]`

Get or set runtime configuration values using dot notation.

```bash
slm config get evolution.enabled
slm config set evolution.enabled true
slm config set security.require_login true
slm config set retention.default_policy gdpr-30d
slm config set embedding.model nomic-embed-text-v1.5
```

Common keys:

| Key | Description |
|-----|-------------|
| `evolution.enabled` | Enable/disable skill evolution |
| `security.require_login` | Require login for dashboard and API |
| `retention.default_policy` | `indefinite`, `gdpr-30d`, `hipaa-7y`, `custom` |
| `embedding.model` | Embedding model name |
| `mesh.discovery` | `on`/`off` for mDNS peer discovery |

### `slm adapters [subcommand]`

Manage ingestion adapters: Gmail, Calendar, and Transcript.

```bash
slm adapters list                  # List all adapters and their status
slm adapters enable gmail          # Enable Gmail adapter
slm adapters disable gmail         # Disable Gmail adapter
slm adapters start gmail           # Start adapter ingestion
slm adapters stop gmail            # Stop adapter ingestion
slm adapters status gmail          # Check adapter status
```

### `slm ingest [--source ecc|jsonl]`

Import external observations into SLM's learning system.

```bash
slm ingest --source ecc            # Import Claude Code (ECC) sessions
slm ingest --source jsonl --file /path/to/data.jsonl
slm ingest --source ecc --dry-run  # Preview without writing
```

Supported sources:
- `ecc` — Claude Code sessions (auto-discovers ECC observation files)
- `jsonl` — Generic JSONL with `content` and optional `timestamp` fields

### `slm evolve [--session <id>] [--profile <id>]`

Run post-session skill evolution. Normally called automatically by the Stop hook; invoke manually to process a specific session.

```bash
slm evolve
slm evolve --session abc123 --profile work
```

---

## Bounded Loops (v3.8.0)

### `slm loop demo [--iterations N] [--json]`

Run the keyless convergence demo. A stub proposer runs laps against a deterministic gate (fails twice, passes on lap 3). Every lap is written to SLM memory under tag `loop:convergence-demo` and is visible on the dashboard.

```bash
slm loop demo
slm loop demo --iterations 5
slm loop demo --json
```

Confirms the bounded-loop engine, SLM ledger, and gate mechanism end to end without a credentialed agent or a running daemon. Use it to verify the installation after an upgrade.

### `slm loop history [--name NAME] [--json]`

List recorded runs for a loop name from the SLM ledger.

```bash
slm loop history                          # defaults to "convergence-demo"
slm loop history --name deploy-gate
slm loop history --name deploy-gate --json
```

| Option | Default | Description |
|--------|---------|-------------|
| `--name NAME` | `convergence-demo` | Loop name to query |
| `--json` | off | Emit structured JSON |

### `slm loop show <run_id> [--json]`

Show every lap of one run, in order.

```bash
slm loop show <run_id>
slm loop show <run_id> --json
```

`run_id` is printed by `slm loop demo` and returned by the `slm_loop_run` MCP tool. Each lap row records `decision`, gate pass/fail, detail text, and token budget.

The MCP equivalents are `slm_loop_run`, `slm_loop_history`, and `slm_loop_show`. See [MCP Tools Reference → Bounded-Loop Tools](mcp-tools.md#bounded-loop-tools-v380).

---

## Database Maintenance

### `slm db migrate [options]`

Inspect or run additive database schema migrations.

```bash
slm db migrate             # Apply pending migrations
slm db migrate --dry-run   # Preview pending migrations
slm db migrate --rollback  # Roll back to previous schema backup
```

### `slm db scale status|prepare|verify|promote|rollback|adopt`

Manage the optional Scale Engine (CozoDB graph + LanceDB vector projections).

```bash
slm db scale status                              # Show current Scale Engine state
slm db scale prepare                             # Stage a new projection
slm db scale verify --stage-id <id>             # Verify parity with canonical SQLite
slm db scale promote --stage-id <id>            # Promote verified projection
slm db scale rollback --backup-id <id>          # Roll back to a prior projection
slm db scale adopt                               # Adopt a detected pre-v3.7 projection
```

SQLite + sqlite-vec remain canonical. Projections are parity-gated; a failed verify leaves recall on SQLite with the rejected manifest retained for inspection.

---

*SuperLocalMemory V3 — Copyright 2026 Varun Pratap Bhardwaj. AGPL-3.0-or-later. Part of Qualixar.*
