# CLI Reference
> SuperLocalMemory V3 Documentation
> https://superlocalmemory.com | Part of Qualixar

Complete reference for the `slm` command-line interface.

---

## Setup & Configuration

### `slm setup`

Run the interactive setup wizard. Guides you through mode selection, IDE connection, and verification.

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

Supported IDEs: `claude`, `cursor`, `vscode`, `windsurf`, `gemini`, `jetbrains`, `continue`, `zed`

## Memory Operations

### `slm remember "content" [options]`

Store a memory.

```bash
slm remember "API rate limit is 100 req/min on staging"
slm remember "Use camelCase for JS, snake_case for Python" --tags "style,convention"
slm remember "Maria owns the auth service" --tags "team,ownership"
```

| Option | Description |
|--------|-------------|
| `--tags "a,b"` | Comma-separated tags for categorization |
| `--profile name` | Store in a specific profile (overrides active profile) |

### `slm recall "query" [options]`

Search your memories. Returns the most relevant results.

```bash
slm recall "rate limit"
slm recall "who owns auth" --limit 5
slm recall "database config" --profile work
```

| Option | Default | Description |
|--------|---------|-------------|
| `--limit N` | 10 | Maximum results to return |
| `--profile name` | active | Search in a specific profile |

### `slm search "query" [options]`

Alias for `slm recall`. Same behavior, same options.

### `slm forget "query" [options]`

Delete memories matching a query.

```bash
slm forget "old staging credentials"
slm forget --id 42                    # Delete by memory ID
slm forget --before "2026-01-01"      # Delete memories before a date
```

| Option | Description |
|--------|-------------|
| `--id N` | Delete a specific memory by ID |
| `--before "date"` | Delete all memories before this date |
| `--confirm` | Skip the confirmation prompt |

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

Output shows scores from each channel:
- Semantic (vector similarity)
- BM25 (keyword matching)
- Entity Graph (relationship traversal)
- Temporal (time-based relevance)

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

### `slm consistency`

Run a consistency check across your memories. Detects contradictions and outdated information.

```bash
slm consistency
```

## Migration

### `slm migrate [options]`

Migrate a V2 database to V3 format.

```bash
slm migrate                # Run migration
slm migrate --dry-run      # Preview what will change
slm migrate --rollback     # Undo migration (within 30 days)
```

| Option | Description |
|--------|-------------|
| `--dry-run` | Show what would change without modifying anything |
| `--rollback` | Revert to V2 format (backup must exist) |

## Profile Management

### `slm profile [command]`

Manage memory profiles (isolated memory contexts).

```bash
slm profile list                  # List all profiles
slm profile switch work           # Switch to "work" profile
slm profile create client-acme    # Create a new profile
slm profile delete old-project    # Delete a profile
slm profile export work > backup.json  # Export a profile
```

## System & Maintenance

### `slm status`

Show system status: mode, profile, memory count, database location, health.

```bash
slm status
```

### `slm compact`

Compress and optimize the memory database. Merges redundant memories and reclaims space.

```bash
slm compact
```

### `slm backup`

Check backup status or create a manual backup.

```bash
slm backup              # Show backup status
slm backup create       # Create a backup now
```

### `slm audit [options]`

View the audit trail. Shows all memory operations with timestamps and hash-chain verification.

```bash
slm audit               # Recent audit entries
slm audit --limit 100   # Last 100 entries
slm audit --verify      # Verify hash-chain integrity
```

### `slm retention [policy]`

Manage retention policies.

```bash
slm retention                          # Show current policy
slm retention set gdpr-30d            # Apply GDPR 30-day policy
slm retention set hipaa-7y            # Apply HIPAA 7-year policy
slm retention set custom --days 90    # Custom retention period
```

## Global Options

These options work with any command:

| Option | Description |
|--------|-------------|
| `--help` | Show help for a command |
| `--version` | Show SLM version |
| `--verbose` | Show detailed output |
| `--json` | Output structured JSON with agent-native envelope (for AI agents, scripts, CI/CD) |
| `--profile name` | Override the active profile for this command |

## Agent-Native JSON Output

All data-returning commands support `--json` for structured output. The envelope follows the 2026 agent-native CLI standard:

```json
{
  "success": true,
  "command": "recall",
  "version": "3.0.22",
  "data": {
    "results": [
      {"fact_id": "abc123", "score": 0.87, "content": "Database uses PostgreSQL 16"}
    ],
    "count": 1,
    "query_type": "semantic"
  },
  "next_actions": [
    {"command": "slm list --json", "description": "List recent memories"}
  ]
}
```

### Supported Commands

`recall`, `remember`, `list`, `status`, `health`, `trace`, `forget`, `delete`, `update`, `mode`, `profile`, `connect`

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

*SuperLocalMemory V3 — Copyright 2026 Varun Pratap Bhardwaj. AGPL-3.0-or-later. Part of Qualixar.*
