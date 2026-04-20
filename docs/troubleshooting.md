# Troubleshooting
> SuperLocalMemory V3 Documentation
> https://superlocalmemory.com | Part of Qualixar

Solutions for common issues. If your problem is not listed here, run `slm status --json` and check the output for clues.

<!-- UX-G1: document the planned slm doctor --fix auto-repair mode so the
     health-ladder self-healing surface has a user-facing verb in docs now,
     even though the CLI flag lands in a follow-up cycle. -->
> **Planned (next cycle): `slm doctor --fix`** — an auto-repair mode that
> applies the safe, reversible fixes `slm doctor` identifies (rebuild
> trigram cache, stop orphan worker processes, WAL truncate, port reset).
> Today, run `slm doctor` to see findings and apply the recommended
> commands manually. The `--fix` flag lands in a follow-up release;
> tracked as Stage 8 UX-G1.

---

## Installation Issues

### "slm: command not found"

The npm global bin directory is not in your shell's PATH.

**Fix:**

```bash
# Find where npm puts global binaries
npm root -g
# Example output: /usr/local/lib/node_modules

# The bin directory is one level up
# Add to your shell profile (~/.zshrc, ~/.bashrc, or ~/.bash_profile):
export PATH="$(npm prefix -g)/bin:$PATH"

# Reload your shell
source ~/.zshrc   # or source ~/.bashrc
```

**Alternative — use npx:**

```bash
npx superlocalmemory status
```

### "Python not found" during setup

SLM V3 requires Python 3.10 or later for the math engine.

```bash
# Check Python version
python3 --version

# Install if missing
# macOS:
brew install python@3.12

# Ubuntu/Debian:
sudo apt install python3.12

# Windows:
winget install Python.Python.3.12
```

### "Permission denied" during install

```bash
# Option 1: Fix npm permissions (recommended)
mkdir -p ~/.npm-global
npm config set prefix '~/.npm-global'
export PATH=~/.npm-global/bin:$PATH
npm install -g superlocalmemory

# Option 2: Use sudo (not recommended)
sudo npm install -g superlocalmemory
```

## Recall Issues

### "No memories found" when you know they exist

**Check your active profile:**

```bash
slm profile list
```

Memories are profile-scoped. If you stored a memory in the `work` profile but are currently in `default`, it will not appear.

```bash
slm profile switch work
slm recall "your query"
```

**Check your mode:**

```bash
slm mode
```

Mode A uses math-based retrieval. If you recently switched from Mode C, the retrieval behavior changes. Both modes search the same memories, but ranking differs.

**Try a broader query:**

```bash
slm recall "database"          # Instead of "PostgreSQL 16 configuration on staging"
slm list --limit 20            # Browse recent memories directly
```

### Recall returns irrelevant results

**Lower your result count:**

```bash
slm recall "your query" --limit 3
```

Fewer results means only the highest-confidence matches are returned.

**Use trace to debug:**

```bash
slm trace "your query"
```

This shows which channels contributed what. If BM25 is dominating with weak keyword matches, the query may need different terms.

**Rebuild the graph:**

```bash
slm consolidate --cognitive
```

`consolidate --cognitive` rebuilds the entity graph (and pattern index) from your memories; use it when entity relationships look wrong or after bulk imports.

## Mode C Issues

### "API key not set" or authentication errors

```bash
# Check your provider configuration
slm provider

# Reset your provider and key
slm provider set openai
# Enter your API key when prompted

# Or set via environment variable
export OPENAI_API_KEY="sk-..."
```

### "Connection timeout" or network errors

```bash
# Test connectivity to your provider
slm status --json

# Check if you're behind a proxy
echo $HTTP_PROXY
echo $HTTPS_PROXY
```

If behind a corporate proxy, set the proxy variables:

```bash
export HTTPS_PROXY="http://proxy.company.com:8080"
```

### Mode C is slow

Cloud LLM calls add latency. If speed matters more than maximum recall quality:

```bash
slm mode b    # Local LLM (fast, no network)
slm mode a    # Math-only (fastest)
```

## Migration Issues

### Migration failed or was interrupted

```bash
# Check if backup exists
ls ~/.superlocalmemory/backups/

# Roll back to V2
slm migrate --rollback

# Try migration again
slm migrate
```

### "Database is locked"

Close all IDE sessions that might be accessing SLM, then retry:

```bash
# Check for processes using the database
lsof ~/.superlocalmemory/memory.db

# Close IDEs, then retry
slm migrate
```

### Migration succeeded but recall quality seems worse

After migration, the entity graph and BM25 index are rebuilt from existing data. This process is automatic but can take a moment for large databases.

```bash
# Force a full re-index + consolidation pass
slm consolidate --cognitive
slm decay
```

## IDE Connection Issues

### IDE does not show SLM tools

1. **Verify SLM is installed:**

```bash
npm list -g superlocalmemory
```

2. **Check the IDE config file has correct JSON:**

```bash
slm connect <your-ide>    # Regenerates the config
```

3. **Restart the IDE completely** (not just reload the window).

4. **Test the MCP server directly:**

```bash
npx superlocalmemory mcp --test
```

### "Connection refused" in IDE

The MCP server failed to start. Common causes:

- Node.js version too old (need 18+)
- Port conflict with another service
- Corrupted installation

```bash
# Reinstall
npm uninstall -g superlocalmemory
npm install -g superlocalmemory

# Verify
slm status
```

### Multiple IDEs conflicting

Each IDE has its own MCP config file. They do not conflict. All IDEs share the same underlying database. Concurrent access is safe (SQLite WAL mode handles this).

## Database Issues

### Database corruption

Extremely rare with SQLite WAL mode, but if it happens:

```bash
# Check integrity
slm status --json

# Restore from automatic backup
ls ~/.superlocalmemory/backups/
cp ~/.superlocalmemory/backups/memory-2026-03-15.db ~/.superlocalmemory/memory.db
```

### Database is too large

```bash
# Check size — the JSON payload reports memory.db size + row counts.
slm status --json

# Consolidate (merges redundant memories, reclaims space)
slm consolidate --cognitive

# Decay applies the lifecycle policy to fade / archive stale memories.
# Tune retention via the configuration file (see docs/configuration.md).
slm decay
```

## Health Check

Run a full diagnostic:

```bash
slm health
```

This reports the status of:

| Component | What it checks |
|-----------|---------------|
| Database | Integrity, size, table counts |
| Embedding model | Loaded, version, dimension |
| Fisher-Rao | Similarity layer active |
| Sheaf | Consistency layer active |
| Langevin | Lifecycle layer active |
| BM25 index | Token count, index health |
| Entity graph | Node count, edge count |

If any component shows an error, the output includes a suggested fix.

## Getting Help

If none of the above resolves your issue:

1. Run `slm status --json` and note the output
2. Check the [GitHub Issues](https://github.com/qualixar/superlocalmemory/issues)
3. Open a new issue with your `slm status --json` output

---

*SuperLocalMemory V3 — Copyright 2026 Varun Pratap Bhardwaj. AGPL-3.0-or-later. Part of Qualixar.*
