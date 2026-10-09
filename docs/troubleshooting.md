# Troubleshooting

Solutions for common issues. If your problem is not listed here, run `slm doctor`
and `slm status --json` and check the output for clues.

`slm doctor` checks dependencies, the embedding worker, daemon connectivity and
configuration. `slm doctor --fix` first repairs what it can (re-downloads missing
models, installs sqlite-vec) and then reports. `slm doctor --quick` runs only the
fast checks, and `slm doctor --deep` reads every database page.

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

**Installed with `uv tool install`?** The command lives in uv's tool bin
directory (`uv tool dir --bin`, usually `~/.local/bin`). Run
`uv tool update-shell` and open a new shell.

### "Python not found" during setup

SLM requires Python 3.12 or later, up to 3.14.

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

# Option 2: install inside an activated Python virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install superlocalmemory
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

**Check for a filter:** a `--project`, `--kind`, `--tag` or `--window` narrows
the search, and `--known-as-of` or `--valid-at` can hide recent memories. Run the
recall again without them. An empty result with a `tag_scope` or `project_scope`
note says what the filter did. See [Recall](recall.md).

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

Fewer results means only the top matches are returned.

**Use trace to debug:**

```bash
slm trace "your query"
```

This shows which channels contributed what. If BM25 is dominating with weak keyword matches, the query may need different terms.

**Re-derive the graph:**

```bash
slm db regraph --check    # how far the graph copy has drifted; changes nothing
slm db regraph            # re-derive it from the store
```

Use it when entity relationships look wrong or after bulk imports.

## Answer Check Issues

See [Answer Check](answer-check.md) for the full feature. Quick fixes:

### "On this Mac" isn't offered / "Needs a Mac with Apple Silicon"

The on-device option needs Apple Silicon. Choose **Online with Jev** in
Settings → Answer check instead, or leave the check off.

### Setup fails with a download error

If the dashboard shows *"Couldn't reach the download server. If your
network blocks downloads, use 'Use an existing install'"* — common on a
locked-down company network — open **Advanced — use an existing install**
and point it at a Python interpreter and model copied over from another
machine, instead of downloading again.

### A saved key won't test successfully

- *"The key was not accepted."* — re-check or regenerate the key.
- *"The account has no credit left."* — add credit with your provider.
- *"Too many requests — try again in a minute."* — wait, then retry; keys
  are rate-limited to one test every few seconds.

### Every recall reports the check as unjudged on one of several SLM processes

Expected when more than one SLM process runs on the same machine at once —
only one of them runs the on-device model; the others report recalls as if
the check were off rather than loading a second copy of it.

### Uninstalling SLM doesn't remove the on-device model

Uninstalling leaves the on-device answer-check model (about 1.1 GB) in
`~/.superlocalmemory/runtimes/laya`; choose **Remove** in Settings → Answer
check first, or delete that folder afterwards.

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
# Re-run the provider connection test
slm provider set <provider>

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
slm mode b    # A model on this machine (no network)
slm mode a    # No language model (fastest)
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

Embeddings and the graph are rebuilt from existing data, which can take a moment
on a large database. Check and complete them:

```bash
slm db integrity            # read-only health report
slm db reembed              # backfill facts that never got an embedding
slm db regraph              # re-derive the graph copy
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

4. **Check the install:**

```bash
slm doctor
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

Each IDE has its own MCP config file. They do not conflict. All IDEs share the same underlying database through the SLM daemon, which is the single writer.

## Database Issues

### Check the memory store

```bash
slm db integrity            # read-only; safe while SLM runs
slm db integrity --pages    # also reads every page (slow on a large store)
slm db repair --root ~/.superlocalmemory   # preview what a repair would do
```

See [Memory store check](#memory-store-check) below.

### Database corruption

Extremely rare with SQLite WAL mode. If `slm db integrity --pages` reports the
file itself is unsound:

1. **Stop the daemon first** — a live `cp` of `memory.db` alone is unsafe (WAL/SHM may be uncheckpointed and companion stores diverge):

   ```bash
   slm serve stop
   ```

2. **Restore a complete, verified data-root/store-set backup** taken with the daemon stopped. Include `memory.db` plus sidecars (`memory.db-wal`, `memory.db-shm`) and any present `lance/` or other store directories — not a single-file `cp` over a running daemon. Backups taken via `BackupManager` are per-file `sqlite3.backup()` snapshots; a coherent offline copy is a whole-root copy with the daemon stopped so WAL checkpoints (see `SECURITY.md` Backup and credential-at-rest caveats).

3. **Verify the restore** before restarting:

   ```bash
   slm status --json
   sqlite3 ~/.superlocalmemory/memory.db "PRAGMA integrity_check;"
   ls -l ~/.superlocalmemory/memory.db* ~/.superlocalmemory/lance 2>&1 | head -20
   slm serve start
   slm status --json
   ```

   Do not rely on `slm migrate --rollback` for schema downgrade — `slm db migrate` is **forward-only** (`status`/`--dry-run`/apply, no rollback) and there is no supported downgrade path except restoring a verified pre-upgrade complete backup.

### Database is too large

```bash
# Check size: the JSON payload reports memory.db size and row counts.
slm status --json

# Decay applies the lifecycle policy: it fades and archives stale memories.
slm decay              # preview
slm decay --execute

# Drop old LanceDB vector-store versions (only on a store that uses LanceDB).
slm db compact
```

## Native libraries on macOS

On a Mac, NumPy, SciPy and PyTorch (which SLM uses for its maths and its
embedding model) hand linear algebra to Apple's Accelerate framework. Two
kinds of work go there:

- **Matrix products and vector lengths.** Everything SLM does when it saves,
  recalls, runs background maintenance or computes embeddings is this kind.
- **Matrix factorizations** (QR, Cholesky, SVD and the solvers built on
  them). Accelerate has reported defects here on recent macOS releases: QR
  writes outside its memory for square matrices of roughly 576 to 1,000 rows
  (found by us; before 4.1.20 it could crash SLM at random), Cholesky can
  crash at very large sizes ([scipy#26145](https://github.com/scipy/scipy/issues/26145)), SVD can
  hang on a matrix that contains an infinite value
  ([numpy#32591](https://github.com/numpy/numpy/issues/32591)), and one
  symmetric solver no longer notices a singular matrix on macOS 26.5
  ([scipy#25313](https://github.com/scipy/scipy/issues/25313)).

What SLM does about it:

- Saving, recall, maintenance and embedding make **no** factorization calls.
  The test suite counts every call into Accelerate's factorization routines
  while it saves, recalls and maintains a store, and fails if there is one.
- The random rotation used to compress embeddings is computed without
  Accelerate's QR since 4.1.20.
- One feature still uses a factorization: learning the match threshold of the
  experimental semantic cache in SLM Optimize (off by default). Its matrix is
  never larger than 10 x 10, far below any reported failure size, and a test
  runs it under macOS Guard Malloc, which stops the process on the first
  out-of-bounds write.

Nothing needs to be configured. If SLM ever quits unexpectedly on a Mac and
the crash report names `libLAPACK` or `Accelerate`, please open an issue with
that report and the output of `slm status --json`.

## Memory store check

SLM checks your memory store once after each upgrade, a few minutes after it
starts. The check only reads and counts; it never changes the store. The result
appears in the dashboard under **Health**, in the **Memory store** card, in plain
words: for example search vectors that no longer match their memory, LanceDB
entries for memories that are gone, leftover rows from deleted memories, erased
words still stored, index updates that never finished, and memories that lost
their searchable fact. **Check again** runs it on demand.

If something is listed, **Repair now** fixes it. It first saves a full backup copy
of your memory, and if that copy cannot be made nothing is changed. Then it runs
the same repair as `slm db repair --apply` and checks again. Recall keeps working
meanwhile, and the repair removes no memory you have. The card shows the backup's
location. From a terminal, use `slm db integrity` and `slm db repair`; see the
[CLI reference](cli-reference.md#embedding-models-and-store-health).

If the card says the check or repair did not finish, it shows the reason. Run
`slm doctor`, look at `logs/daemon.log` in the data folder, and try again.

## Web access

Web access connects an AI app on the internet to your memory; see [Web
access](remote-access/README.md). Problems show up on the dashboard's
**Connected apps** page or as an error code in the app.

**The page says Connected apps are not available right now.** The local service
is not running or cannot be reached. Run `slm status`, then `slm restart`.

**Sign-in does not finish or the link expired.** Use **Continue sign-in** if your
browser blocked the page, or **Restart sign-in**, which keeps the permissions you
chose. Sign in with the same GitHub account in the app.

**The app says the computer is asleep or offline** (`connector_asleep`,
`connector_offline`). Web access needs this computer on and online. The gateway
treats it as asleep after 45 seconds with no heartbeat. Wake it, check its
network, and try again. A sleep never shows up as an empty recall.

**`DAILY_LIMIT_REACHED`.** The free daily allowance of tool calls is used up. It
resets at midnight UTC. Local use is not affected.

**`relay_busy` or `relay_timeout`.** Too many calls at once, or one took longer
than 25 seconds. Make one call at a time and try again.

**`TOOL_DENIED` or `INSUFFICIENT_SCOPE`.** The app was not given that permission,
for example saving. Remove the app under **Your connected apps** and add it again
with the permission ticked.

**`REVOKED` or `ENTITLEMENT_REQUIRED`.** The app was removed, or Web access has
ended. Turn it on again on the Connected apps page. If it says **Web access ends
on (date)** or **Sign in again to keep Web access working**, check that this
computer is online and sign in again.

**The app saves nothing.** Saving is a separate permission. Check the app's row
under **Your connected apps**: it shows **Save** only when that was allowed.

## Health Check

Run a diagnostic:

```bash
slm health
```

It reports the number of memories, how many have a similarity-layer entry, how
many have a lifecycle position, and the current mode. For dependencies, the
embedding worker and the daemon, use `slm doctor`; for the database, use
`slm db integrity`.

## Getting Help

If none of the above resolves your issue:

1. Run `slm status --json` and note the output
2. Check the [GitHub Issues](https://github.com/qualixar/superlocalmemory/issues)
3. Open a new issue with your `slm status --json` output

---

*SuperLocalMemory — Copyright 2026 Varun Pratap Bhardwaj. AGPL-3.0-or-later. Part of Qualixar.*
