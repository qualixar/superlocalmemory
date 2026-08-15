# Installation — V4.0.4

SuperLocalMemory V4 has two primary paths: an npm global CLI with a
package-owned Python environment, and a Python CLI + SDK inside an activated
virtual environment. Repository-clone installers share the same release
identity but have different ownership and verification contracts.

> **Current release:** **V4.0.4** (`2026-08-15`, see [CHANGELOG](https://github.com/qualixar/superlocalmemory/blob/main/CHANGELOG.md)). All commands below describe the installed V4.0.4 artifact.

## Prerequisites

| Requirement | Version | Check |
|:-----------|:--------|:------|
| **Python** | 3.11 – 3.14 | `python3 --version` |
| **Node.js** (for npm install) | 18+ | `node --version` |

Python 3.11+ is required for the V4 engine (built on the V3.8 control plane). Node.js is only needed if you install via npm.

> **Platform boundary (V4):** **Apple Silicon macOS, 64-bit Windows, 64-bit Linux.** Intel Mac and 32-bit Windows are **not supported** by the pinned `cryptography==50.0.0` runtime — the package has no wheel for those architectures. `package.json` `os: [darwin, linux, win32]` is the npm publish descriptor and does not hard-block architectures; install will fail where `cryptography==50.0.0` wheels are absent (see `pyproject.toml` `cryptography==50.0.0`). The Python native dependency enforces the narrower boundary above.

---

## Primary path 1: npm global CLI

This installs the CLI and MCP runtime into a package-owned Python environment.

```bash
npm install -g superlocalmemory
```

The npm lifecycle does not mutate protected system Python, install hooks, edit
IDE configuration, start a daemon, download a model, or create the memory data
root. Activation is explicit:

```bash
slm setup     # Choose mode and integrations
slm warmup    # Pre-download embedding model (~500MB, one-time)
slm doctor    # Verify the installed runtime and configuration
```

Hooks remain opt-in through `slm setup` or `slm hooks install`.

### Upgrade existing host integrations

Package installation updates the SLM executable, but deliberately does not
rewrite agent configuration, hooks, or plugins. Inspect the existing
integrations first, then explicitly approve an upgrade:

```bash
slm upgrade-hosts
slm upgrade-hosts --host codex --apply
# or, after reviewing the preview:
slm upgrade-hosts --all-detected --apply
```

See [Host Integration Upgrades](Host-Upgrades) for Claude Code, Codex, and
portable MCP-host behavior.

### Verify

```bash
slm status
```

You should see:
```
SuperLocalMemory V4
  Mode: A
  Provider: none
  Base dir: /home/you/.superlocalmemory
  Database: /home/you/.superlocalmemory/memory.db
```

---

## Primary path 2: Python CLI + SDK in an activated virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install superlocalmemory
slm setup
slm doctor
```

Keep the environment active whenever you run `slm` or import the SDK. Do not
use global pip, `sudo pip`, or externally-managed-system-Python overrides.

## Repository clone (research and development)

```bash
git clone https://github.com/qualixar/superlocalmemory.git
cd superlocalmemory
./scripts/install.sh install   # macOS/Linux; requires existing uv or pipx
# Windows PowerShell: .\scripts\install.ps1 -Action Install
```

Then:

```bash
slm setup
slm warmup
slm status
```

---

## Resource expectations

Dependency and model footprints vary by Python platform, resolver, selected
backend, and configured embedding model. The values below are orientation from
the historical default stack, not a V4 release envelope:

| Component | Size | When |
|:----------|:-----|:-----|
| Core math libraries (numpy, scipy, networkx) | ~50MB | During install |
| Search engine (sentence-transformers, einops, torch) | ~200MB | During install |
| Embedding model (nomic-ai/nomic-embed-text-v1.5, 768d) | ~500MB | First use or `slm warmup` |

**Historical orientation:** ~750MB after first use (mostly PyTorch + an
embedding model). Measure the frozen artifact on each supported platform before
using this value for capacity planning.

**Historical orientation:** ~500-800MB peak during default embedding-model
load and ~20-50MB steady state. Backend and model selection can change this
materially.

If an optional retrieval dependency is unavailable, inspect `slm doctor`,
health, and trace output. Do not assume degraded retrieval is equivalent to the
declared full topology.

---

## Platform Notes

### Apple Silicon macOS (supported)

```bash
npm install -g superlocalmemory
slm setup
```

Use an existing supported Python 3.11–3.14 runtime. The npm installer does not
bootstrap Homebrew, uv, pipx, or Python.

> Intel Mac is **not supported** in V4 (pinned `cryptography==50.0.0` has no Intel macOS wheel).

### 64-bit Linux (Ubuntu/Debian/Fedora — supported)

```bash
npm install -g superlocalmemory
slm setup
```

Ensure Python 3.11+ is installed: `sudo apt install python3.11` (Ubuntu) or `sudo dnf install python3.11` (Fedora).

### 64-bit Windows (supported)

```bash
npm install -g superlocalmemory
slm setup
```

Requires an installed supported Python runtime (3.11–3.14, 64-bit).

> 32-bit Windows (Win32) is **not supported** in V4.

Hosted Windows artifact proof must pass for the frozen release before the channel is marked verified (see `benchmark/` — evidence is currently macOS-only; do not claim cross-platform verification beyond the installed runtime's `slm doctor`).

---

## MCP Integration (IDE Setup)

After installing, connect to your AI IDE:

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

Or auto-configure all detected IDEs:

```bash
slm connect        # Configure all detected IDEs
slm connect --list # See which IDEs are configured
```

See [IDE Setup](IDE-Setup) for per-IDE instructions. MCP tool counts in V4.0.4: `core` 14 / `code` 29 / `full` 47 / `power` 59 / `mesh` 8 / **`whole` 92** (see [MCP Tools](MCP-Tools)).

---

## Upgrading from V2

If you have V2 (2.8.6 or earlier) installed — **V2→V3 migration** (spans file copies, commits, and rename/symlink — not a single global transaction; verify after):

```bash
npm install -g superlocalmemory    # Installs V4 alongside V2
slm migrate                        # V2→V3 data migration (V2Migrator)
# Rollback only while the created backup still exists — verify before use:
ls -lh ~/.superlocalmemory/memory-v2-backup.db; ls -ld ~/.claude-memory-v2-original
slm migrate --rollback             # Valid only while migration backup still exists (no automatic 30-day deletion)
```

V3 is a complete architectural reinvention — new mathematical engine, new retrieval pipeline (5 candidate producers + graph enhancement), new storage schema. A backup is created (`~/.superlocalmemory/memory-v2-backup.db` / `~/.claude-memory-v2-original`); operators must verify (`slm status`, `slm health`, `slm status --json | jq '.data.fact_count'`) before decommissioning the prior state. This is not a zero-data-loss global transaction guarantee.

**Not to be confused with `slm db migrate`** — the V4 additive schema maintenance command (forward only — see below and [CLI Reference](CLI-Reference)):

```bash
slm db migrate --status            # Inspect forward/deferred migrations
slm db migrate --dry-run           # Preview (no writes)
slm db migrate                     # Apply pending additive migrations (forward only; no rollback)
# V4.0.0 M038 (eager) + M039 (deferred) are auto-applied at startup; no manual command normally required.
# Schema downgrade is unsupported; restore a verified pre-upgrade backup of the complete data root (stop daemon, include WAL/SHM).
```

See [Migration from V2](Migration-from-V2) for the full V2→V3 guide.

---

## Troubleshooting

### `slm: command not found`
- **npm install:** Make sure npm global bin is in your PATH. Run `npm bin -g` to find the location.
- **pip install:** Make sure Python scripts directory is in your PATH.

### `ModuleNotFoundError: No module named 'superlocalmemory'`
- Ensure Python 3.11+ is the default: `python3 --version`
- Activate the environment used for SLM, then reinstall with
  `python -m pip install --force-reinstall superlocalmemory`.

### `cryptography` install fails on Intel Mac or Win32
- Expected: V4's pinned `cryptography==50.0.0` has no wheel for those architectures — packaging metadata does not hard-block them, but install will fail where `cryptography==50.0.0` wheels are absent. Use Apple Silicon macOS, 64-bit Windows, or 64-bit Linux.

### Embedding model fails to download
- Check internet connection
- Try manual warmup: `slm warmup`
- If behind a proxy, set `HTTP_PROXY` and `HTTPS_PROXY` environment variables

### Permission errors on macOS/Linux
- Use `npm install -g superlocalmemory` (not sudo)
- If npm global directory needs permissions: `npm config set prefix ~/.npm-global` and add `~/.npm-global/bin` to PATH

---

## Next Steps

- [Quick Start Tutorial](Quick-Start-Tutorial) — Your first memory in 2 minutes
- [Modes Explained](Modes-Explained) — Choose between A (zero-cloud), B (local Ollama), C (full power)
- [CLI Reference](CLI-Reference) — Current command guidance and installed-help contract (`slm --help` is the source of truth)
- [MCP Tools](MCP-Tools) — V4.0.4 profile counts and the whole 92 distinction

---
*Part of [Qualixar](https://qualixar.com) | Created by [Varun Pratap Bhardwaj](https://varunpratap.com)*
