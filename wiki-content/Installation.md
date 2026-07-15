# Installation

SuperLocalMemory V3 has two primary paths: an npm global CLI with a
package-owned Python environment, and a Python CLI + SDK inside an activated
virtual environment. Repository-clone installers share the same release
identity but have different ownership and verification contracts.

## Prerequisites

| Requirement | Version | Check |
|:-----------|:--------|:------|
| **Python** | 3.11+ | `python3 --version` |
| **Node.js** (for npm install) | 18+ | `node --version` |

Python 3.11+ is required for the V3 engine. Node.js is only needed if you install via npm.

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

### Verify

```bash
slm status
```

You should see:
```
SuperLocalMemory V3
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
the historical default stack, not a V3.7 release envelope:

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

### macOS (Apple Silicon + Intel)

```bash
npm install -g superlocalmemory
slm setup
```

Use an existing supported Python 3.11–3.14 runtime. The npm installer does not
bootstrap Homebrew, uv, pipx, or Python.

### Linux (Ubuntu/Debian/Fedora)

```bash
npm install -g superlocalmemory
slm setup
```

Ensure Python 3.11+ is installed: `sudo apt install python3.11` (Ubuntu) or `sudo dnf install python3.11` (Fedora).

### Windows

```bash
npm install -g superlocalmemory
slm setup
```

Requires an installed supported Python runtime. Hosted Windows artifact proof
must pass for the frozen V3.7 release before the channel is marked verified.

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

See [IDE Setup](IDE-Setup) for per-IDE instructions.

---

## Upgrading from V2

If you have V2 (2.8.6 or earlier) installed:

```bash
npm install -g superlocalmemory    # Installs V3 alongside V2
slm migrate                        # Migrates V2 data to V3 schema
```

V3 is a complete architectural reinvention — new mathematical engine, new retrieval pipeline, new storage schema. Your existing data is preserved. A backup is created automatically before migration.

See [Migration from V2](Migration-from-V2) for the full guide.

---

## Troubleshooting

### `slm: command not found`
- **npm install:** Make sure npm global bin is in your PATH. Run `npm bin -g` to find the location.
- **pip install:** Make sure Python scripts directory is in your PATH.

### `ModuleNotFoundError: No module named 'superlocalmemory'`
- Ensure Python 3.11+ is the default: `python3 --version`
- Activate the environment used for SLM, then reinstall with
  `python -m pip install --force-reinstall superlocalmemory`.

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
- [CLI Reference](CLI-Reference) — Current command guidance and installed-help contract

---
*Part of [Qualixar](https://qualixar.com) | Created by [Varun Pratap Bhardwaj](https://varunpratap.com)*
