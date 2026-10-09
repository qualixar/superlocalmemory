# Installing SuperLocalMemory on Linux

> **Linux support:** SuperLocalMemory supports 64-bit Linux only. Packaging metadata does not hard-block unsupported architectures; unsupported platforms fail at runtime dependency resolution rather than at install metadata.

SuperLocalMemory requires Python 3.12–3.14. Installation code and durable memory
data have separate ownership: installers manage the executable environment;
`SLM_DATA_DIR` selects memory data. No supported installer moves or deletes data.

## Primary path 1: npm global CLI

Use this when you want the guided profile setup and the `slm` command without
managing a Python environment yourself.

```bash
npm install -g superlocalmemory
slm setup
slm doctor
```

Node 18+ is required. npm creates a package-owned Python virtual environment.
It does not install into the operating system's Python and does not run setup,
install hooks, start a daemon, or download models during `npm install`.

## Primary path 2: Python CLI + SDK in an activated virtual environment

Use a dedicated virtual environment for both the `slm` command and Python API:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install superlocalmemory
```

Do not use `sudo pip`, do not override an externally managed Python, and do not
install the same `slm` command through multiple tool managers.

## Isolated `slm` command with uv

If you use [uv](https://docs.astral.sh/uv/), install SLM as a uv tool. uv gives
the package its own environment and puts only the `slm` command on your PATH
(in `~/.local/bin`). Your system Python is left untouched and there is nothing
to activate before running `slm`.

```bash
uv tool install --python 3.12 superlocalmemory
slm setup
slm doctor
```

`--python 3.12` pins the tool environment to a supported interpreter; uv
downloads a Python build when no matching one is installed. `slm doctor` checks
that this interpreter can load the SQLite vector extension. If `slm` is not
found afterwards, run `uv tool update-shell` and open a new shell.

Upgrade and uninstall through uv as well. Stop the daemon before upgrading so
it does not keep running on a mix of old and new package files:

```bash
slm serve stop
uv tool upgrade superlocalmemory
slm restart && slm doctor

uv tool uninstall superlocalmemory   # removes code only; memory data is preserved
```

The uv tool environment is not an activated virtual environment, so the Python
SDK (`import superlocalmemory`) is not available from your own scripts. Use the
activated virtual environment above when you need the SDK.

## Repository clone

Researchers and contributors can install the checked-out source through the
scoped repository lifecycle installer (which delegates to an existing uv or
pipx installation):

```bash
git clone https://github.com/qualixar/superlocalmemory.git
cd superlocalmemory
./scripts/install.sh install
```

Preview the exact command with `--dry-run`. Use `upgrade` or `uninstall` with
the same script and tool manager. Uninstall removes code only; memory data is
preserved.

## Optional systemd service

First verify the isolated installation with `slm doctor`. Generate or install a
service definition only after choosing the final data root; the service must
persist `SLM_DATA_DIR` and the executable path from the owning tool environment.

For LAN access and authentication requirements, see
[distributed deployment](distributed-deployment.md).
