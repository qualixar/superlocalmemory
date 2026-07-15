# Installing SuperLocalMemory on Linux

SuperLocalMemory requires Python 3.11–3.14. Installation code and durable memory
data have separate ownership: installers manage the executable environment;
`SLM_DATA_DIR` selects memory data. No supported installer moves or deletes data.

## npm interactive path

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

## Python CLI

Install the command as an isolated Python application with either tool manager:

```bash
uv tool install superlocalmemory
# or
pipx install superlocalmemory --python python3.11

slm setup
slm doctor
```

Install uv or pipx using your operating system package manager and inspect its
official installation instructions before using another bootstrap method.

## Python library

Use a project virtual environment when importing SLM from Python:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install superlocalmemory
```

Do not use `sudo pip`, do not override an externally managed Python, and do not
install the same `slm` command through multiple tool managers.

## Repository clone

Researchers and contributors can install the checked-out source through an
already-installed uv or pipx:

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
