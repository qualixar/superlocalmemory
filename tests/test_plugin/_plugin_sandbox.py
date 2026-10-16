# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE file
"""Sandbox helpers for running the plugin's POSIX scripts for real.

Every run gets its own HOME, SLM_DATA_DIR and CLAUDE_PLUGIN_DATA under the
test's tmp_path, and a PATH built from scratch: a directory of stubs first,
then the system directories. Nothing the developer has installed (a pipx slm,
a real python3) can leak into the answer.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC_SCRIPTS = REPO / "plugin-src" / "scripts"
BUILT_PLUGIN = REPO / "plugin"

SYSTEM_PATH = "/usr/bin:/bin"


def system_path_has_slm() -> bool:
    return shutil.which("slm", path=SYSTEM_PATH) is not None


def requirements_sentinel_digest(plugin_root: Path) -> str:
    """Mirror ensure-venv.sh's `hash_req` exactly: sha256(requirements.txt),
    folded with sha256(requirements-cpu-torch.txt) when that file exists, then
    re-hashed once more so the sentinel stays a single opaque token either way.
    A hand test that precomputes the OLD single-file digest silently stops
    matching the real sentinel the day a CPU-torch pin file is added — this is
    the one place both sides read, so they cannot drift apart again.
    """
    req = plugin_root / "requirements.txt"
    combined = hashlib.sha256(req.read_bytes()).hexdigest()
    torch_pin = plugin_root / "requirements-cpu-torch.txt"
    if torch_pin.is_file():
        combined = f"{combined}:{hashlib.sha256(torch_pin.read_bytes()).hexdigest()}"
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


def write_exe(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)
    return path


def stub_slm(path: Path, tag: str, *, exit_code: int = 0, stderr: str = "") -> Path:
    """An slm stand-in that reports who it is, its argv and its stdin."""
    err = f'echo "{stderr}" >&2\n' if stderr else ""
    return write_exe(
        path,
        "#!/bin/sh\n"
        'case "$1" in --version) echo "superlocalmemory 9.9.9"; exit 0 ;; serve) exit 0 ;; esac\n'
        'IN="$(cat)"\n'
        f'echo "RAN:{tag} ARGS:$* STDIN:$IN"\n'
        f"{err}"
        f"exit {exit_code}\n",
    )


def stub_python(directory: Path, version: str) -> Path:
    """A python3 that answers the version guard as `version` would."""
    major, minor = (int(x) for x in version.split(".")[:2])
    ok = 0 if (major, minor) >= (3, 12) else 1
    return write_exe(
        directory / "python3",
        "#!/bin/sh\n"
        f'case "$*" in *--version*) echo "Python {version}"; exit 0 ;; '
        f'*sys.version_info*) exit {ok} ;; esac\n'
        'echo "unexpected python3 call: $*" >&2\n'
        "exit 97\n",
    )


def sandbox_env(tmp_path: Path, stub_dir: Path, **extra: str | None) -> dict[str, str]:
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    env = {
        "HOME": str(home),
        "PATH": f"{stub_dir}:{SYSTEM_PATH}",
        "SLM_DATA_DIR": str(tmp_path / "slm-data"),
        "CLAUDE_PLUGIN_ROOT": str(BUILT_PLUGIN),
        "CLAUDE_PLUGIN_DATA": str(tmp_path / "plugin-data"),
        "LANG": os.environ.get("LANG", "C"),
    }
    for key, value in extra.items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = value
    return env


def run(argv: list[str], env: dict[str, str], *, stdin: str = "", timeout: int = 30):
    return subprocess.run(
        argv, env=env, input=stdin, capture_output=True, text=True, timeout=timeout,
    )


def plugin_tree_lock():
    """Cross-process lock for tests that build or check the tracked plugin/ tree.

    Under pytest-xdist one worker can rebuild plugin/ while another runs
    ``build-plugin.mjs --check`` against it and sees a half-written tree.
    """
    from filelock import FileLock

    return FileLock(str(REPO / ".pytest-plugin-tree.lock"))
