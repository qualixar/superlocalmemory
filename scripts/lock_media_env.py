#!/usr/bin/env python3
"""Generate the hashed package lists for the media environment.

Writes ``src/superlocalmemory/runtimes/locks/media-<platform>-py<maj><min>.txt``
for each platform and Python minor version, using ``uv pip compile`` with
hashes. Linux and Windows resolve torch from the CPU index. Run it on a machine
that can reach the package indexes::

    python scripts/lock_media_env.py [--platform darwin-arm64 ...] [--python 3.12 ...]

Without a lock for a computer, the app reports images and documents as
unsupported there instead of guessing.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCKS = ROOT / "src" / "superlocalmemory" / "runtimes" / "locks"
sys.path.insert(0, str(ROOT / "src"))

TARGETS = {
    "darwin-arm64": "aarch64-apple-darwin",
    "linux-x86_64": "x86_64-unknown-linux-gnu",
    "linux-aarch64": "aarch64-unknown-linux-gnu",
    "windows-amd64": "x86_64-pc-windows-msvc",
}
PYTHONS = ("3.12", "3.13", "3.14")
MACOS_MIN = "14.0"
CPU_INDEX = "https://download.pytorch.org/whl/cpu"


def lock_filename(platform_tag: str, python: str) -> str:
    major, minor = python.split(".")[:2]
    return f"media-{platform_tag}-py{major}{minor}.txt"


def build_command(platform_tag: str, python: str, requirements: Path, output: Path) -> list[str]:
    cmd = ["uv", "pip", "compile", str(requirements), "--generate-hashes", "--emit-index-url",
           "--python-platform", TARGETS[platform_tag], "--python-version", python,
           "--output-file", str(output), "--no-header"]
    if not platform_tag.startswith("darwin"):
        cmd += ["--index-url", CPU_INDEX, "--extra-index-url", "https://pypi.org/simple",
                "--index-strategy", "unsafe-best-match"]
    return cmd


def build_env(platform_tag: str) -> dict[str, str]:
    """Environment for uv. torch ships its macOS wheels for macOS 14 and later only."""
    env = dict(os.environ)
    if platform_tag.startswith("darwin"):
        env["MACOSX_DEPLOYMENT_TARGET"] = MACOS_MIN
    return env


def main(argv: list[str] | None = None) -> int:
    from superlocalmemory.runtimes.media_env import MEDIA_REQUIREMENTS

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", action="append", choices=sorted(TARGETS))
    parser.add_argument("--python", action="append", choices=PYTHONS)
    args = parser.parse_args(argv)
    LOCKS.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        reqs = Path(tmp) / "requirements.in"
        reqs.write_text("\n".join(MEDIA_REQUIREMENTS) + "\n", encoding="utf-8")
        for tag in args.platform or sorted(TARGETS):
            for python in args.python or PYTHONS:
                out = LOCKS / lock_filename(tag, python)
                print(f"locking {out.name}")
                subprocess.run(build_command(tag, python, reqs, out), check=True, env=build_env(tag))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
