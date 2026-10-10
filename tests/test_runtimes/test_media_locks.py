"""The committed hashed package lists for the media environment."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from superlocalmemory.runtimes.media_env import MEDIA_REQUIREMENTS

LOCKS = Path(__file__).resolve().parents[2] / "src" / "superlocalmemory" / "runtimes" / "locks"
NAME = re.compile(r"media-(darwin-arm64|linux-x86_64|windows-amd64)-py3(12|13|14)\.txt")
FILES = sorted(p for p in LOCKS.glob("media-*.txt"))


def test_a_lock_exists_for_the_cloud_platform():
    assert (LOCKS / "media-linux-x86_64-py312.txt").is_file()


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.name)
def test_lock_file_name_and_format(path):
    assert NAME.fullmatch(path.name)
    text = path.read_text(encoding="utf-8")
    pins = re.findall(r"^([A-Za-z0-9_.\-]+)==([^\s;\\]+)", text, flags=re.M)
    assert pins, "no pinned packages"
    blocks = re.split(r"\n(?=[A-Za-z0-9_.\-\[\]]+==)", text)
    for block in blocks:
        if re.match(r"[A-Za-z0-9_.\-\[\]]+==", block):
            assert "--hash=sha256:" in block, block.splitlines()[0]
    names = {n.lower().replace("_", "-") for n, _ in pins}
    for req in MEDIA_REQUIREMENTS:
        top = re.split(r"[=\[;<>]", req)[0].strip().lower().replace("_", "-")
        if top.startswith("pyobjc") and "darwin" not in path.name:
            continue
        assert top in names, f"{top} missing from {path.name}"
    for pin in ("torch==2.14.1", "transformers==5.19.0", "sentence-transformers==6.1.0"):
        assert re.search(rf"^{re.escape(pin)}\b", text, flags=re.M), pin


@pytest.mark.parametrize("path", [p for p in FILES if not p.name.startswith("media-darwin")], ids=lambda p: p.name)
def test_cpu_platforms_use_the_cpu_torch_index(path):
    assert "download.pytorch.org/whl/cpu" in path.read_text(encoding="utf-8")


def test_lock_script_targets_macos_14_for_darwin():
    import importlib.util

    path = Path(__file__).resolve().parents[2] / "scripts" / "lock_media_env.py"
    spec = importlib.util.spec_from_file_location("lock_media_env_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.build_env("darwin-arm64")["MACOSX_DEPLOYMENT_TARGET"] == "14.0"
    assert mod.build_env("linux-x86_64").get("MACOSX_DEPLOYMENT_TARGET") != "14.0"
