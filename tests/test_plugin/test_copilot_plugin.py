# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""P4d: the GitHub Copilot plugin ships at parity with Claude + Codex plugins,
generated from the single plugin-src/ source and kept in sync (drift guard).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
CP = REPO / "copilot-plugin"
MANIFEST = json.loads((REPO / "plugin-src" / "manifest.json").read_text("utf-8"))
VERSION = MANIFEST["version"]
SKILLS = [s["name"] for s in MANIFEST["skills"]]
AGENTS = sorted(p.stem for p in (REPO / "plugin-src" / "agents").glob("*.md"))


def test_core_structure_present() -> None:
    assert (CP / ".github" / "copilot-instructions.md").exists()
    assert (CP / ".github" / "hooks" / "slm-hooks.json").exists()
    assert (CP / ".vscode" / "mcp.json").exists()
    assert (CP / "README.md").exists()
    assert (CP / "_GENERATED.md").exists()
    for s in ("ensure-venv.sh", "ensure-venv.bat", "slm-launch", "slm-launch.bat"):
        assert (CP / "scripts" / s).exists(), f"missing script {s}"


def test_one_prompt_per_skill() -> None:
    prompts = {p.name for p in (CP / ".github" / "prompts").glob("*.prompt.md")}
    assert prompts == {f"{s}.prompt.md" for s in SKILLS}, "prompt set != skill set"


def test_one_agent_per_source_advisor_with_vscode_target() -> None:
    for a in AGENTS:
        f = CP / ".github" / "agents" / f"{a}.agent.md"
        assert f.exists(), f"missing agent {a}"
        text = f.read_text("utf-8")
        assert "target: vscode" in text
        assert f'version: "{VERSION}"' in text


def test_instructions_are_marker_bounded() -> None:
    text = (CP / ".github" / "copilot-instructions.md").read_text("utf-8")
    assert "<!-- SLM-START -->" in text and "<!-- SLM-END -->" in text


def test_mcp_matches_shared_config() -> None:
    plugin_mcp = json.loads((CP / ".vscode" / "mcp.json").read_text("utf-8"))
    shared = json.loads((REPO / "ide" / "configs" / "vscode-copilot-mcp.json").read_text("utf-8"))
    assert plugin_mcp == shared
    assert plugin_mcp["servers"]["superlocalmemory"]["command"] == "slm"


def test_hooks_are_copilot_schema() -> None:
    hooks = json.loads((CP / ".github" / "hooks" / "slm-hooks.json").read_text("utf-8"))
    assert hooks["version"] == 1
    assert "sessionStart" in hooks["hooks"] and "sessionEnd" in hooks["hooks"]
    # Copilot schema uses bash + timeoutSec (not Codex's command + timeout)
    first = hooks["hooks"]["sessionStart"][0]
    assert "bash" in first and "timeoutSec" in first


def test_version_stamped_consistently() -> None:
    recall = (CP / ".github" / "prompts" / "slm-recall.prompt.md").read_text("utf-8")
    assert f'version: "{VERSION}"' in recall
    assert VERSION in (CP / "README.md").read_text("utf-8")


def test_build_is_in_sync() -> None:
    """The committed copilot-plugin/ must match a fresh --check build (no drift)."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available")
    r = subprocess.run(
        [node, "scripts/build-copilot-plugin.mjs", "--check"],
        cwd=REPO, capture_output=True, text=True,
    )
    assert r.returncode == 0, f"copilot-plugin drift: {r.stderr or r.stdout}"
