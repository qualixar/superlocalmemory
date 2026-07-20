"""Codex hook installer tests.

The installer is deliberately isolated from a real home directory: hook setup
must be additive, idempotent, and reversible without touching user-owned hook
definitions.
"""

from __future__ import annotations

import json

from superlocalmemory.hooks import codex_hooks


def _read(path):
    return json.loads(path.read_text())


def test_install_merges_only_slm_owned_entries_and_creates_backup(tmp_path):
    hooks_path = tmp_path / ".codex" / "hooks.json"
    hooks_path.parent.mkdir()
    original = {
        "hooks": {
            "SessionStart": [{"hooks": [{"type": "command", "command": "my-start"}]}],
            "Stop": [{"hooks": [{"type": "command", "command": "my-stop"}]}],
        },
        "custom": {"keep": True},
    }
    hooks_path.write_text(json.dumps(original, indent=2) + "\n")

    result = codex_hooks.install_hooks(hooks_path=hooks_path)

    assert result["success"] is True
    assert _read(hooks_path.with_suffix(".json.slm.bak")) == original
    data = _read(hooks_path)
    assert data["custom"] == {"keep": True}
    assert any("my-start" in str(entry) for entry in data["hooks"]["SessionStart"])
    assert any(codex_hooks.is_slm_hook_entry(entry) for entry in data["hooks"]["SessionStart"])


def test_install_is_idempotent(tmp_path):
    hooks_path = tmp_path / ".codex" / "hooks.json"

    codex_hooks.install_hooks(hooks_path=hooks_path)
    codex_hooks.install_hooks(hooks_path=hooks_path)

    data = _read(hooks_path)
    for entries in data["hooks"].values():
        assert sum(codex_hooks.is_slm_hook_entry(entry) for entry in entries) == 1


def test_malformed_json_fails_without_overwrite(tmp_path):
    hooks_path = tmp_path / ".codex" / "hooks.json"
    hooks_path.parent.mkdir()
    hooks_path.write_text("{ definitely not json")

    result = codex_hooks.install_hooks(hooks_path=hooks_path)

    assert result["success"] is False
    assert hooks_path.read_text() == "{ definitely not json"
    assert not hooks_path.with_suffix(".json.slm.bak").exists()


def test_remove_only_removes_owned_entries(tmp_path):
    hooks_path = tmp_path / ".codex" / "hooks.json"
    hooks_path.parent.mkdir()
    hooks_path.write_text(json.dumps({"hooks": {
        "SessionStart": [{"hooks": [{"type": "command", "command": "my-start"}]}],
        "Stop": [{"hooks": [{"type": "command", "command": "my-stop"}]}],
    }}))
    codex_hooks.install_hooks(hooks_path=hooks_path)

    result = codex_hooks.remove_hooks(hooks_path=hooks_path)

    assert result["success"] is True
    data = _read(hooks_path)
    assert data["hooks"]["SessionStart"] == [{"hooks": [{"type": "command", "command": "my-start"}]}]
    assert data["hooks"]["Stop"] == [{"hooks": [{"type": "command", "command": "my-stop"}]}]


def test_install_retires_legacy_slm_commands_without_removing_mixed_user_group(tmp_path):
    hooks_path = tmp_path / ".codex" / "hooks.json"
    hooks_path.parent.mkdir()
    hooks_path.write_text(json.dumps({"hooks": {"SessionStart": [{"hooks": [
        {"type": "command", "command": "python3 ~/.codex/hooks/auto-recall.py"},
        {"type": "command", "command": "python3 ~/.agents/hooks/universal-hook.py --intent slm_recall --agent codex"},
        {"type": "command", "command": "my-unrelated-start"},
    ]}]}}))

    result = codex_hooks.install_hooks(hooks_path=hooks_path)

    assert result["success"] is True
    entries = _read(hooks_path)["hooks"]["SessionStart"]
    commands = [hook["command"] for entry in entries for hook in entry["hooks"]]
    assert "my-unrelated-start" in commands
    assert not any("auto-recall.py" in command or "--intent slm_" in command for command in commands)
    assert any(codex_hooks.SLM_MARKER in command for command in commands)


def test_definitions_are_portable_and_cover_supported_lifecycle():
    definitions = codex_hooks.hook_definitions()

    assert set(definitions) == {"SessionStart", "PostToolUse", "UserPromptSubmit", "Stop"}
    commands = [hook["command"] for groups in definitions.values() for group in groups for hook in group["hooks"]]
    assert all(command.startswith("slm hook ") for command in commands)
    assert not any("/Users/" in command or "~/." in command for command in commands)


def test_status_reports_corrupt_config_as_indeterminate(tmp_path):
    hooks_path = tmp_path / ".codex" / "hooks.json"
    hooks_path.parent.mkdir()
    hooks_path.write_text("{")

    status = codex_hooks.check_status(hooks_path=hooks_path)

    assert status["installed"] is None
    assert "parse error" in status["error"].lower()
