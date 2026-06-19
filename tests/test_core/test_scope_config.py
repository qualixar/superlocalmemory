"""ScopeConfig — user-facing defaults for multi-scope memory. Shared memory is
OPT-IN: defaults reproduce 3.6.14 behaviour (personal writes, recall returns
ONLY this profile's own facts — no global/shared leak-in), and must round-trip
through config.json load/save.
"""
import json
import tempfile
from pathlib import Path

import pytest

from superlocalmemory.core.config import SLMConfig, ScopeConfig, Mode


def test_defaults_match_3614_behaviour():
    # Shared memory is opt-in: a fresh install is pure profile isolation,
    # exactly like 3.6.14 (writes personal, recall surfaces nobody else's data).
    sc = ScopeConfig()
    assert sc.default_scope == "personal"
    assert sc.recall_include_global is False
    assert sc.recall_include_shared is False


def test_for_mode_has_scope_defaults():
    c = SLMConfig.for_mode(Mode.A)
    assert c.scope.default_scope == "personal"
    assert c.scope.recall_include_global is False
    assert c.scope.recall_include_shared is False


def test_invalid_default_scope_rejected():
    with pytest.raises(ValueError):
        ScopeConfig(default_scope="bogus")
    for s in ("personal", "shared", "global"):
        assert ScopeConfig(default_scope=s).default_scope == s


def test_config_file_roundtrip():
    d = Path(tempfile.mkdtemp())
    (d / "config.json").write_text(json.dumps({
        "mode": "a",
        "scope": {"default_scope": "shared", "recall_include_global": False,
                  "recall_include_shared": True},
    }))
    c = SLMConfig.load(d / "config.json")
    assert c.scope.default_scope == "shared"
    assert c.scope.recall_include_global is False
    assert c.scope.recall_include_shared is True

    c.save(config_path=d / "out.json")
    back = json.loads((d / "out.json").read_text())
    assert back["scope"] == {
        "default_scope": "shared",
        "recall_include_global": False,
        "recall_include_shared": True,
    }


def test_absent_scope_section_uses_defaults():
    d = Path(tempfile.mkdtemp())
    (d / "config.json").write_text(json.dumps({"mode": "a"}))
    c = SLMConfig.load(d / "config.json")
    assert c.scope.default_scope == "personal"
    # No scope section → opt-in defaults (shared off), matching 3.6.14 isolation.
    assert c.scope.recall_include_global is False
    assert c.scope.recall_include_shared is False
