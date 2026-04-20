# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.22 — LLD-04 §6.4

"""Tests for deep-iteration secret redaction on preferences (LLD-04 §4.1)."""

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi", reason="fastapi not installed")

from superlocalmemory.server.routes.brain import redact_secrets_in_preferences


# ---------------------------------------------------------------------------
# Known-pattern redaction
# ---------------------------------------------------------------------------


def test_openai_key_redacted() -> None:
    payload = {
        "topics": [{"name": "sk-" + "A" * 48, "strength": 0.5}],
    }
    out = redact_secrets_in_preferences(payload)
    assert out["redacted_count"] >= 1
    import json
    assert "sk-" + "A" * 48 not in json.dumps(out)
    assert "[REDACTED:OPENAI" in json.dumps(out)


def test_aws_key_redacted() -> None:
    payload = {"topics": [{"name": "AKIAIOSFODNN7EXAMPLE"}]}
    out = redact_secrets_in_preferences(payload)
    assert out["redacted_count"] >= 1
    import json
    dumped = json.dumps(out)
    assert "AKIAIOSFODNN7EXAMPLE" not in dumped
    assert "[REDACTED:AWS" in dumped


def test_jwt_redacted() -> None:
    jwt = ("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
           "eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ."
           "SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c")
    payload = {"entities": [{"name": jwt}]}
    out = redact_secrets_in_preferences(payload)
    assert out["redacted_count"] >= 1
    import json
    assert jwt not in json.dumps(out)


def test_no_false_positive_on_public_data() -> None:
    payload = {
        "topics": [{"name": "ai_agents", "strength": 0.87}],
        "entities": [{"name": "Qualixar", "mention_count": 142}],
        "tech": [{"name": "Python", "frequency": 0.62}],
        "source": "_store_patterns",
    }
    out = redact_secrets_in_preferences(payload)
    assert out["redacted_count"] == 0
    assert out["topics"][0]["name"] == "ai_agents"
    assert out["entities"][0]["name"] == "Qualixar"
    assert out["tech"][0]["name"] == "Python"


def test_redacts_inside_nested_lists_and_dicts() -> None:
    payload = {
        "topics": [
            {"name": "safe", "nested": {"token": "AKIAIOSFODNN7EXAMPLE"}},
        ],
        "entities": ["AKIAABCDEFGHIJKLMNOP", "Qualixar"],
    }
    out = redact_secrets_in_preferences(payload)
    assert out["redacted_count"] >= 2
    import json
    dumped = json.dumps(out)
    assert "AKIAIOSFODNN7EXAMPLE" not in dumped
    assert "AKIAABCDEFGHIJKLMNOP" not in dumped
    assert "Qualixar" in dumped  # public data preserved


def test_returns_new_dict_not_mutated_input() -> None:
    payload = {"topics": [{"name": "AKIAIOSFODNN7EXAMPLE"}]}
    original_name = payload["topics"][0]["name"]
    out = redact_secrets_in_preferences(payload)
    # Input must not be mutated (immutable pattern).
    assert payload["topics"][0]["name"] == original_name
    assert out is not payload
