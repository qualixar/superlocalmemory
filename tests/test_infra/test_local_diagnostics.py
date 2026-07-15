"""Privacy and boundedness contract for local operational diagnostics."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from superlocalmemory.infra.local_diagnostics import LocalDiagnostics


class _Clock:
    def __init__(self, value: datetime) -> None:
        self.value = value

    def __call__(self) -> datetime:
        return self.value


def _flatten(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True)


def test_export_contains_only_bounded_aggregate_dimensions(tmp_path: Path) -> None:
    clock = _Clock(datetime(2026, 7, 15, 8, 0, tzinfo=UTC))
    diagnostics = LocalDiagnostics(tmp_path / "diagnostics.db", clock=clock)
    secret_query = "where is /Users/alice/acme/secret-plan.md"
    secret_fact_id = "fact-99-secret"
    secret_user = "alice@example.com"

    diagnostics.record(
        "recall",
        client=f"codex:{secret_user}",
        duration_ms=123.456,
        error=RuntimeError(f"failed for {secret_query} {secret_fact_id}"),
        source_clients=("claude:alice@example.com", "cursor:/Users/alice/acme"),
    )
    payload = diagnostics.export_payload()
    encoded = _flatten(payload)

    assert payload["schema_version"] == 1
    assert payload["privacy"]["local_only"] is True
    assert payload["privacy"]["reporting"] == "manual_export_only"
    assert payload["retention_days"] == 31
    assert secret_query not in encoded
    assert secret_fact_id not in encoded
    assert secret_user not in encoded
    assert "/Users/" not in encoded
    assert "RuntimeError" not in encoded
    assert "internal" in encoded
    assert "codex" in encoded
    assert "claude" in encoded


def test_export_is_byte_deterministic_and_private(tmp_path: Path) -> None:
    diagnostics = LocalDiagnostics(
        tmp_path / "diagnostics.db",
        clock=lambda: datetime(2026, 7, 15, 8, 0, tzinfo=UTC),
    )
    diagnostics.record("activation", client="cursor")
    diagnostics.record("remember", client="cli", duration_ms=4.2)

    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    diagnostics.export_json(first)
    diagnostics.export_json(second)

    assert first.read_bytes() == second.read_bytes()
    assert first.stat().st_mode & 0o077 == 0
    payload = json.loads(first.read_text(encoding="utf-8"))
    assert "generated_at" not in payload
    assert "path" not in _flatten(payload).lower()
    assert "user" not in _flatten(payload).lower()


def test_retention_and_dimensions_are_bounded(tmp_path: Path) -> None:
    clock = _Clock(datetime(2026, 6, 1, tzinfo=UTC))
    diagnostics = LocalDiagnostics(tmp_path / "diagnostics.db", clock=clock)

    for day in range(45):
        clock.value = datetime(2026, 6, 1, tzinfo=UTC) + timedelta(days=day)
        diagnostics.record(
            "recall",
            client=f"unknown-client-{day}",
            duration_ms=10**day,
            error=ValueError(f"raw-message-{day}"),
        )

    payload = diagnostics.export_payload()
    assert len(payload["days"]) == 31
    assert payload["days"][0]["day"] == "2026-06-15"
    encoded = _flatten(payload)
    assert "unknown-client" not in encoded
    assert "raw-message" not in encoded
    assert "other" in encoded
    assert "validation" in encoded


def test_unknown_operation_is_rejected_without_persistence(tmp_path: Path) -> None:
    diagnostics = LocalDiagnostics(tmp_path / "diagnostics.db")

    with pytest.raises(ValueError, match="unsupported diagnostic operation"):
        diagnostics.record("raw-query", client="codex")

    assert diagnostics.export_payload()["days"] == []


def test_diagnostics_module_has_no_network_reporting_surface() -> None:
    module_path = (
        Path(__file__).resolve().parents[2]
        / "src/superlocalmemory/infra/local_diagnostics.py"
    )
    source = module_path.read_text(encoding="utf-8")

    for forbidden in ("requests", "httpx", "urllib", "socket", "webhook"):
        assert forbidden not in source

