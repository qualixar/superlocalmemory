# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""F-03 invariant: data-locality label is mode-record-sourced, never a UI literal.

Cloud / provider-assisted modes must never yield a local-only label.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from superlocalmemory.core.modes import (
    MODE_A,
    MODE_B,
    MODE_C,
    ModeCapabilities,
    get_capabilities,
)
from superlocalmemory.storage.models import Mode

_REPO = Path(__file__).resolve().parents[1]
_DASHBOARD_JS = _REPO / "src" / "superlocalmemory" / "ui" / "js" / "dashboard.js"


def test_every_mode_exposes_data_locality_label_from_record() -> None:
    for mode, record in (
        (Mode.A, MODE_A),
        (Mode.B, MODE_B),
        (Mode.C, MODE_C),
    ):
        caps = get_capabilities(mode)
        assert caps is record
        assert hasattr(caps, "data_locality_label")
        label = caps.data_locality_label
        assert isinstance(label, str) and label.strip()
        # Label must be derived from data_stays_local on the same record.
        if caps.data_stays_local:
            assert "local" in label.lower()
            assert "provider" not in label.lower()
        else:
            assert "local-only" not in label.lower()
            assert "local only" not in label.lower()


def test_cloud_mode_never_yields_local_only_label() -> None:
    label = get_capabilities(Mode.C).data_locality_label
    assert MODE_C.data_stays_local is False
    assert re.search(r"local[\s-]*only", label, re.IGNORECASE) is None
    assert "provider" in label.lower() or "cloud" in label.lower()


def test_local_modes_yield_local_only_label() -> None:
    for mode in (Mode.A, Mode.B):
        label = get_capabilities(mode).data_locality_label
        assert re.search(r"local[\s-]*only", label, re.IGNORECASE)


def test_dashboard_js_does_not_hardcode_local_only_literal() -> None:
    """UI must consume server field, not concatenate a fixed locality claim."""
    source = _DASHBOARD_JS.read_text(encoding="utf-8")
    # Forbidden: unconditional literal used for the subtitle locality segment.
    assert "· local-only ·" not in source
    # No JS string/template literal that invents locality without the API field.
    assert re.search(
        r"""['"]\s*·\s*local-only\s*·\s*['"]|local-only\s*·\s*v""",
        source,
    ) is None
    # Required: locality comes from the dashboard payload field.
    assert "data_locality_label" in source
    assert "data.data_locality_label" in source


def test_mode_capabilities_locality_label_matches_data_stays_local() -> None:
    """Frozen records cannot disagree with their derived label."""
    for caps in (MODE_A, MODE_B, MODE_C):
        assert isinstance(caps, ModeCapabilities)
        derived = "local-only" if caps.data_stays_local else "provider-assisted"
        assert caps.data_locality_label == derived


def _dashboard_client_for_mode(mode: Mode, tmp_path: Path):
    """Minimal ASGI harness for the real /api/v3/dashboard handler.

    Mounts the v3 router only (no unified_daemon monkeypatch path) so the
    middle link — the route itself serving data_locality_label — is tested.
    """
    from types import SimpleNamespace

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from superlocalmemory.server.routes.v3_api import router

    app = FastAPI()
    app.state.config = SimpleNamespace(
        mode=mode,
        active_profile="default",
        base_dir=tmp_path,
        llm=SimpleNamespace(provider="none", model=""),
    )
    app.include_router(router)
    return TestClient(app)


def test_dashboard_api_serves_data_locality_label_for_local_mode(tmp_path: Path) -> None:
    """Route-level: local modes must expose a local-only locality label."""
    client = _dashboard_client_for_mode(Mode.A, tmp_path)
    resp = client.get("/api/v3/dashboard")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "data_locality_label" in body
    label = body["data_locality_label"]
    assert isinstance(label, str) and label.strip()
    assert re.search(r"local[\s-]*only", label, re.IGNORECASE)


def test_dashboard_api_serves_non_local_only_label_for_cloud_mode(tmp_path: Path) -> None:
    """Route-level: provider-assisted Mode C must not claim local-only."""
    client = _dashboard_client_for_mode(Mode.C, tmp_path)
    resp = client.get("/api/v3/dashboard")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "data_locality_label" in body
    label = body["data_locality_label"]
    assert isinstance(label, str) and label.strip()
    assert re.search(r"local[\s-]*only", label, re.IGNORECASE) is None
    assert body.get("data_stays_local") is False
