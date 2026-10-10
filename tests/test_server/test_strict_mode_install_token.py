# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""With SLM_REQUIRE_CREDENTIALS=1 the install token is not handed out to the page.

The dashboard route /internal/token gives the install token to any process on
this computer. That is fine in the normal posture, but the strict setting says
"even a local process must present a key". Handing the key to anyone who asks
would make the setting meaningless, so the route refuses and the dashboard asks
the person to paste the key (`slm token show`).
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

LOOPBACK = ("127.0.0.1", 50123)


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
    from superlocalmemory.core.security_primitives import ensure_install_token
    from superlocalmemory.server.routes.token import router

    ensure_install_token()
    app = FastAPI()
    app.include_router(router)
    return TestClient(app, base_url="http://127.0.0.1:8765", client=LOOPBACK)


def test_normal_mode_still_serves_the_token(client, monkeypatch):
    monkeypatch.delenv("SLM_REQUIRE_CREDENTIALS", raising=False)

    r = client.get("/internal/token")

    assert r.status_code == 200
    assert r.json()["token"]


def test_strict_mode_refuses_with_a_plain_reason(client, monkeypatch):
    monkeypatch.setenv("SLM_REQUIRE_CREDENTIALS", "1")

    r = client.get("/internal/token")

    assert r.status_code == 403
    body = r.json()
    assert "token" not in body
    assert body["error"] == "key_required"
    assert "slm token show" in body["message"]


def test_strict_mode_off_values_do_not_refuse(client, monkeypatch):
    monkeypatch.setenv("SLM_REQUIRE_CREDENTIALS", "0")

    assert client.get("/internal/token").status_code == 200
