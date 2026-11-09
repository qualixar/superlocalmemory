# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""`GET /mesh/state/delta` reaches the delta endpoint, not `/mesh/state/{key}`.

The general mesh router was mounted first, so its `/mesh/state/{key}` matched
`delta` as a key and answered 404 "key not found": fleet state sync
(mesh/remote_sync.py pulls `/mesh/state/delta`) never received a row. Both
routes must resolve, and WP9's read authorization must still apply to both
(the denial cases live in test_mesh_read_permission.py and run unchanged).
"""

# ruff: noqa: F811 (pytest fixtures are imported and redeclared as parameters)

from __future__ import annotations

from tests.test_security.test_mesh_read_permission import (  # noqa: F401  (fixture)
    _FLEET,
    _HOST,
    _SECRET_STATE,
    _company,
    _fleet_node,
    _install_token,
    mesh,
)


def test_the_delta_route_is_the_delta_endpoint(mesh) -> None:
    client, _app, _rbac = mesh
    resp = client.get("/mesh/state/delta", headers={**_HOST, **_install_token()})
    assert resp.status_code == 200, resp.text[:200]
    body = resp.json()
    assert set(body) == {"entries", "node_id"}, body
    assert [e["key"] for e in body["entries"]] == ["plan"]
    assert body["entries"][0]["value"] == _SECRET_STATE


def test_a_single_key_still_resolves(mesh) -> None:
    client, _app, _rbac = mesh
    resp = client.get("/mesh/state/plan", headers={**_HOST, **_install_token()})
    assert resp.status_code == 200, resp.text[:200]
    assert resp.json()["value"] == _SECRET_STATE
    missing = client.get("/mesh/state/nope", headers={**_HOST, **_install_token()})
    assert missing.status_code == 404


def test_a_fleet_node_syncs_state_through_the_delta_route(mesh) -> None:
    _client_, app, rbac = mesh
    _company(rbac)
    node = _fleet_node(app)
    resp = node.get("/mesh/state/delta", params={"since": 0}, headers={"X-Mesh-Secret": _FLEET})
    assert resp.status_code == 200, resp.text[:200]
    assert [e["value"] for e in resp.json()["entries"]] == [_SECRET_STATE]

