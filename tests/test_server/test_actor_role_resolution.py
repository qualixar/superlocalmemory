"""Server-derived ActorContext role resolution.

The daemon must map the authenticated principal to real ActorContext roles so
the operation policy layer enforces roles rather than treating every caller as
owner. Identity is derived from server state only, never the request body.
"""

from __future__ import annotations

from types import SimpleNamespace

from superlocalmemory.access.rbac import Role
from superlocalmemory.core.actor_context import ActorRole
from superlocalmemory.server.rbac_enforce import resolve_actor_roles


class _FakeRbac:
    def __init__(self, *, session_user=None, role=None):
        self._user = session_user
        self._role = role

    def resolve_session(self, token):
        return self._user

    def get_role(self, user_id, profile):
        return self._role


def _request(*, rbac=None, session_token=""):
    headers = {"X-SLM-User-Session": session_token} if session_token else {}
    return SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(rbac=rbac)),
        headers=headers,
        cookies={},
    )


def test_owner_without_session_maps_to_owner_role():
    req = _request(rbac=_FakeRbac(), session_token="")
    assert resolve_actor_roles(req, profile="default") == frozenset({ActorRole.OWNER})


def test_no_rbac_engine_maps_to_owner_role():
    req = _request(rbac=None, session_token="tok")
    assert resolve_actor_roles(req, profile="default") == frozenset({ActorRole.OWNER})


def test_admin_user_maps_to_admin_role():
    rbac = _FakeRbac(session_user={"user_id": "u1", "username": "a"}, role=Role.ADMIN)
    req = _request(rbac=rbac, session_token="tok")
    assert resolve_actor_roles(req, profile="default") == frozenset({ActorRole.ADMIN})


def test_member_user_maps_to_member_role():
    rbac = _FakeRbac(session_user={"user_id": "u2", "username": "m"}, role=Role.MEMBER)
    req = _request(rbac=rbac, session_token="tok")
    assert resolve_actor_roles(req, profile="default") == frozenset({ActorRole.MEMBER})


def test_viewer_user_maps_to_viewer_role():
    rbac = _FakeRbac(session_user={"user_id": "u3", "username": "v"}, role=Role.VIEWER)
    req = _request(rbac=rbac, session_token="tok")
    assert resolve_actor_roles(req, profile="default") == frozenset({ActorRole.VIEWER})


def test_user_without_role_is_anonymous():
    rbac = _FakeRbac(session_user={"user_id": "u4", "username": "x"}, role=None)
    req = _request(rbac=rbac, session_token="tok")
    assert resolve_actor_roles(req, profile="default") == frozenset({ActorRole.ANONYMOUS})
