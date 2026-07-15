# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later

"""Shared HTTP write-identity boundary.

Caller-selected IDE or agent labels are audit metadata.  Authorization is
derived only from the private capability for the exact daemon instance or the
local install token used by the same-origin dashboard.
"""

from __future__ import annotations

import hashlib
import hmac
from typing import Any

from fastapi import HTTPException, Request


def _header(request: Request, name: str) -> str:
    headers = request.headers
    value = headers.get(name, "")
    if value:
        return str(value)
    lowered = name.lower()
    for key, candidate in headers.items():
        if str(key).lower() == lowered:
            return str(candidate)
    return ""


def require_daemon_actor(request: Request, descriptor: Any | None) -> str:
    """Require the private capability for one exact daemon instance."""
    capability = _header(request, "X-SLM-Daemon-Capability")
    if descriptor is None or not hmac.compare_digest(
        capability,
        str(descriptor.capability),
    ):
        raise HTTPException(403, detail="Invalid daemon capability")
    target_instance = _header(request, "X-SLM-Target-Instance")
    if not hmac.compare_digest(
        target_instance,
        str(descriptor.instance_id),
    ):
        raise HTTPException(409, detail="Daemon instance changed")
    return f"daemon-capability:{descriptor.capability_fingerprint}"


def require_write_actor(
    request: Request,
    descriptor: Any | None,
    *,
    actor_kind: str = "dashboard",
) -> str:
    """Return a capability-derived actor or reject the write."""
    if _header(request, "X-SLM-Daemon-Capability"):
        return require_daemon_actor(request, descriptor)

    from superlocalmemory.core.security_primitives import verify_install_token

    if verify_install_token(_header(request, "X-Install-Token")):
        from superlocalmemory.core.engine_ingestion import local_trusted_actor_id

        return local_trusted_actor_id(actor_kind)

    from superlocalmemory.infra.auth_middleware import verify_api_key

    api_key = _header(request, "X-SLM-API-Key")
    if verify_api_key(api_key):
        fingerprint = hashlib.sha256(
            b"superlocalmemory-api-actor-v1\0" + api_key.encode("utf-8")
        ).hexdigest()
        return f"api-key:{actor_kind}:{fingerprint}"

    raise HTTPException(403, detail="Authenticated write capability required")


__all__ = ["require_daemon_actor", "require_write_actor"]
