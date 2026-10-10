"""Display names of connected web apps, from the gateway's connected-apps list.

Only names for the owner's own screens; never an identity. Held in memory and
refreshed by the runtime.
"""

from __future__ import annotations

import threading

_lock = threading.Lock()
_names: dict[str, dict[str, str]] = {}


def set_names(connection_id: str, names: dict[str, str]) -> None:
    """Replace the names known for one connection."""
    with _lock:
        if names:
            _names[connection_id] = dict(names)
        else:
            _names.pop(connection_id, None)


def display_name(connection_id: str, authorization_id: str, peer_ref: str) -> str:
    with _lock:
        known = _names.get(connection_id, {}).get(authorization_id)
    return known or "Web app " + peer_ref[2:8]
