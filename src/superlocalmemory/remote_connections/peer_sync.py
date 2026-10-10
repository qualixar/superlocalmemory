# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V4 | https://qualixar.com | https://varunpratap.com

"""Keep the mesh in step with the apps the gateway still lists for a connection.

The gateway's connected-apps list is the truth for which web apps may still
call. Each time the runtime reads it, the names shown for those apps are
refreshed, and, when the whole list could be read, the mesh peer of every app
no longer on it is retired (its queued messages are dropped). Blocking: callers
run :func:`apply` in a worker thread.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

from superlocalmemory.remote_connections import peer_names
from superlocalmemory.remote_connections.grant import peer_ref

logger = logging.getLogger(__name__)

#: How often a running connection re-reads the list.
INTERVAL_S = 600.0


def started_at() -> str:
    """The moment a list is requested; only peers older than this may be retired
    by it, so an app first used while the answer was in flight is kept."""
    return datetime.now(timezone.utc).isoformat()


def is_complete(apps: object, valid: Sequence[Mapping[str, Any]]) -> bool:
    """Whether ``valid`` is the whole list: an answer with a row we could not
    read, or no list at all, proves nothing about who was removed."""
    return isinstance(apps, list) and len(valid) == len(apps)


def _end_links_of_missing_apps(connection_id: str, listed: set[str]) -> None:
    """An app revoked anywhere (not only on this computer) is gone from the list: its open upload links end."""
    try:
        from superlocalmemory.media.upload_links import default_links

        ended = default_links().fail_unlisted_authorizations(connection_id, listed)
    except Exception as exc:  # noqa: BLE001 - the gateway checks the app at every step anyway
        logger.warning("upload links of unlisted apps were not closed (%s)", type(exc).__name__)
        return
    if ended:
        logger.info("ended %d upload link(s) of revoked web app(s)", ended)


def apply(connection_id: str, valid: Sequence[Mapping[str, Any]], *, complete: bool,
          broker: Any, started: str) -> None:
    """Refresh the names for ``connection_id`` and, if ``complete``, retire the
    peers of apps that are no longer listed."""
    peer_names.set_names(connection_id, {
        app["authorization_id"]: app["name"] for app in valid})
    if complete:
        _end_links_of_missing_apps(connection_id, {app["authorization_id"] for app in valid})
    if not complete or broker is None:
        return
    refs = {peer_ref(connection_id, app["authorization_id"]) for app in valid}
    retired = broker.retire_missing_web_peers(connection_id, refs, registered_before=started)
    if retired:
        logger.info("retired %d revoked web app peer(s)", len(retired))
