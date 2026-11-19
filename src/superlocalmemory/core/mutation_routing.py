# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Which mutations may target a profile other than the daemon's active one.

The canonical writer serves one active profile, but profiles are rows in one
store, and every mutation body already names its profile explicitly. What
keeps a mutation on the active profile is therefore a rule, not a mechanism,
and this module is that rule: a mutation reaches another profile only when its
kind is listed in ``ROUTABLE_MUTATIONS`` and the profile exists.

A kind is listed only together with the route that authorizes the routed
profile for it -- the caller's role on THAT profile, checked before the
profile's existence is revealed. ``replaces`` (POST /remember), correction
review (/api/corrections), deleting or correcting one memory (DELETE and PATCH
/api/memories/{id}) and setting a memory's kind (/api/memory-kinds) do. Nothing
else does, so a route that forwards a client's ``profile_id`` into a content
rewrite, archive, merge or scope change by mistake is still refused by the
writer.
"""

from __future__ import annotations

import enum
import sqlite3

from superlocalmemory.storage.write_coordinator import CommandKind

ROUTABLE_MUTATIONS: frozenset[CommandKind] = frozenset({
    CommandKind.REPLACE_BY_CALLER,
    CommandKind.APPLY_CORRECTION,
    CommandKind.REJECT_CORRECTION,
    CommandKind.ROLLBACK_CORRECTION,
    # 4.1.21: what a remote key bound to one profile needs to manage its own
    # memories while this computer uses another profile.
    CommandKind.DELETE_FACT,
    CommandKind.PROPOSE_CORRECTION,
    CommandKind.SET_FACT_KIND,
    # 4.2.30: the promotion a save declared as a rule or decision carries, for POST /remember
    # with a ``profile_id`` (the save and its promotion go to the same profile).
    CommandKind.PROMOTE_DECLARED,
})


class MutationTarget(enum.Enum):
    """Where a mutation command may go, as decided by ``classify_target``."""

    BOUND = "bound"                      # the daemon's active profile
    ROUTED = "routed"                    # a routable kind, to an existing profile
    NOT_ROUTABLE = "not_routable"        # any other kind, to another profile
    UNKNOWN_PROFILE = "unknown_profile"  # a routable kind, to no such profile


def classify_target(conn: sqlite3.Connection, kind: CommandKind, profile_id: str,
                    bound_profile: str) -> MutationTarget:
    """Decide whether ``kind`` may mutate ``profile_id``.

    ``conn`` must be the writer's own transaction connection: the existence
    check then sees exactly what the mutation will see, and no profile can be
    deleted between the two while the transaction holds the write lock.
    """
    if profile_id == bound_profile:
        return MutationTarget.BOUND
    if kind not in ROUTABLE_MUTATIONS:
        return MutationTarget.NOT_ROUTABLE
    row = conn.execute(
        "SELECT 1 FROM profiles WHERE profile_id = ?", (profile_id,)).fetchone()
    return MutationTarget.ROUTED if row is not None else MutationTarget.UNKNOWN_PROFILE


__all__ = ["ROUTABLE_MUTATIONS", "MutationTarget", "classify_target"]
