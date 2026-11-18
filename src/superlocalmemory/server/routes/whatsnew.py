# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""``GET /api/whatsnew/state``: which 4.2.30 features are already on (§8.10.3 l).

The dashboard's "New in 4.2.30" entries are shown only while the feature they describe is not
yet turned on, so the page needs each flag's state and nothing else. Read-only and open to the
dashboard like ``GET /api/outcomes/measurement``: it reveals only whether a switch is on, never
any memory. A switch that the kill switch holds off reads as off, and says the kill switch is on.

``features`` maps an entry id to ``{"on": bool, "flag": <flag id or null>}``. An entry with no
flag (``why``: the reason a memory was shown is always available) is never "on", so it shows
until the person dismisses the card. Measuring is "on" once its notice was read.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)
router = APIRouter(tags=["whatsnew"])

#: Entry id -> the flag whose ``on`` state hides it (``None``: no flag, never hidden by state).
ENTRIES: dict[str, str | None] = {
    "need_gate": "delivery.need_gate",
    "cues": "delivery.cues",
    "measurement": None,  # on by default: hidden once the notice was read (see below)
    "origin": "trust.taint",
    "firewall": "trust.taint",
    "secrets": "trust.secret_screen",
    "signing": "trust.signing",
    "why": None,
    "proposals": "compiler.dream",
    "wiki": "compiler.wiki",
    "import": "portability.import",
    "media_packs": "media.captions",
    "decisions": None,  # D-49: agents decide; always available, no switch
}


@router.get("/api/whatsnew/state")
def whatsnew_state():
    from superlocalmemory.core.feature_flags import FeatureFlags
    from superlocalmemory.infra.data_root import canonical_data_root
    from superlocalmemory.outcomes import disclosure

    try:
        home = canonical_data_root()
        flags = FeatureFlags(home)
        features = {}
        for entry, flag in ENTRIES.items():
            on = flags.enabled(flag) if flag else False
            features[entry] = {"on": on, "flag": flag}
        features["measurement"]["on"] = not disclosure.pending(home)
        return {"success": True, "kill_switch": flags.killed, "features": features}
    except Exception:
        logger.exception("whatsnew state failed")
        raise HTTPException(500, detail="Could not read which features are on.")
