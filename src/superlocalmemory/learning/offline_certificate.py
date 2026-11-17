# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Whether a ranker candidate holds an offline certificate (D-45).

A candidate may enter the live sequential test (served to half the recalls)
only once the offline gate has certified it: its improvement over the order
recall actually serves, measured on logged deliveries, has an anytime-valid
lower bound above 0 (W3-6d writes ``offline_certificate`` into the candidate's
``shadow_results_json``). Until then nobody is served an unproven model, and an
uncertified candidate is not "under test": the next retrain replaces it.
"""

from __future__ import annotations

import json
from typing import Any

KEY = "offline_certificate"


def certified(shadow_results_json: Any) -> bool:
    """True only for an explicit, recorded certificate."""
    try:
        value = json.loads(shadow_results_json) if isinstance(shadow_results_json, str) \
            else shadow_results_json
    except (TypeError, ValueError):
        return False
    return isinstance(value, dict) and value.get(KEY) is True


__all__ = ["KEY", "certified"]
