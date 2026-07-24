# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later

"""F10 regression: _compute_ranker_phase must bypass the model cache on the
first call per daemon start (tamper detection) and use the cache thereafter.

Scenario reproduced: 3.8.0 called load_active(..., use_cache=False) for every
dashboard load.  3.8.1 switched to use_cache=True permanently.  A model swapped
on disk (offline tampering) is no longer detected until next model promotion.

Fix: track which profiles have had the initial cache-bypass check in a
module-level set.  First call per daemon start per profile → use_cache=False.
Subsequent calls → use_cache=True.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _clear_tamper_state(profile: str) -> None:
    """Clear the per-profile tamper-check state introduced by the F10 fix."""
    import superlocalmemory.server.routes.learning as m
    if hasattr(m, "_ranker_initial_check_done"):
        m._ranker_initial_check_done.discard(profile)


def test_f10_first_call_bypasses_cache_for_tamper_detection() -> None:
    """First _compute_ranker_phase call per profile must use use_cache=False."""
    import superlocalmemory.server.routes.learning as learning_mod

    profile = "f10-tamper-test-profile"
    _clear_tamper_state(profile)

    use_cache_values: list[bool | None] = []

    def track_load_active(db, prof_id, *, use_cache):
        use_cache_values.append(use_cache)
        return None  # no active model

    mock_db = MagicMock()
    mock_db.exists.return_value = True

    mock_store = MagicMock()
    mock_store.count_signals.return_value = 0

    with patch("superlocalmemory.learning.model_cache.load_active", side_effect=track_load_active), \
         patch.object(learning_mod, "LEARNING_DB", mock_db), \
         patch("superlocalmemory.server.routes.learning.ReadOnlyRankerStore",
               return_value=mock_store):
        learning_mod._compute_ranker_phase(profile)

    assert use_cache_values, "load_active must have been called"
    assert use_cache_values[0] is False, (
        f"F10: first call must use use_cache=False for tamper detection "
        f"(got use_cache={use_cache_values[0]!r}). "
        "Initial tamper-check bypass not implemented."
    )


def test_f10_second_call_uses_cache() -> None:
    """Second and subsequent _compute_ranker_phase calls must use use_cache=True."""
    import superlocalmemory.server.routes.learning as learning_mod

    profile = "f10-cache-reuse-profile"
    _clear_tamper_state(profile)

    use_cache_values: list[bool | None] = []

    def track_load_active(db, prof_id, *, use_cache):
        use_cache_values.append(use_cache)
        return None

    mock_db = MagicMock()
    mock_db.exists.return_value = True

    mock_store = MagicMock()
    mock_store.count_signals.return_value = 0

    with patch("superlocalmemory.learning.model_cache.load_active", side_effect=track_load_active), \
         patch.object(learning_mod, "LEARNING_DB", mock_db), \
         patch("superlocalmemory.server.routes.learning.ReadOnlyRankerStore",
               return_value=mock_store):
        learning_mod._compute_ranker_phase(profile)  # first: use_cache=False
        learning_mod._compute_ranker_phase(profile)  # second: use_cache=True

    assert len(use_cache_values) >= 2, "load_active must be called at least twice"
    assert use_cache_values[1] is True, (
        f"F10: second call must use use_cache=True (got {use_cache_values[1]!r})"
    )


def test_f10_different_profiles_each_get_initial_check() -> None:
    """Each distinct profile must get its own initial cache-bypass check."""
    import superlocalmemory.server.routes.learning as learning_mod

    profile_a = "f10-profile-alpha"
    profile_b = "f10-profile-beta"
    _clear_tamper_state(profile_a)
    _clear_tamper_state(profile_b)

    use_cache_by_profile: dict[str, list[bool | None]] = {}

    def track_load_active(db, prof_id, *, use_cache):
        use_cache_by_profile.setdefault(prof_id, []).append(use_cache)
        return None

    mock_db = MagicMock()
    mock_db.exists.return_value = True

    mock_store = MagicMock()
    mock_store.count_signals.return_value = 0

    with patch("superlocalmemory.learning.model_cache.load_active", side_effect=track_load_active), \
         patch.object(learning_mod, "LEARNING_DB", mock_db), \
         patch("superlocalmemory.server.routes.learning.ReadOnlyRankerStore",
               return_value=mock_store):
        learning_mod._compute_ranker_phase(profile_a)
        learning_mod._compute_ranker_phase(profile_b)

    assert use_cache_by_profile.get(profile_a, [None])[0] is False, (
        f"profile_a first call must use use_cache=False "
        f"(got {use_cache_by_profile.get(profile_a)!r})"
    )
    assert use_cache_by_profile.get(profile_b, [None])[0] is False, (
        f"profile_b first call must use use_cache=False "
        f"(got {use_cache_by_profile.get(profile_b)!r})"
    )
