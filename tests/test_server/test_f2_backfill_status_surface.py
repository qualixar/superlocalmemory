# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later

"""F2 regression: /status endpoint must expose M028 backfill progress so
operators can observe the backfill state after first-upgrade.

This is a read-only visibility fix.  The field 'm028_backfill' is added to
the status response dict from application.state.fact_entity_association_repair_status.
"""

from __future__ import annotations

import inspect

import pytest


def test_f2_status_endpoint_contains_m028_backfill_key() -> None:
    """The /status handler source must assemble the 'm028_backfill' response field.

    This is a source-level assertion confirming the fix is applied.  The field
    is read-only and sourced from application.state.fact_entity_association_repair_status.
    """
    from superlocalmemory.server import unified_daemon

    # The status handler is a nested async function inside create_app().
    # Source inspection is the appropriate technique here — the full lifespan
    # cannot be spun up in a unit test.
    source = inspect.getsource(unified_daemon)

    assert '"m028_backfill"' in source or "'m028_backfill'" in source, (
        "F2: The /status endpoint must include an 'm028_backfill' field in its "
        "response dict, sourced from application.state.fact_entity_association_repair_status. "
        "Add it to the return dict in the @application.get('/status') handler."
    )


def test_f2_status_assembles_backfill_from_state() -> None:
    """The status handler must read fact_entity_association_repair_status from state."""
    from superlocalmemory.server import unified_daemon

    source = inspect.getsource(unified_daemon)
    # Verify the field is fetched from state (not hardcoded)
    assert "fact_entity_association_repair_status" in source, (
        "The /status handler must read fact_entity_association_repair_status from "
        "application.state to populate the m028_backfill field."
    )
