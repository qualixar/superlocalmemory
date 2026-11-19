# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Every kind of decision that can be undone has exactly one reversal (F-15 refactor guard)."""

from __future__ import annotations

from superlocalmemory.core import decision_undo, decisions


def test_every_undoable_kind_has_its_reversal() -> None:
    undoable = {kind for kind, actions in decisions.UNDOABLE.items() if actions}
    assert undoable <= set(decision_undo._REVERSALS), undoable - set(decision_undo._REVERSALS)


def test_no_reversal_for_a_kind_that_is_never_undone() -> None:
    assert decisions.SIGNING not in decision_undo._REVERSALS
