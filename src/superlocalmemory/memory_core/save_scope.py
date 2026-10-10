# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Who a new memory is visible to: the one rule typed text, pictures, pages and folders share."""

from __future__ import annotations

from typing import Any

SCOPES = frozenset({"personal", "project", "shared", "global"})
#: Scopes that put a memory in front of other profiles; saving one takes the SHARE permission.
BROAD_SCOPES = frozenset({"shared", "global"})


def default_scope(config: Any) -> str:
    """The configured default scope; personal when none is set."""
    return str(getattr(getattr(config, "scope", None), "default_scope", "") or "personal")


def resolve_scope(config: Any, scope: str | None) -> str:
    """``scope`` when given, else the configured default. ValueError for a scope that does not exist."""
    chosen = (scope or "").strip() or default_scope(config)
    if chosen not in SCOPES:
        raise ValueError(f"unsupported scope: {chosen}")
    return chosen
