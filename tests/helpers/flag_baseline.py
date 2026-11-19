"""The "every 4.2.30 flag off" baseline, for tests that exercise one flag at a time.

Every flag is on by default (Varun, 19 Nov 2026). A test written when flags shipped dark builds a
``FeatureFlags(file_values={...}, environ={})`` holding only the flags it cares about and relies on the rest
being off. Those tests say so explicitly with ``off_except`` instead of depending on a default.
"""

from __future__ import annotations

from collections.abc import Mapping

from superlocalmemory.core.feature_flags import FLAGS

#: Every registered flag, explicitly "off".
ALL_OFF: Mapping[str, str] = {flag_id: "off" for flag_id in FLAGS}


def off_except(states: Mapping[str, object] | None = None, **more: object) -> dict[str, object]:
    """``ALL_OFF`` with the given flags set (a mapping, or ``flag_id=value`` keyword pairs)."""
    return {**ALL_OFF, **dict(states or {}), **more}


def env_baseline_off(monkeypatch, *, keep: tuple[str, ...] = ("outcomes.",)) -> None:
    """Set ``SLM_FEATURE_<ID>=off`` for every flag except those whose id starts with ``keep``.

    For test folders written when 4.2.30 shipped dark: each test turns on (or to shadow) the one flag it is
    about with ``monkeypatch.setenv``, which replaces this baseline. The measuring flags (``outcomes.``) were
    on from the start and stay as they are.
    """
    from superlocalmemory.core.feature_flags import env_name

    for flag_id in FLAGS:
        if not flag_id.startswith(keep):
            monkeypatch.setenv(env_name(flag_id), "off")
