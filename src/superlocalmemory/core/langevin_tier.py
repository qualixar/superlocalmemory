# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com
"""Where a memory sits in the Langevin ball, and so which tier it is in.

The radius is ``1 - R(t)``, the memory's Ebbinghaus retention on the store's
own timescale. New memories are seeded there (GitHub #136) and every
maintenance pass keeps them there (``_langevin_pass``), so the tier follows how
a memory is used and nothing else. Split out of ``core/maintenance.py``, which
re-exports every name for existing importers.
"""

from __future__ import annotations

import hashlib
import math as _math
from datetime import UTC, datetime

import numpy as np

_LANGEVIN_DIM = 8
_MAX_NORM = 0.99


def _age_days(created_at: str | None) -> float:
    """Age in days from an ISO timestamp.

    Naive timestamps (no offset, no Z) are assumed UTC — some store paths
    persist created_at without timezone info, and subtracting a naive
    datetime from datetime.now(UTC) raises TypeError, which previously
    aborted the whole backfill loop.
    """
    if not created_at:
        return 0.0
    try:
        created = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return 0.0
    if created.tzinfo is None:
        created = created.replace(tzinfo=UTC)
    return max(0.0, (datetime.now(UTC) - created).total_seconds() / 86400.0)


def _compute_equilibrium_radius(
    access_count: int,
    age_days: float,
    importance: float,
    temperature: float = 0.3,
    dim: int = 8,
) -> float:
    """SUPERSEDED by ``_retention_radius``. No production caller remains.

    Kept because its behaviour is what several tests characterise, and because
    deleting the thing a bug report names makes the report unreadable later.

    Do not reach for this as the seed authority. ``r_eq = sqrt(T*dim / 2*a)``
    scales as sqrt(dim) while the band boundaries in ``math/langevin.py`` are
    dimension-independent constants, so at T=0.3, dim=8 its entire reachable
    range over every possible input is [0.5210, 0.6330]: ACTIVE needs
    alpha_eff > 13.33 against a maximum of ~4.05, and a memory accessed
    100,000 times at maximum importance still lands in WARM. GitHub #136.

    r_eq ≈ sqrt(T * dim / (2 * effective_alpha))
    """
    alpha, beta, gamma, delta = 3.0, 0.8, 0.005, 0.5
    effective_alpha = (
        alpha
        + beta * _math.log(access_count + 1) / 10.0
        - gamma * min(age_days, 365.0) / 365.0
        + delta * importance
    )
    effective_alpha = max(0.1, effective_alpha)
    r_eq = _math.sqrt(temperature * dim / (2.0 * effective_alpha))
    return min(r_eq, _MAX_NORM * 0.95)


def _direction_seed(fact_id: str) -> int:
    """A stable per-fact seed. ``hash()`` is salted per process and unusable."""
    digest = hashlib.blake2b(fact_id.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big")


def _curve(curve: object | None) -> object:
    """The person's curve when the caller has it; the decay pass's defaults else.

    Callers on the maintenance path pass ``EbbinghausCurve(config.forgetting)``
    -- the same settings the decay pass reads -- so both passes decide "may
    this memory archive" from one function and one set of settings.
    """
    if curve is not None:
        return curve
    from superlocalmemory.core.config import ForgettingConfig
    from superlocalmemory.math.ebbinghaus import EbbinghausCurve

    return EbbinghausCurve(ForgettingConfig())


def _retention_and_zone(
    access_count: int, age_days: float, importance: float, curve: object | None = None,
) -> tuple[float, str]:
    """``(R, zone)`` from ``EbbinghausCurve.retention_and_zone``, the function the
    decay pass and the ELC step use."""
    retention, _strength, zone = _curve(curve).retention_and_zone(  # type: ignore[attr-defined]
        access_count=access_count, importance=importance, confirmation_count=0,
        emotional_salience=0.0, hours_since=max(0.0, age_days) * 24.0,
    )
    return retention, zone


def _retention_radius(
    access_count: int, age_days: float, importance: float, curve: object | None = None,
) -> float:
    """``1 - R(t)`` on the store's own timescale, with the person's settings.

    ``EbbinghausCurve`` is parameterised in HOURS; applied verbatim it puts a
    one-day-old memory at r=0.94, archived. ``store_scaled_strength`` takes the
    time constant from the ladder in ``core/tier_manager.py`` (GitHub #136).

    Until 4.2.8 this built ``EbbinghausCurve(ForgettingConfig())`` -- the
    defaults -- whatever the person had set, while the decay pass used their
    settings. On one real store that archived 630 memories the person's own
    settings keep. ``curve`` is now the caller's, from the same settings.
    """
    from superlocalmemory.math.langevin import _MAX_NORM

    retention, _zone = _retention_and_zone(access_count, age_days, importance, curve)
    return min(max(1.0 - retention, 0.0), _MAX_NORM * 0.95)


def langevin_lifecycle(
    access_count: int, age_days: float, importance: float, curve: object | None = None,
) -> str:
    """The tier the Langevin pass writes: the shared zone, in mirror spelling.

    The radius still sets the retrieval weight; the tier (and so whether a
    memory may archive) is the decay zone, so the two passes cannot disagree.
    """
    from superlocalmemory.core.lifecycle_state import atomic_lifecycle_for

    _retention, zone = _retention_and_zone(access_count, age_days, importance, curve)
    return atomic_lifecycle_for(zone)


def _seed_langevin_position(
    access_count: int,
    age_days: float,
    importance: float,
    temperature: float = 0.3,
    dim: int = 8,
    *,
    fact_id: str = "",
    curve: object | None = None,
) -> list[float]:
    """Place a fact at the radius its Ebbinghaus retention implies.

    Radius is ``1 - R(t)``. A memory written a moment ago has R = 1 and sits at
    the centre; one left alone decays outward; using it raises S and pulls it
    back in. That is the README's claim, and until 4.1.15 the code did not
    implement it: the old equilibrium radius ``sqrt(T*dim / 2*alpha_eff)``
    could only ever return a value in [0.5210, 0.6330] whatever the inputs, so
    ACTIVE was unreachable and the whole metadata domain was worth 0.91
    standard deviations of the diffusion noise applied on top of it.

    The direction is seeded from ``fact_id`` so a tier is reproducible. A tier
    nobody can reproduce is a support case nobody can answer. Direction still
    varies per fact, so facts do not collapse onto one point.
    """
    r_eq = _retention_radius(access_count, age_days, importance, curve)
    rng = np.random.default_rng(_direction_seed(fact_id))
    direction = rng.standard_normal(dim)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-8:
        direction = np.ones(dim)
        norm = float(np.linalg.norm(direction))
    return (direction / norm * r_eq).tolist()


def _anchor_to_retention(
    position: list[float], access_count: int, age_days: float, importance: float,
    curve: object | None = None,
) -> list[float]:
    """Keep the direction a Langevin step chose; take the radius from retention.

    The step alone has one stable place, the edge of the ball: as the radius
    grows its inward pull is scaled by ``lambda^-2`` and fades to nothing,
    while its outward forgetting push does not. Integrated once per pass, it
    archived every memory -- one used 1,000 times at full importance within
    about 800 passes -- and archived memories are removed from recall. The
    radius is ``1 - R(t)``, the relationship the seed already uses, so a pass
    can move a memory between tiers only when its retention moves.
    """
    radius = _retention_radius(access_count, age_days, importance, curve)
    vector = np.asarray(position, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        vector = np.ones(_LANGEVIN_DIM, dtype=np.float64)
        norm = float(np.linalg.norm(vector))
    return (vector / norm * radius).tolist()


def _langevin_pass(ld: object, fact_dicts: list[dict],
                   curve: object | None = None) -> list[dict]:
    """One maintenance step for each fact: the Langevin step, anchored.

    Returns ``fact_id``, ``position`` and ``lifecycle`` per fact, the shape
    ``_persist_lifecycle`` expects. ``curve`` is the person's
    (``EbbinghausCurve(config.forgetting)``); the tier is the shared decay
    zone (:func:`langevin_lifecycle`), not the radius bands, so this pass can
    never archive a memory the decay pass keeps.
    """
    results = []
    for fact, stepped in zip(fact_dicts, ld.batch_step(fact_dicts)):
        position = _anchor_to_retention(
            stepped["position"], fact["access_count"], fact["age_days"],
            fact["importance"], curve,
        )
        results.append({
            "fact_id": fact["fact_id"],
            "position": position,
            "lifecycle": langevin_lifecycle(fact["access_count"], fact["age_days"],
                                            fact["importance"], curve),
        })
    return results
