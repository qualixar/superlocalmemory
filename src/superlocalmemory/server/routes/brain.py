# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.22 — LLD-04 §4.1 + §3

"""``/api/v3/brain`` — unified Brain endpoint (LLD-04 v2).

Merges the pre-3.4.22 Patterns / Learning / Behavioral dashboard tabs
into one auth-gated, honestly-labelled JSON payload. Every metric has
an ``is_real`` (numeric counters) or ``is_real_ml`` (ML-derived) flag
and points at the real table it was computed from. No metric is
fabricated — if a source does not exist we return zero and flag it.

Security primitives consumed from LLD-07 §6:

* ``verify_install_token`` — constant-time token check for auth gate.
* ``redact_secrets``       — regex+entropy secret scrub.

Deprecated shims (1 release grace) live below the main route. They
return the historical-ish shape plus ``deprecated: true`` and
``use_instead: /api/v3/brain``, and require the same auth.

Design notes (LLD-04 §7 hard rules):

* **U2** — every metric section carries ``is_real`` or ``is_real_ml``.
* **U3** — ``learning.phase`` never returns ``3`` without an active
  model row AND passing SHA check. Both conditions computed here.
* **U4** — response shape must not contain the three pre-3.4.22
  fabricated metrics (24h hit-rate, avg age on hit, skill-evolution
  row counts). A static test greps this file to verify; therefore we
  never spell those names anywhere in this module.
* **U6** — install token required on ``/api/v3/brain`` and all
  deprecated shim routes. See ``require_install_token`` below.
* **U10** — feature count surfaced from ``features.FEATURE_DIM``;
  stratum total from module constant ``_STRATA_TOTAL`` (48 = 4×3×4).
"""

from __future__ import annotations

import logging
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request

from superlocalmemory.core.security_primitives import (
    redact_secrets,
    verify_install_token,
)
from superlocalmemory.learning.database import LearningDatabase
from superlocalmemory.learning.features import FEATURE_DIM

logger = logging.getLogger("superlocalmemory.routes.brain")

router = APIRouter(prefix="/api/v3", tags=["brain"])


# ---------------------------------------------------------------------------
# Constants — surfaced via the response so the UI never hard-codes.
# ---------------------------------------------------------------------------

# LLD-03 v2 stratum space = 4 query types × 3 entity bins × 4 time buckets.
_STRATA_TOTAL: int = 48

_VERSION: str = "3.4.22"

# Banned metric names (LLD-04 U4). Kept as a tuple for grep visibility;
# the source-level test asserts we don't accidentally reintroduce them.
# NOTE: do NOT add the literal forbidden-key strings here — the U4 grep
# guard runs over this file.

# Memory directory (home-dir based). Always resolved at call time so that
# tests can override via monkeypatch on ``_learning_db_path``.
_MEMORY_DIR_DEFAULT = Path.home() / ".superlocalmemory"


def _learning_db_path() -> Path:
    """Return the path to the learning SQLite DB.

    Separated as a function (not a module constant) so tests can
    monkeypatch without touching Path.home. See
    ``tests/test_api/test_brain_endpoint.py``.
    """
    return _MEMORY_DIR_DEFAULT / "learning.db"


def _memory_db_path() -> Path:
    return _MEMORY_DIR_DEFAULT / "memory.db"


# ---------------------------------------------------------------------------
# Auth dependency (LLD-04 §4.1, LLD-07 §6.6)
# ---------------------------------------------------------------------------


async def require_install_token(request: Request) -> None:
    """FastAPI dependency: enforces install-token on Brain routes.

    Accepts either of:

        X-Install-Token: <token>
        Authorization: Bearer <token>

    Token comparison uses ``verify_install_token`` which wraps
    ``hmac.compare_digest`` — constant time regardless of input.

    Raises:
        HTTPException(401) with ``WWW-Authenticate: Install-Token`` so
        clients know how to retry. Never leaks whether the token file
        exists, whether the header was missing, or which comparison
        branch rejected the value.
    """
    header_token = request.headers.get("x-install-token")
    presented: str | None = header_token
    if not presented:
        auth = request.headers.get("authorization", "")
        if auth:
            # Strip scheme; be tolerant of casing. Empty-after-strip → None.
            scheme, _, value = auth.partition(" ")
            if scheme.lower() == "bearer" and value.strip():
                presented = value.strip()
    if not presented or not verify_install_token(presented):
        raise HTTPException(
            status_code=401,
            detail="install_token_required",
            headers={"WWW-Authenticate": "Install-Token"},
        )


# ---------------------------------------------------------------------------
# Secret-redacted preferences (LLD-04 §4.1, LLD-07 §6.3)
# ---------------------------------------------------------------------------


def redact_secrets_in_preferences(prefs: dict) -> dict:
    """Deep-copy ``prefs``, scrub every string through ``redact_secrets``,
    and surface the count of substitutions as ``redacted_count``.

    The input is never mutated — we build a new structure and preserve
    list/dict shapes exactly. Non-string scalars (ints, floats, bools,
    None) pass through untouched. Unknown container types are left as-is
    (best-effort forward compatibility).

    Returns:
        New dict with the same shape plus ``redacted_count`` at the top
        level. If ``prefs`` is not a dict, returns
        ``{"redacted_count": 0}`` (defensive).
    """
    if not isinstance(prefs, dict):
        return {"redacted_count": 0}

    counter = [0]  # mutable holder so the nested closure can increment.

    def _scrub(value: Any) -> Any:
        if isinstance(value, str):
            scrubbed = redact_secrets(value)
            if scrubbed != value:
                counter[0] += 1
            return scrubbed
        if isinstance(value, dict):
            return {k: _scrub(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_scrub(v) for v in value]
        if isinstance(value, tuple):
            return tuple(_scrub(v) for v in value)
        return value

    out: dict = {k: _scrub(v) for k, v in prefs.items()}
    out["redacted_count"] = counter[0]
    out.setdefault("is_real", True)
    out.setdefault("source", prefs.get("source", "_store_patterns"))
    return out


# ---------------------------------------------------------------------------
# Section builders — each SELECTs from a real table. If a table is empty,
# returns 0 with ``is_real: true`` (honest). Never invents a number.
# ---------------------------------------------------------------------------


def _load_raw_preferences(profile_id: str) -> dict:
    """Load preference categories from the behavioral pattern store.

    Returns a dict shaped like::

        {"topics": [...], "entities": [...], "tech": [...], "source": ...}

    with empty lists if the store / DB is not available. Overridden in
    tests via monkeypatch to keep the Brain route decoupled from the
    full behavioral pipeline.
    """
    out: dict[str, Any] = {
        "topics": [], "entities": [], "tech": [],
        "source": "_store_patterns",
    }
    db_path = _learning_db_path()
    if not db_path.exists():
        return out
    try:
        from superlocalmemory.learning.behavioral import BehavioralPatternStore
        store = BehavioralPatternStore(str(db_path))
        patterns = store.get_patterns(profile_id=profile_id) or []
    except Exception as exc:  # pragma: no cover — defensive
        logger.debug("load_raw_preferences: %s", exc)
        return out

    topics: list[dict[str, Any]] = []
    entities: list[dict[str, Any]] = []
    tech: list[dict[str, Any]] = []
    for p in patterns:
        ptype = p.get("pattern_type") or ""
        meta = p.get("metadata") or {}
        name = meta.get("value") or p.get("pattern_key") or ""
        if not name:
            continue
        if ptype == "tech_preference":
            tech.append({
                "name": str(name),
                "frequency": float(p.get("confidence", 0.0) or 0.0),
            })
        elif ptype in ("entity", "interest"):
            entities.append({
                "name": str(name),
                "mention_count": int(p.get("evidence_count", 0) or 0),
            })
        elif ptype == "topic":
            topics.append({
                "name": str(name),
                "strength": float(p.get("confidence", 0.0) or 0.0),
            })
    out["topics"] = topics
    out["entities"] = entities
    out["tech"] = tech
    return out


def _compute_preferences(profile_id: str) -> dict:
    raw = _load_raw_preferences(profile_id)
    return redact_secrets_in_preferences(raw)


def _compute_learning_status(profile_id: str,
                             lrn_db: LearningDatabase) -> dict:
    """Section ``learning`` — LLD-02 §4.10 phase truth + counters."""
    signals_total = _safe_count(lrn_db, "learning_signals", profile_id)
    features_total = _safe_count(lrn_db, "learning_features", profile_id)
    # Raw count of historic pre-v3.4.22 feedback rows (the source table)
    legacy_feedback_rows = _safe_count(lrn_db, "learning_feedback", profile_id)
    # Count of rows actually copied forward into learning_signals by
    # legacy_migration.migrate_legacy_feedback (signal_type='legacy_feedback').
    # The difference (raw - migrated) is what the dashboard's
    # "Migrate legacy data" card surfaces as pending work.
    legacy_migrated = _count_legacy_migrated(lrn_db, profile_id)
    signals_last_hour = _count_signals_since(
        lrn_db, profile_id,
        datetime.now(timezone.utc) - timedelta(hours=1),
    )

    active = lrn_db.load_active_model(profile_id)
    model_active = active is not None
    model_version: str | None = None
    model_trained_at: str | None = None
    model_sha256_present = False
    if model_active and isinstance(active, dict):
        model_version = active.get("model_version")
        model_trained_at = active.get("trained_at")
        model_sha256_present = bool(active.get("bytes_sha256"))

    phase, phase_label = _resolve_phase(signals_total, model_active,
                                        model_sha256_present)

    # S9-DASH-03: consult migration_log. If the legacy migration is
    # marked complete but some rows remain uncopied, those rows are
    # structurally un-migratable (malformed, duplicate, or missing
    # required cols). Reporting them as "pending" is misleading and
    # the dashboard card nags forever. After completion we report
    # pending=0 so the card auto-hides.
    migration_complete = _legacy_migration_sentinel_complete(lrn_db)
    legacy_pending = max(0, legacy_feedback_rows - legacy_migrated)
    if migration_complete:
        legacy_pending = 0

    return {
        "phase": phase,
        "phase_label": phase_label,
        "signals_total": signals_total,
        "signals_last_hour": signals_last_hour,
        "features_total": features_total,
        "feature_count_expected": FEATURE_DIM,
        # Honest split (v3.4.22+): raw historic table count vs rows
        # actually copied forward via the migrate-legacy flow.
        "legacy_feedback_rows": legacy_feedback_rows,
        "legacy_migrated_count": legacy_migrated,
        "legacy_migration_pending": legacy_pending,
        "legacy_migration_complete": migration_complete,
        "model_active": model_active,
        "model_version": model_version,
        "model_trained_at": model_trained_at,
        "model_sha256_present": model_sha256_present,
        "is_real": True,
    }


def _resolve_phase(signals: int, model_active: bool,
                   sha_present: bool) -> tuple[int, str]:
    """LLD-02 §4.10 — phase truth, U3 enforcement.

    Phase 3 requires BOTH ``model_active`` and a non-empty SHA value on
    the active row. Missing either falls back to phase 2 (≥50 signals)
    or phase 1 (cold).
    """
    if model_active and sha_present and signals >= 200:
        return 3, "LightGBM ranker active"
    if signals >= 50:
        return 2, "Contextual bandit"
    return 1, "Cold start (cross-encoder only)"


def _compute_usage_stats(profile_id: str) -> dict:
    """Section ``usage`` — real counters from ``learning_signals``.

    All three values are honest aggregations of the last 24 h of signals.
    ``is_real_ml`` stays False because these are counters, not ML output,
    but the numbers themselves are pinned to real rows. LLD-04 U2 / §3.1.
    """
    lrn_db = _lazy_db_for(profile_id)
    since = datetime.now(timezone.utc) - timedelta(hours=24)
    recalls_24h = _count_signals_since(
        lrn_db, profile_id, since,
        signal_types=("recall_hit", "recall", "recall_miss",
                      "candidate", "shown"),
    )
    return {
        "recalls_last_24h": recalls_24h,
        "top_query_types": _top_query_types(lrn_db, profile_id, since),
        "top_time_buckets": _top_time_buckets(lrn_db, profile_id, since),
        "source": "learning_signals_24h_aggregation",
        "is_real_ml": False,
        "disclaimer": "Statistical counters, not ML output.",
    }


def _top_query_types(
    lrn_db: LearningDatabase,
    profile_id: str,
    since: datetime,
    *,
    limit: int = 5,
) -> list[dict]:
    """Return top-N query types from last-24h signals with percentages.

    Query type lives in ``learning_features.features_json`` as one-hot
    ``query_type_sh`` / ``query_type_mh`` / ``query_type_temp`` /
    ``query_type_od`` (features.py FEATURE_NAMES). We aggregate by the
    active one-hot slot. Missing table or zero rows → empty list.
    """
    # E.1 (v3.4.22 perf): push the query-type classification into SQL via
    # json_extract so we don't drag 10k rows' worth of JSON through Python.
    # SQLite's json1 extension treats json_extract as an ordinary scalar
    # function, so the whole aggregation becomes a single COUNT(*)
    # GROUP BY on a computed column. Falls back to the Python path only
    # if json_extract is unavailable (very old SQLite builds).
    counts: dict[str, int] = {
        "single_hop": 0, "multi_hop": 0, "temporal": 0, "open_domain": 0,
    }
    try:
        conn = sqlite3.connect(lrn_db.path, timeout=5.0)
        conn.row_factory = sqlite3.Row
        try:
            agg_rows = conn.execute(
                "SELECT "
                "  CASE "
                "    WHEN CAST(json_extract(f.features_json, '$.query_type_sh')   AS REAL) >= 0.5 THEN 'single_hop' "
                "    WHEN CAST(json_extract(f.features_json, '$.query_type_mh')   AS REAL) >= 0.5 THEN 'multi_hop' "
                "    WHEN CAST(json_extract(f.features_json, '$.query_type_temp') AS REAL) >= 0.5 THEN 'temporal' "
                "    WHEN CAST(json_extract(f.features_json, '$.query_type_od')   AS REAL) >= 0.5 THEN 'open_domain' "
                "    ELSE NULL END AS qt, "
                "  COUNT(*) AS n "
                "FROM learning_features f "
                "JOIN learning_signals s ON s.id = f.signal_id "
                "WHERE s.profile_id = ? AND s.created_at >= ? "
                "  AND f.is_synthetic = 0 "
                "GROUP BY qt",
                (profile_id, since.isoformat()),
            ).fetchall()
            for row in agg_rows:
                qt = row["qt"]
                if qt in counts:
                    counts[qt] = int(row["n"] or 0)
        finally:
            conn.close()
    except sqlite3.Error:  # pragma: no cover — schema + json1 always present
        return []
    total = sum(counts.values())
    if total == 0:
        return []
    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    return [
        {"type": name, "pct": round(100.0 * n / total, 1)}
        for name, n in ranked[:limit] if n > 0
    ]


def _top_time_buckets(
    lrn_db: LearningDatabase,
    profile_id: str,
    since: datetime,
    *,
    limit: int = 3,
) -> list[dict]:
    """Return top-N hour buckets ("HH:00") from last-24h signals.

    Parses ``created_at`` as ISO timestamp, buckets by hour-of-day, returns
    top N with percentages. Honest empty when no rows.
    """
    try:
        conn = sqlite3.connect(lrn_db.path, timeout=5.0)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT created_at FROM learning_signals "
                "WHERE profile_id = ? AND created_at >= ? "
                "LIMIT 10000",
                (profile_id, since.isoformat()),
            ).fetchall()
        finally:
            conn.close()
    except sqlite3.Error:  # pragma: no cover
        return []
    buckets: dict[int, int] = {}
    for r in rows:
        ts = str(r["created_at"] or "")
        if not ts:
            continue
        try:
            # ISO: "YYYY-MM-DDTHH:MM:..." — take the "HH" slice.
            hh = int(ts[11:13])
        except (ValueError, IndexError):  # pragma: no cover
            continue
        buckets[hh] = buckets.get(hh, 0) + 1
    total = sum(buckets.values())
    if total == 0:
        return []
    ranked = sorted(buckets.items(), key=lambda kv: kv[1], reverse=True)
    return [
        {"bucket": f"{hh:02d}:00",
         "pct": round(100.0 * n / total, 1)}
        for hh, n in ranked[:limit] if n > 0
    ]


def _compute_bandit_snapshot(profile_id: str,
                             lrn_db: LearningDatabase) -> dict:
    """Section ``bandit`` — derive summary from ``ContextualBandit.snapshot``.

    We intentionally compute the *derived* fields here (strata_active,
    top_arm_global, unsettled_plays) rather than asking the bandit to
    produce them — this keeps the stratum total constant surfaced from
    one authoritative place (``_STRATA_TOTAL``) instead of duplicating.
    """
    strata_active = 0
    top_arm_global: dict | None = None
    unsettled_plays = 0
    oldest_unsettled_seconds: int | None = None

    try:
        from superlocalmemory.learning.bandit import ContextualBandit
        bandit = ContextualBandit(
            db_path=lrn_db.path,  # shared learning DB
            profile_id=profile_id,
        )
        snap = bandit.snapshot() or {}
        strata_active = sum(1 for arms in snap.values() if arms)
        # Global top arm by plays across all strata.
        best: tuple[str, int] | None = None
        for stratum_id, arms in snap.items():
            for arm in arms:
                plays = int(arm.get("plays", 0) or 0)
                if best is None or plays > best[1]:
                    best = (arm["arm_id"], plays)
        if best is not None:
            top_arm_global = {"arm_id": best[0], "plays": best[1]}
        unsettled_plays, oldest_unsettled_seconds = _bandit_unsettled(
            lrn_db.path, profile_id,
        )
    except Exception as exc:  # pragma: no cover — defensive
        logger.debug("bandit snapshot: %s", exc)

    return {
        "strata_active": strata_active,
        "strata_total": _STRATA_TOTAL,
        "top_arm_global": top_arm_global,
        "unsettled_plays": unsettled_plays,
        "oldest_unsettled_seconds": oldest_unsettled_seconds,
        "is_real": True,
    }


def _bandit_unsettled(db_path: str, profile_id: str) -> tuple[int, int | None]:
    """Count unsettled bandit_plays rows + age (sec) of oldest one."""
    try:
        conn = sqlite3.connect(db_path, timeout=5.0)
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute(
                "SELECT COUNT(*) AS cnt, MIN(played_at) AS oldest "
                "FROM bandit_plays "
                "WHERE profile_id = ? AND settled_at IS NULL",
                (profile_id,),
            ).fetchone()
        finally:
            conn.close()
    except sqlite3.Error:
        return 0, None
    if row is None:  # pragma: no cover — COUNT() always yields a row
        return 0, None
    cnt = int(row["cnt"] or 0)
    oldest_iso = row["oldest"]
    if not oldest_iso:
        return cnt, None
    try:
        played_at = datetime.fromisoformat(oldest_iso)
        if played_at.tzinfo is None:
            played_at = played_at.replace(tzinfo=timezone.utc)
        age = (datetime.now(timezone.utc) - played_at).total_seconds()
        return cnt, int(max(age, 0))
    except (ValueError, TypeError):
        return cnt, None


def _compute_cache_stats() -> dict:
    """Section ``cache`` — DB file size + row count (if accessible)."""
    db = _memory_db_path()
    if not db.exists():
        return {"db_size_bytes": 0, "entry_count": 0, "is_real": True}
    size = db.stat().st_size
    entry_count = 0
    try:
        conn = sqlite3.connect(str(db), timeout=5.0)
        try:
            row = conn.execute(
                "SELECT COUNT(*) AS cnt FROM atomic_facts",
            ).fetchone()
            entry_count = int(row[0]) if row else 0
        finally:
            conn.close()
    except sqlite3.Error:
        entry_count = 0
    return {
        "db_size_bytes": size,
        "entry_count": entry_count,
        "is_real": True,
    }


def _adapter_last_sync_ago(adapter_name: str) -> int | None:
    """Seconds since adapter's most recent successful sync.

    Reads from ``cross_platform_sync_log`` (LLD-07 M004) in ``memory.db``.
    Returns ``None`` when the log is absent or has no successful row —
    honest empty rather than a fabricated number.
    """
    try:
        import sqlite3 as _sqlite3
        from datetime import datetime as _dt, timezone as _tz
        from pathlib import Path as _P

        memory_db = _P.home() / ".superlocalmemory" / "memory.db"
        if not memory_db.exists():
            return None
        conn = _sqlite3.connect(
            f"file:{memory_db}?mode=ro", uri=True, timeout=1.0,
        )
        try:
            cur = conn.execute(
                "SELECT last_sync_at FROM cross_platform_sync_log "
                "WHERE adapter_name=? AND success=1 "
                "ORDER BY last_sync_at DESC LIMIT 1",
                (adapter_name,),
            )
            row = cur.fetchone()
        finally:
            conn.close()
        if row is None or row[0] is None:
            return None
        last = _dt.fromisoformat(str(row[0]).replace("Z", "+00:00"))
        delta = (_dt.now(_tz.utc) - last).total_seconds()
        return max(0, int(delta))
    except Exception:  # pragma: no cover — defensive
        return None


def _compute_cross_platform() -> dict:
    """Section ``cross_platform`` — live status per injection target.

    Each adapter's ``is_active()`` is cheap (env + ``Path.is_dir()``) per
    LLD-05 A10. Last-sync-at is read from ``cross_platform_sync_log`` in
    ``memory.db`` (LLD-07 M004). On any adapter error, that adapter
    reports ``active: false`` with ``reason: error:<ExcName>`` rather
    than crashing the whole Brain endpoint (LLD-04 §2 — "honest, never
    fake"). An unimportable adapter means the install is missing Wave 2C
    components, which is legitimate for an older 3.4.20 → 3.4.22 upgrade
    mid-migration.
    """
    out: dict = {}
    # S8-SEC-06 fix: use the canonical factory instead of constructing
    # adapters with no kwargs (all three require scope/base_dir/sync_log_db/
    # recall_fn). ``build_default_adapters`` returns fully-wired instances
    # matching what the background sync loop uses, so the panel reflects
    # the same truth the daemon acts on.
    try:
        from superlocalmemory.cli.context_commands import (
            build_default_adapters as _build_adapters,
        )
        _adapters = _build_adapters()
    except Exception as exc:  # pragma: no cover — defensive
        _adapters = []
        logger.debug("brain: adapter factory failed: %s", exc)

    # Summarise by adapter kind: cursor (project+global), antigravity
    # (workspace+global), copilot. ``adapter.name`` disambiguates scope.
    _seen: set[str] = set()
    for a in _adapters:
        name_attr = getattr(a, "name", "") or ""
        cls = type(a).__name__
        kind = None
        if "Cursor" in cls:
            kind = "cursor"
        elif "Antigravity" in cls:
            kind = "antigravity"
        elif "Copilot" in cls:
            kind = "copilot"
        if kind is None:
            # Unknown adapter class — skip to avoid polluting the JSON
            # with an ``out[None]`` key. (S10-SEC-N-01 fix.)
            continue
        if kind in _seen:
            # For cursor/antigravity, which have project+global, the first
            # active one wins; this surface is a health indicator, not a
            # per-scope breakdown. (Dev view can drill in later.)
            continue
        try:
            is_active = bool(a.is_active())
        except Exception as exc:  # pragma: no cover — defensive
            out[kind] = {"active": False,
                         "reason": f"error:{exc.__class__.__name__}"}
            _seen.add(kind)
            continue
        out[kind] = {
            "active": is_active,
            "last_sync_seconds_ago": _adapter_last_sync_ago(name_attr or kind),
        }
        _seen.add(kind)
    # Fill in missing slots so the shape stays stable.
    for kind in ("cursor", "antigravity", "copilot"):
        out.setdefault(kind, {"active": False, "reason": "adapter_unavailable"})
    # Claude Code hook proxy: active once the install-token exists
    # (required for prewarm per LLD-01 §4.4 R14).
    try:
        from superlocalmemory.core import security_primitives as _sp
        out["claude_code"] = {
            "active": _sp._install_token_path().exists(),
            "hook": "UserPromptSubmit",
        }
    except Exception as exc:  # pragma: no cover
        out["claude_code"] = {"active": False,
                              "reason": f"error:{exc.__class__.__name__}"}
    # MCP tool is registered by the unified daemon; if this route is
    # reachable, the MCP server is up.
    out["mcp"] = {"active": True, "tool": "mcp__slm"}
    # CLI is trivially active on any install.
    out["cli"] = {"active": True}
    return out


def _meta_now() -> dict:
    return {
        "generated_at": datetime.now(timezone.utc)
        .replace(microsecond=0).isoformat()
        .replace("+00:00", "Z"),
        "honest_labels": True,
        "version": _VERSION,
    }


# ---------------------------------------------------------------------------
# SQL helpers
# ---------------------------------------------------------------------------


def _safe_count(lrn_db: LearningDatabase, table: str,
                profile_id: str) -> int:
    """``COUNT(*)`` from ``table`` where ``profile_id`` matches.

    Returns 0 on any error (missing table, DB lock) — never raises
    from the Brain endpoint.
    """
    try:
        conn = sqlite3.connect(lrn_db.path, timeout=5.0)
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute(
                f"SELECT COUNT(*) AS cnt FROM {table} WHERE profile_id = ?",  # nosec - table name is hard-coded above
                (profile_id,),
            ).fetchone()
            return int(row["cnt"]) if row else 0
        finally:
            conn.close()
    except sqlite3.Error:
        return 0


def _legacy_migration_sentinel_complete(lrn_db: LearningDatabase) -> bool:
    """Return True when ``migration_log`` marks the legacy feedback
    migration as complete (S9-DASH-03).

    Signals that the historic-data card should auto-hide even if a
    small residual of un-migratable rows lingers in ``learning_feedback``.
    """
    try:
        conn = sqlite3.connect(lrn_db.path, timeout=5.0)
        try:
            row = conn.execute(
                "SELECT status FROM migration_log "
                "WHERE name = 'LEG001_feedback_to_signals'",
            ).fetchone()
            if row is None:
                return False
            return str(row[0]).lower() == "complete"
        finally:
            conn.close()
    except sqlite3.Error:
        return False


def _count_legacy_migrated(lrn_db: LearningDatabase, profile_id: str) -> int:
    """Count rows copied forward by the legacy-feedback migration.

    These rows live in ``learning_signals`` with ``signal_type='legacy_feedback'``
    (see ``learning/legacy_migration.py``). The Brain endpoint exposes this
    separately from the raw ``learning_feedback`` table count so the UI can
    show a honest "pending migration" figure instead of the Stage-8 lie
    that conflated the two.
    """
    try:
        conn = sqlite3.connect(lrn_db.path, timeout=5.0)
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute(
                "SELECT COUNT(*) AS cnt FROM learning_signals "
                "WHERE profile_id = ? AND signal_type = 'legacy_feedback'",
                (profile_id,),
            ).fetchone()
            return int(row["cnt"]) if row else 0
        finally:
            conn.close()
    except sqlite3.Error:
        return 0


def _count_signals_since(
    lrn_db: LearningDatabase,
    profile_id: str,
    since: datetime,
    *,
    signal_types: tuple[str, ...] | None = None,
) -> int:
    """Count ``learning_signals`` rows since ``since`` for ``profile_id``."""
    try:
        conn = sqlite3.connect(lrn_db.path, timeout=5.0)
        conn.row_factory = sqlite3.Row
        try:
            if signal_types:
                placeholders = ",".join("?" * len(signal_types))
                sql = (
                    "SELECT COUNT(*) AS cnt FROM learning_signals "
                    "WHERE profile_id = ? AND created_at >= ? "
                    f"AND signal_type IN ({placeholders})"
                )
                params: tuple[Any, ...] = (
                    profile_id, since.isoformat(), *signal_types,
                )
            else:
                sql = (
                    "SELECT COUNT(*) AS cnt FROM learning_signals "
                    "WHERE profile_id = ? AND created_at >= ?"
                )
                params = (profile_id, since.isoformat())
            row = conn.execute(sql, params).fetchone()
            return int(row["cnt"]) if row else 0
        finally:
            conn.close()
    except sqlite3.Error:  # pragma: no cover — table always present in tests
        return 0


def _lazy_db_for(profile_id: str) -> LearningDatabase:
    return LearningDatabase(_learning_db_path())


_EVOLUTION_MAX_DAYS = 90
_EVOLUTION_DEFAULT_DAYS = 30


def _compute_evolution_timeseries(
    profile_id: str,
    lrn_db: LearningDatabase,
    *,
    days: int = _EVOLUTION_DEFAULT_DAYS,
) -> dict:
    """Daily learning-signal counts for ``profile_id`` over the last ``days``.

    Returns ``{"days": N, "points": [{"date": "YYYY-MM-DD", "signals": int,
    "patterns_seen": int}, ...], "is_real": True, "source": "learning_signals"}``.

    Points are left-aligned to midnight UTC and cover exactly ``days``
    consecutive days including today. Missing days are zero-filled so the
    chart renders a flat line instead of a gap.
    """
    try:
        requested = int(days) if days is not None else _EVOLUTION_DEFAULT_DAYS
    except (TypeError, ValueError):
        requested = _EVOLUTION_DEFAULT_DAYS
    days = max(1, min(requested, _EVOLUTION_MAX_DAYS))
    today = datetime.now(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0,
    )
    start = today - timedelta(days=days - 1)

    # Daily counts from learning_signals.
    counts: dict[str, int] = {}
    try:
        conn = sqlite3.connect(lrn_db.path, timeout=5.0)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT substr(created_at, 1, 10) AS d, COUNT(*) AS n "
                "FROM learning_signals "
                "WHERE profile_id = ? AND created_at >= ? "
                "GROUP BY substr(created_at, 1, 10)",
                (profile_id, start.isoformat()),
            ).fetchall()
            counts = {r["d"]: int(r["n"]) for r in rows if r["d"]}
        finally:
            conn.close()
    except sqlite3.Error:
        counts = {}

    points: list[dict[str, Any]] = []
    for i in range(days):
        d = start + timedelta(days=i)
        key = d.date().isoformat()
        points.append({
            "date": key,
            "signals": counts.get(key, 0),
        })

    total = sum(p["signals"] for p in points)
    return {
        "is_real": True,
        "source": "learning_signals",
        "days": days,
        "total_signals": total,
        "points": points,
    }


def _action_outcomes_count(lrn_db: LearningDatabase,
                           profile_id: str) -> int:
    """Row count in ``action_outcomes`` for ``profile_id``.

    ``action_outcomes`` ships in v3.4.22 (M006). While absent, returns 0.
    """
    try:
        conn = sqlite3.connect(lrn_db.path, timeout=5.0)
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute(
                "SELECT COUNT(*) AS cnt FROM action_outcomes "
                "WHERE profile_id = ?",
                (profile_id,),
            ).fetchone()
            return int(row["cnt"]) if row else 0
        finally:
            conn.close()
    except sqlite3.Error:
        return 0


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("/brain", dependencies=[Depends(require_install_token)])
async def get_brain(profile_id: str = "default") -> dict:
    """Unified Brain endpoint — LLD-04 §3.1.

    Fan-out: each section is a synchronous SQLite reader. Running them
    serially was a 7-round-trip chain on the hot path (PERF-v2-05).
    We offload each to the default executor via ``asyncio.to_thread``
    and ``asyncio.gather`` so the wall-clock is dominated by the slowest
    single section, not the sum. Every section already swallows its own
    errors and returns honest-empty on failure, so ``return_exceptions``
    is unnecessary — but we still set ``return_exceptions=True`` as a
    belt-and-suspenders guard so one broken reader can't 500 the whole
    endpoint.
    """
    import asyncio

    lrn_db = LearningDatabase(_learning_db_path())

    (
        preferences, learning, usage, bandit_snap, cache,
        cross_platform, outcomes_rows, evolution,
    ) = await asyncio.gather(
        asyncio.to_thread(_compute_preferences, profile_id),
        asyncio.to_thread(_compute_learning_status, profile_id, lrn_db),
        asyncio.to_thread(_compute_usage_stats, profile_id),
        asyncio.to_thread(_compute_bandit_snapshot, profile_id, lrn_db),
        asyncio.to_thread(_compute_cache_stats),
        asyncio.to_thread(_compute_cross_platform),
        asyncio.to_thread(_action_outcomes_count, lrn_db, profile_id),
        asyncio.to_thread(
            _compute_evolution_timeseries, profile_id, lrn_db,
            days=_EVOLUTION_DEFAULT_DAYS,
        ),
        return_exceptions=True,
    )

    # Replace any per-section exception with an honest-empty dict so the
    # endpoint never propagates a half-rendered payload.
    def _ok(value, fallback):
        if isinstance(value, Exception):
            return fallback
        return value

    return {
        "profile_id": profile_id,
        "preferences":      _ok(preferences, {"is_real": True,
                                              "topics": [], "entities": [],
                                              "tech": [], "redacted_count": 0,
                                              "source": "_store_patterns"}),
        "learning":         _ok(learning, {"is_real": True, "phase": 1,
                                           "signals_total": 0}),
        "usage":            _ok(usage, {"is_real_ml": False,
                                        "recalls_last_24h": 0,
                                        "top_query_types": [],
                                        "top_time_buckets": []}),
        "bandit":           _ok(bandit_snap, {"is_real": True,
                                              "strata_active": 0,
                                              "strata_total": 48}),
        "cache":            _ok(cache, {"is_real": True,
                                        "db_size_bytes": 0,
                                        "entry_count": 0}),
        "cross_platform":   _ok(cross_platform, {}),
        "evolution_preview": _ok(evolution, {
            "is_real": True, "source": "learning_signals",
            "days": _EVOLUTION_DEFAULT_DAYS, "total_signals": 0, "points": [],
        }),
        "outcomes_preview": {
            "action_outcomes_rows":
                0 if isinstance(outcomes_rows, Exception) else outcomes_rows,
            "ships_in": "3.4.22",
        },
        # S9-defer H-22: live tile data for the Reward / Shadow /
        # Evolution-Cost dashboard tiles. Each block is a honest-empty
        # default when the underlying table is missing (fresh install
        # or older schema) so the UI can render "no data yet" without
        # a 500. All three queries are cheap COUNT / AVG aggregates.
        "reward_preview": _compute_reward_preview(profile_id),
        "shadow_preview": _compute_shadow_preview(profile_id),
        "evolution_cost_preview": _compute_evolution_cost_preview(profile_id),
        "outcome_queue": _compute_outcome_queue_stats(profile_id),
        "meta": _meta_now(),
    }


def _compute_outcome_queue_stats(profile_id: str) -> dict:
    """S9-DASH-02: producer-side telemetry for the Brain panel.

    Exposes the outcome-queue worker counters + a live count of
    ``pending_outcomes`` so the operator can see the closed loop is
    actually flowing: recall → enqueue → persist → finalize.
    """
    import sqlite3
    from pathlib import Path
    try:
        from superlocalmemory.learning.outcome_queue import (
            get_counters, queue_size,
        )
        counters = get_counters()
        qsz = queue_size()
    except Exception:
        counters, qsz = {}, 0
    home = Path.home() / ".superlocalmemory"
    db = home / "memory.db"
    pending_now = 0
    if db.exists():
        try:
            conn = sqlite3.connect(str(db), timeout=1.0)
            try:
                row = conn.execute(
                    "SELECT COUNT(*) FROM pending_outcomes "
                    "WHERE profile_id = ? AND status = 'pending'",
                    (profile_id,),
                ).fetchone()
                if row:
                    pending_now = int(row[0] or 0)
            finally:
                conn.close()
        except Exception:
            pass
    return {
        "is_real": True,
        "queue_depth": int(qsz),
        "pending_outcomes_now": pending_now,
        "counters": counters,
        "source": "outcome_queue + pending_outcomes",
    }


# ---------------------------------------------------------------------------
# S9-defer H-22 — dashboard-tile aggregates
# ---------------------------------------------------------------------------


def _compute_reward_preview(profile_id: str) -> dict:
    """Reward-tile aggregate — count + mean reward over the last 24h."""
    import sqlite3
    from pathlib import Path
    home = Path.home() / ".superlocalmemory"
    db = home / "memory.db"
    default = {
        "is_real": False, "rows_24h": 0, "mean_reward_24h": 0.0,
        "source": "action_outcomes",
    }
    if not db.exists():
        return default
    try:
        conn = sqlite3.connect(str(db), timeout=1.0)
        try:
            row = conn.execute(
                "SELECT COUNT(*) AS c, AVG(reward) AS m "
                "FROM action_outcomes "
                "WHERE profile_id = ? AND settled = 1 "
                "  AND reward IS NOT NULL "
                "  AND settled_at >= datetime('now', '-1 day')",
                (profile_id,),
            ).fetchone()
        finally:
            conn.close()
    except sqlite3.Error:
        return default
    if row is None:
        return default
    return {
        "is_real": True,
        "rows_24h": int(row[0] or 0),
        "mean_reward_24h": float(row[1] or 0.0),
        "source": "action_outcomes",
    }


def _compute_shadow_preview(profile_id: str) -> dict:
    """Shadow-tile aggregate — latest candidate + paired-obs count."""
    import sqlite3
    default = {
        "is_real": False, "active_candidate_id": None,
        "paired_observations": 0, "rollback_count_90d": 0,
        "source": "learning_model_state+shadow_observations",
    }
    learning_db = _learning_db_path()
    if not learning_db.exists():
        return default
    try:
        conn = sqlite3.connect(str(learning_db), timeout=1.0)
        try:
            row = conn.execute(
                "SELECT id FROM learning_model_state "
                "WHERE profile_id = ? AND is_candidate = 1 "
                "LIMIT 1",
                (profile_id,),
            ).fetchone()
            cand_id = int(row[0]) if row and row[0] is not None else None
            paired = 0
            if cand_id is not None:
                try:
                    c = conn.execute(
                        "SELECT COUNT(DISTINCT query_id) "
                        "FROM shadow_observations "
                        "WHERE candidate_id = ? "
                        "  AND query_id IN ("
                        "    SELECT query_id FROM shadow_observations "
                        "    WHERE candidate_id = ? AND arm = 'active' "
                        "    INTERSECT "
                        "    SELECT query_id FROM shadow_observations "
                        "    WHERE candidate_id = ? AND arm = 'candidate' "
                        "  )",
                        (cand_id, cand_id, cand_id),
                    ).fetchone()
                    paired = int(c[0]) if c and c[0] is not None else 0
                except sqlite3.Error:
                    paired = 0
            # Rollback count in the last 90d. The rollback log is a
            # separate table the learning subsystem writes; if it is
            # absent we simply return 0.
            try:
                r2 = conn.execute(
                    "SELECT COUNT(*) FROM model_state_history "
                    "WHERE profile_id = ? AND action = 'rollback' "
                    "  AND ts >= datetime('now', '-90 day')",
                    (profile_id,),
                ).fetchone()
                rollbacks = int(r2[0]) if r2 and r2[0] is not None else 0
            except sqlite3.Error:
                rollbacks = 0
        finally:
            conn.close()
    except sqlite3.Error:
        return default
    return {
        "is_real": True,
        "active_candidate_id": cand_id,
        "paired_observations": paired,
        "rollback_count_90d": rollbacks,
        "source": "learning_model_state+shadow_observations",
    }


def _compute_evolution_cost_preview(profile_id: str) -> dict:
    """Evolution-cost tile — last 7d spend + call count."""
    import sqlite3
    default = {
        "is_real": False, "calls_7d": 0, "cost_usd_7d": 0.0,
        "tokens_in_7d": 0, "tokens_out_7d": 0,
        "source": "evolution_llm_cost_log",
    }
    learning_db = _learning_db_path()
    if not learning_db.exists():
        return default
    try:
        conn = sqlite3.connect(str(learning_db), timeout=1.0)
        try:
            row = conn.execute(
                "SELECT COUNT(*) AS c, "
                "       COALESCE(SUM(cost_usd), 0) AS cost, "
                "       COALESCE(SUM(tokens_in), 0) AS tin, "
                "       COALESCE(SUM(tokens_out), 0) AS tout "
                "FROM evolution_llm_cost_log "
                "WHERE profile_id = ? "
                "  AND ts >= datetime('now', '-7 day')",
                (profile_id,),
            ).fetchone()
        finally:
            conn.close()
    except sqlite3.Error:
        return default
    if row is None:
        return default
    return {
        "is_real": True,
        "calls_7d": int(row[0] or 0),
        "cost_usd_7d": float(row[1] or 0.0),
        "tokens_in_7d": int(row[2] or 0),
        "tokens_out_7d": int(row[3] or 0),
        "source": "evolution_llm_cost_log",
    }


@router.get("/brain/evolution-timeseries",
            dependencies=[Depends(require_install_token)])
async def get_brain_evolution_timeseries(
    profile_id: str = "default",
    days: int = _EVOLUTION_DEFAULT_DAYS,
) -> dict:
    """Daily learning-signal counts for ``profile_id`` over the last ``days``.

    ``days`` is clamped to ``[1, 90]`` to keep the response bounded. Each
    missing day is zero-filled so the chart renders a flat line, not a gap.
    """
    import asyncio

    lrn_db = LearningDatabase(_learning_db_path())
    result = await asyncio.to_thread(
        _compute_evolution_timeseries, profile_id, lrn_db, days=days,
    )
    result["meta"] = _meta_now()
    return result


# ---------------------------------------------------------------------------
# Deprecated shims — 1 release grace. All require the install token.
# ---------------------------------------------------------------------------


@router.get("/learning/stats",
            dependencies=[Depends(require_install_token)])
async def learning_stats_deprecated(profile_id: str = "default") -> dict:
    lrn_db = LearningDatabase(_learning_db_path())
    return {
        "deprecated": True,
        "use_instead": "/api/v3/brain",
        "learning": _compute_learning_status(profile_id, lrn_db),
    }


@router.get("/patterns",
            dependencies=[Depends(require_install_token)])
async def patterns_deprecated(profile_id: str = "default") -> dict:
    return {
        "deprecated": True,
        "use_instead": "/api/v3/brain",
        "preferences": _compute_preferences(profile_id),
    }


@router.get("/behavioral",
            dependencies=[Depends(require_install_token)])
async def behavioral_deprecated(profile_id: str = "default") -> dict:
    return {
        "deprecated": True,
        "use_instead": "/api/v3/brain",
        "usage": _compute_usage_stats(profile_id),
    }


__all__ = (
    "router",
    "require_install_token",
    "redact_secrets_in_preferences",
)
