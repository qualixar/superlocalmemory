# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com
"""SuperLocalMemory V3 - Learning Routes
 - AGPL-3.0-or-later

Routes: /api/learning/status, /api/feedback, /api/feedback/dwell,
        /api/feedback/stats, /api/learning/backup, /api/learning/reset,
        /api/learning/retrain
Uses V3 learning modules: FeedbackCollector, EngagementTracker, AdaptiveLearner.
"""
import shutil
import logging
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter

from .helpers import get_active_profile, MEMORY_DIR

logger = logging.getLogger("superlocalmemory.routes.learning")
router = APIRouter()

LEARNING_DB = MEMORY_DIR / "learning.db"


# ---------------------------------------------------------------------------
# LLD-02 §4.10 — Dashboard phase truth
# ---------------------------------------------------------------------------


def _compute_ranker_phase(
    profile_id: str,
    *,
    learning_db_path: Path | None = None,
) -> dict:
    """Return {phase, label, model_active, signals} — LLD-02 §4.10.

    Phase 3 requires BOTH an active (is_active=1) row AND a successful
    SHA-256 verification on the model_cache load. Tampered bytes fall
    back to phase 2.
    """
    from superlocalmemory.learning.database import LearningDatabase
    from superlocalmemory.learning.model_cache import load_active, invalidate

    db_path = Path(learning_db_path) if learning_db_path else LEARNING_DB
    if not db_path.exists():
        return {
            "phase": 1,
            "label": "Cold start (cross-encoder only)",
            "model_active": False,
            "signals": 0,
        }

    db = LearningDatabase(db_path)
    try:
        signals = db.count_signals(profile_id)
    except Exception as exc:
        logger.warning("count_signals failed: %s", exc)
        signals = 0

    # Force a cache-bypass load — the dashboard read is rare and we want
    # tamper detection to surface immediately.
    invalidate(profile_id)
    try:
        model = load_active(db, profile_id, use_cache=False)
    except Exception as exc:
        logger.warning("load_active failed: %s", exc)
        model = None

    active = model is not None

    if active and signals >= 200:
        return {
            "phase": 3,
            "label": "LightGBM ranker active",
            "model_active": True,
            "signals": signals,
        }
    if signals >= 50:
        return {
            "phase": 2,
            "label": "Contextual bandit",
            "model_active": False,
            "signals": signals,
        }
    return {
        "phase": 1,
        "label": "Cold start (cross-encoder only)",
        "model_active": False,
        "signals": signals,
    }


@router.get("/api/learning/ranker_phase")
async def ranker_phase():
    """Dashboard endpoint — LLD-02 §4.10 phase truth."""
    try:
        profile = get_active_profile()
    except Exception:
        profile = "default"
    return _compute_ranker_phase(profile)

# Feature detection
LEARNING_AVAILABLE = False
BEHAVIORAL_AVAILABLE = False
try:
    from superlocalmemory.learning.feedback import FeedbackCollector
    from superlocalmemory.learning.engagement import EngagementTracker
    from superlocalmemory.learning.ranker import AdaptiveRanker
    LEARNING_AVAILABLE = True
except ImportError as e:
    logger.warning("V3 learning primary import failed: %s", e)
    try:
        from superlocalmemory.learning.adaptive import AdaptiveLearner
        LEARNING_AVAILABLE = True
    except ImportError as e2:
        logger.warning("V3 learning fallback import failed: %s", e2)

try:
    from superlocalmemory.learning.behavioral import BehavioralPatternStore
    BEHAVIORAL_AVAILABLE = True
except ImportError as e:
    logger.warning("V3 behavioral import failed: %s", e)

# Lazy singletons
_feedback: FeedbackCollector | None = None
_engagement: EngagementTracker | None = None


def _get_feedback() -> "FeedbackCollector | None":
    global _feedback
    if _feedback is None and LEARNING_AVAILABLE:
        try:
            _feedback = FeedbackCollector(str(LEARNING_DB))
        except Exception:
            pass
    return _feedback


def _get_engagement() -> "EngagementTracker | None":
    global _engagement
    if _engagement is None and LEARNING_AVAILABLE:
        try:
            _engagement = EngagementTracker(str(LEARNING_DB))
        except Exception:
            pass
    return _engagement


@router.get("/api/learning/status")
async def learning_status():
    """Get comprehensive learning system status for dashboard."""
    if not LEARNING_AVAILABLE:
        return {
            "available": False, "ranking_phase": None,
            "stats": None, "tech_preferences": [], "workflow_patterns": [],
            "source_scores": {}, "engagement": None,
            "message": "Learning features not installed.",
        }

    result = {"available": True}

    try:
        active_profile = get_active_profile()
        result["active_profile"] = active_profile

        # Real signal count from V3.1 learning_feedback table
        signal_count = 0
        unique_queries = 0
        try:
            from superlocalmemory.learning.feedback import FeedbackCollector
            from pathlib import Path
            import sqlite3 as _sqlite3
            learning_db = LEARNING_DB
            if learning_db.exists():
                collector = FeedbackCollector(learning_db)
                signal_count = collector.get_feedback_count(active_profile)
                # Count unique queries for the dashboard
                _conn = _sqlite3.connect(str(learning_db))
                _conn.row_factory = _sqlite3.Row
                try:
                    _row = _conn.execute(
                        "SELECT COUNT(DISTINCT query_hash) AS cnt "
                        "FROM learning_feedback WHERE profile_id = ?",
                        (active_profile,),
                    ).fetchone()
                    unique_queries = _row["cnt"] if _row else 0
                except Exception:
                    pass
                finally:
                    _conn.close()
        except Exception:
            pass

        # Ranking phase based on real signal count
        if signal_count >= 200:
            result["ranking_phase"] = "ml_model"
        elif signal_count >= 20:
            result["ranking_phase"] = "rule_based"
        else:
            result["ranking_phase"] = "baseline"

        # Feedback stats — merge old system + new V3.1 signals
        stats_dict = {"feedback_count": signal_count, "unique_queries": unique_queries, "active_profile": active_profile}
        feedback = _get_feedback()
        if feedback:
            try:
                old_stats = feedback.get_feedback_summary(active_profile)
                if isinstance(old_stats, dict):
                    old_stats["feedback_count"] = signal_count
                    old_stats["unique_queries"] = unique_queries
                    old_stats["active_profile"] = active_profile
                    stats_dict = old_stats
            except Exception as exc:
                logger.debug("feedback summary: %s", exc)

        result["stats"] = stats_dict
        result["profile_feedback"] = {
            "profile": active_profile,
            "signals": signal_count,
        }

        # Engagement — v3.4.8: Fixed method name (was get_engagement_stats, actual is get_stats)
        engagement = _get_engagement()
        if engagement:
            try:
                stats = engagement.get_stats(active_profile)
                health = engagement.get_health(active_profile)
                active_days = stats.get("active_days", 0)
                total_events = stats.get("total_events", 0)
                memories_per_day = (
                    round(total_events / active_days, 1) if active_days > 0 else 0
                )
                result["engagement"] = {
                    "health_status": health.upper(),
                    "days_active": active_days,
                    "memories_per_day": memories_per_day,
                    "total_events": total_events,
                    "recall_count": stats.get("recall_count", 0),
                    "store_count": stats.get("store_count", 0),
                    "session_count": stats.get("session_count", 0),
                    "engagement_score": stats.get("engagement_score", 0),
                }
            except Exception as exc:
                logger.debug("engagement stats: %s", exc)
                result["engagement"] = None
        else:
            result["engagement"] = None

        # Tech preferences + workflow patterns from V3.1 behavioral store
        try:
            from superlocalmemory.learning.behavioral import BehavioralPatternStore
            from pathlib import Path
            learning_db = LEARNING_DB
            if learning_db.exists():
                store = BehavioralPatternStore(str(learning_db))
                all_patterns = store.get_patterns(profile_id=active_profile)
                tech = [
                    {"key": "tech", "value": p.get("metadata", {}).get("value", p.get("pattern_key", "")),
                     "confidence": p.get("confidence", 0), "evidence": p.get("evidence_count", 0)}
                    for p in all_patterns if p.get("pattern_type") == "tech_preference"
                ]
                workflows = [
                    {"type": p.get("pattern_type"), "key": p.get("pattern_key", ""),
                     "value": p.get("metadata", {}).get("value", ""),
                     "confidence": p.get("confidence", 0)}
                    for p in all_patterns if p.get("pattern_type") in ("temporal", "interest")
                ]
                result["tech_preferences"] = tech
                result["workflow_patterns"] = workflows

                # Privacy stats
                import os
                db_size = os.path.getsize(str(learning_db)) // 1024 if learning_db.exists() else 0
                stats_dict["db_size_kb"] = db_size
                stats_dict["transferable_patterns"] = len(all_patterns)
                stats_dict["models_trained"] = 1 if signal_count >= 200 else 0
                stats_dict["tracked_sources"] = len(set(
                    p.get("pattern_type") for p in all_patterns
                ))
            else:
                result["tech_preferences"] = []
                result["workflow_patterns"] = []
        except Exception as exc:
            logger.error("Error fetching behavioral patterns: %s", exc, exc_info=True)
            result["tech_preferences"] = []
            result["workflow_patterns"] = []
            result["pattern_error"] = str(exc)
        result["source_scores"] = {}

    except Exception as e:
        logger.error("Error getting learning status: %s", e)
        result["error"] = str(e)

    return result


# ============================================================================
# FEEDBACK ENDPOINTS
# ============================================================================

@router.post("/api/feedback")
async def record_feedback(data: dict):
    """Record explicit feedback from dashboard (thumbs up/down, pin)."""
    if not LEARNING_AVAILABLE:
        return {"success": False, "error": "Learning system not available"}

    memory_id = data.get("memory_id")
    query = data.get("query", "")
    feedback_type = data.get("feedback_type")

    if not memory_id or not feedback_type:
        return {"success": False, "error": "memory_id and feedback_type required"}

    valid_types = {"thumbs_up", "thumbs_down", "pin"}
    if feedback_type not in valid_types:
        return {"success": False, "error": f"Invalid feedback_type. Must be one of: {valid_types}"}

    try:
        feedback = _get_feedback()
        if not feedback:
            return {"success": False, "error": "Feedback collector not initialized"}

        row_id = feedback.record_dashboard_feedback(
            memory_id=str(memory_id), query=query, feedback_type=feedback_type,
        )

        return {
            "success": True,
            "message": f"Feedback recorded: {feedback_type} for memory #{memory_id}",
            "feedback_id": row_id,
        }
    except Exception as e:
        logger.error("Error recording feedback: %s", e)
        return {"success": False, "error": str(e)}


@router.post("/api/feedback/dwell")
async def record_dwell(data: dict):
    """Record dwell time feedback from dashboard modal."""
    if not LEARNING_AVAILABLE:
        return {"success": False, "error": "Learning system not available"}

    memory_id = data.get("memory_id")
    query = data.get("query", "")
    dwell_time = data.get("dwell_time", 0)

    if not memory_id:
        return {"success": False, "error": "memory_id required"}

    try:
        dwell_seconds = float(dwell_time)
    except (ValueError, TypeError):
        return {"success": False, "error": "dwell_time must be a number"}

    if dwell_seconds >= 10.0:
        feedback_type = "dwell_positive"
    elif dwell_seconds < 2.0:
        feedback_type = "dwell_negative"
    else:
        return {"success": True, "message": "Dwell time in neutral range, no signal recorded"}

    try:
        feedback = _get_feedback()
        if not feedback:
            return {"success": False, "error": "Feedback collector not initialized"}

        row_id = feedback.record_dashboard_feedback(
            memory_id=str(memory_id), query=query, feedback_type=feedback_type,
        )

        return {
            "success": True,
            "message": f"Dwell feedback recorded: {feedback_type} ({dwell_seconds:.1f}s)",
            "feedback_id": row_id,
        }
    except Exception as e:
        logger.error("Error recording dwell: %s", e)
        return {"success": False, "error": str(e)}


@router.get("/api/feedback/stats")
async def feedback_stats():
    """Get feedback signal statistics for dashboard progress bar."""
    if not LEARNING_AVAILABLE:
        return {
            "total_signals": 0, "ranking_phase": "baseline",
            "progress": 0, "target": 200, "available": False,
        }

    try:
        feedback = _get_feedback()
        total = 0
        by_channel = {}
        by_type = {}

        if feedback:
            profile = get_active_profile()
            summary = feedback.get_feedback_summary(profile)
            total = summary.get("total", summary.get("total_signals", 0))
            by_channel = summary.get("by_channel", {})
            by_type = summary.get("by_type", {})

        target = 200
        progress = min(total / target * 100, 100)

        return {
            "total_signals": total, "ranking_phase": "baseline",
            "progress": round(progress, 1), "target": target,
            "by_channel": by_channel, "by_type": by_type, "available": True,
        }
    except Exception as e:
        logger.error("Error getting feedback stats: %s", e)
        return {"total_signals": 0, "ranking_phase": "baseline", "progress": 0, "error": str(e)}


# ============================================================================
# PATTERNS ENDPOINT (v3.4.1 — CRITICAL FIX: frontend calls /api/patterns)
# ============================================================================


@router.delete("/api/patterns/delete")
async def delete_pattern(data: dict) -> dict:
    """S9-DASH-04: delete a single auto-detected pattern by key.

    Body: ``{pattern_type: str, pattern_key: str}``

    Returns ``{success: bool, deleted: int}``. The pattern is scoped
    to the active profile so cross-profile deletion is impossible.
    """
    if not BEHAVIORAL_AVAILABLE:
        return {"success": False, "error": "Behavioral engine not available"}
    ptype = (data or {}).get("pattern_type", "")
    pkey = (data or {}).get("pattern_key", "")
    if not ptype or not pkey:
        return {
            "success": False,
            "error": "pattern_type and pattern_key are required",
        }
    try:
        profile = get_active_profile()
        store = BehavioralPatternStore(str(LEARNING_DB))
        deleted = store.delete_pattern_by_key(
            profile_id=profile,
            pattern_type=ptype,
            pattern_key=pkey,
        )
        return {
            "success": True, "deleted": int(deleted),
            "active_profile": profile,
        }
    except Exception as exc:  # noqa: BLE001
        logger.error("delete_pattern failed: %s", exc)
        return {"success": False, "error": str(exc)}


@router.get("/api/patterns")
async def get_patterns():
    """Get learned behavioral patterns for the Patterns dashboard tab.

    v3.4.1: This endpoint was MISSING — patterns.js calls /api/patterns
    but no backend route existed. The frontend always showed 'No patterns'.
    Now queries BehavioralPatternStore + learning signals.
    """
    active_profile = get_active_profile()

    patterns: dict = {
        "preference": [],
        "style": [],
        "terminology": [],
        "workflow": [],
    }
    result: dict = {
        "available": BEHAVIORAL_AVAILABLE or LEARNING_AVAILABLE,
        "patterns": patterns,
        "signal_stats": {},
    }

    # Behavioral patterns from BehavioralPatternStore
    if BEHAVIORAL_AVAILABLE:
        try:
            store = BehavioralPatternStore(str(LEARNING_DB))
            all_patterns = store.get_patterns(profile_id=active_profile)
            for p in all_patterns:
                ptype = p.get("pattern_type", "")
                entry = {
                    "key": p.get("pattern_key", ""),
                    "value": p.get("metadata", {}).get("value", ""),
                    "confidence": round(float(p.get("confidence", 0)), 3),
                    "evidence_count": p.get("evidence_count", 0),
                    "created_at": p.get("created_at", ""),
                    "updated_at": p.get("updated_at", ""),
                }
                if ptype == "tech_preference":
                    patterns["preference"].append(entry)
                elif ptype == "style":
                    patterns["style"].append(entry)
                elif ptype == "terminology":
                    patterns["terminology"].append(entry)
                elif ptype in ("temporal", "interest", "workflow"):
                    patterns["workflow"].append(entry)
                else:
                    patterns["preference"].append(entry)
        except Exception as exc:
            logger.error("Error loading patterns from behavioral store: %s", exc)
            result["pattern_error"] = str(exc)

    # Learning signal stats (feedback count, co-retrieval, channel credits)
    try:
        from superlocalmemory.learning.signals import LearningSignals
        signals = LearningSignals(str(LEARNING_DB))
        result["signal_stats"] = signals.get_signal_stats(active_profile)
    except Exception as exc:
        logger.debug("Signal stats unavailable: %s", exc)

    # Graph intelligence contribution to learning (v3.4.1)
    try:
        import sqlite3 as _sqlite3
        from superlocalmemory.server.routes.helpers import DB_PATH
        if DB_PATH.exists():
            conn = _sqlite3.connect(str(DB_PATH))
            conn.row_factory = _sqlite3.Row
            row = conn.execute(
                "SELECT COUNT(*) AS cnt, COUNT(DISTINCT community_id) AS communities, "
                "ROUND(AVG(pagerank_score), 4) AS avg_pagerank "
                "FROM fact_importance WHERE profile_id = ?",
                (active_profile,),
            ).fetchone()
            if row:
                d = dict(row)
                result["graph_intelligence"] = {
                    "facts_analyzed": d.get("cnt", 0),
                    "communities_detected": d.get("communities", 0),
                    "avg_pagerank": float(d.get("avg_pagerank", 0) or 0),
                }
            conn.close()
    except Exception:
        pass

    return result


@router.post("/api/learning/backup")
async def learning_backup():
    """Backup learning.db to a timestamped file."""
    try:
        if not LEARNING_DB.exists():
            return {"success": False, "error": "No learning.db found"}

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        backup_name = f"learning.db.backup_{timestamp}"
        backup_path = MEMORY_DIR / backup_name
        shutil.copy2(str(LEARNING_DB), str(backup_path))

        return {
            "success": True, "filename": backup_name,
            "path": str(backup_path),
            "message": f"Learning DB backed up to {backup_name}",
        }
    except Exception as e:
        logger.error("Error backing up learning DB: %s", e)
        return {"success": False, "error": str(e)}


@router.post("/api/learning/reset")
async def learning_reset():
    """Reset all learning data for the active profile. Memories preserved."""
    if not LEARNING_AVAILABLE:
        return {"success": False, "error": "Learning system not available"}
    try:
        from superlocalmemory.learning.database import LearningDatabase
        db = LearningDatabase(LEARNING_DB)
        profile_id = get_active_profile() or "default"
        db.reset(profile_id=profile_id)
        return {
            "success": True,
            "message": "Learning data reset. Memories preserved.",
            "profile_id": profile_id,
        }
    except Exception as exc:  # noqa: BLE001
        logger.error("learning_reset failed: %s", exc)
        return {"success": False, "error": str(exc)}


@router.post("/api/learning/retrain")
async def learning_retrain(data: dict | None = None):
    """Force a retrain of the LightGBM ranker.

    Body (optional, JSON):
        ``{"include_synthetic": bool}`` — when True, migrated legacy rows
        (``is_synthetic=1``) participate in training. Default False.
    """
    if not LEARNING_AVAILABLE:
        return {"success": False, "error": "Learning system not available"}
    include_synthetic = bool(
        data and data.get("include_synthetic")
    ) if isinstance(data, dict) else False
    try:
        from superlocalmemory.learning.consolidation_worker import (
            _retrain_ranker_impl,
        )
        profile_id = get_active_profile() or "default"
        trained = _retrain_ranker_impl(
            LEARNING_DB,
            profile_id,
            include_synthetic=include_synthetic,
        )
        if trained:
            return {
                "success": True,
                "trained": True,
                "profile_id": profile_id,
                "include_synthetic": include_synthetic,
            }
        return {
            "success": True,
            "trained": False,
            "profile_id": profile_id,
            "include_synthetic": include_synthetic,
            "message": (
                "Not enough training rows yet. Keep using SLM, or run "
                "legacy migration + retry with include_synthetic=true."
            ),
        }
    except Exception as exc:  # noqa: BLE001
        logger.error("learning_retrain failed: %s", exc)
        return {"success": False, "error": str(exc)}


@router.post("/api/learning/migrate-legacy")
async def learning_migrate_legacy():
    """Copy ``learning_feedback`` rows into LLD-02 tables for training.

    Idempotent: subsequent calls detect the migration_log sentinel and
    return ``already_done=True`` without re-copying. The rows are written
    with ``is_synthetic=1`` to preserve provenance; the trainer must be
    invoked with ``include_synthetic=True`` to use them.
    """
    if not LEARNING_AVAILABLE:
        return {"success": False, "error": "Learning system not available"}
    try:
        from superlocalmemory.learning.legacy_migration import (
            migrate_legacy_feedback,
        )
        stats = migrate_legacy_feedback(LEARNING_DB)
        return {"success": True, **stats}
    except Exception as exc:  # noqa: BLE001
        logger.error("learning_migrate_legacy failed: %s", exc)
        return {"success": False, "error": str(exc)}
