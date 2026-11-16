# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""What the answer-check routes stand on: the saved settings, the running
check, and the status the dashboard reads.

* Settings are read from and written to one place (core/answer_check_state.py),
  so a route always reads what it wrote, whichever per-mode config file
  ``SLMConfig.load()`` happens to prefer.
* The running check is the judge on the published engine. After every change
  ``enforce()`` holds it to the saved settings: if a switch failed to take the
  hosted check (or its reordering) off, it is taken off here, and a route only
  answers success once that is true.
* ``build_status()`` only reads. It reports what actually runs (the live
  judge) rather than what the config would build, and never starts anything.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from superlocalmemory.core import (
    answer_check_state,
    engine_wiring,
    judge_keys,
    judge_selection,
    laya_runtime,
)
from superlocalmemory.core.config import SLMConfig
from superlocalmemory.retrieval import judge_recipe, sufficiency
from superlocalmemory.server.routes.helpers import MEMORY_DIR

logger = logging.getLogger("superlocalmemory.server.routes.answer_check")

#: Every change to the answer check runs under this lock: changes are rare and
#: made by a person, and each one is read-check-write across the settings and
#: the running check, which must not interleave with another. Re-entrant
#: because a finished background job applies its result through the same path.
MUTATION_LOCK = threading.RLock()

KEY_PROBLEM = (
    "The saved key file was changed outside SLM, so it isn't used. "
    "Remove the key and enter it again."
)
NOT_CONFIRMED = (
    "Your choice is saved, but SLM couldn't confirm the online check "
    "stopped. Restart SLM to be sure."
)

#: How long a worked-out "what would run" answer is reused when no engine is
#: up to ask directly. Working it out can start a Python process; the
#: dashboard polls every 2 s while something installs. A change made through
#: these routes is seen at once (the settings are part of the cache key).
_RESOLVE_TTL_S = 30.0
_resolve_cache: dict[tuple, tuple[float, str]] = {}
_resolve_lock = threading.Lock()


# -- settings ------------------------------------------------------------------


def state_dir() -> Path:
    return (MEMORY_DIR / "config.json").parent


def effective_retrieval() -> Any:
    """The retrieval config with the saved answer-check settings over it."""
    return answer_check_state.overlay(SLMConfig.load().retrieval, state_dir())


def save(change: Callable[[dict[str, Any]], Mapping[str, Any]]) -> Any:
    """Write through the one store; returns the effective retrieval after it."""
    answer_check_state.update(
        state_dir(), change, seed=lambda: answer_check_state.snapshot(effective_retrieval())
    )
    return effective_retrieval()


def without_rerank(values: Mapping[str, Any]) -> dict[str, Any]:
    """Reordering off, its consent included: it never resumes without being asked."""
    return {**values, "sufficiency_jev_rerank": False, "sufficiency_jev_rerank_consent": False}


def laya_paths(status: Any) -> dict[str, str]:
    paths = {"sufficiency_python": status.python, "sufficiency_hf_home": status.hf_home}
    if status.model_path:
        paths["sufficiency_model"] = status.model_path
    return paths


def forget_managed_install(
    values: Mapping[str, Any], *, every_path: bool = False
) -> dict[str, Any]:
    """After the managed install is deleted: its paths go (so the dashboard
    says "Not installed", not "Needs a check"), and an explicit on-device
    choice becomes Off. Paths to an install someone adopted are kept — unless
    ``every_path``: the person chose to stop using that install too."""
    run_dir = laya_runtime.runtime_dir()
    roots = {str(run_dir), str(run_dir.resolve())}

    def inside(path: Any) -> bool:
        text = str(path or "")
        return any(text == root or text.startswith(root + os.sep) for root in roots)

    out = dict(values)
    if every_path or any(
        inside(values.get(k))
        for k in ("sufficiency_python", "sufficiency_model", "sufficiency_hf_home")
    ):
        defaults = answer_check_state.field_defaults()
        for key in ("sufficiency_python", "sufficiency_hf_home", "sufficiency_model"):
            out[key] = defaults[key]
    if values.get("sufficiency_judge") == judge_selection.MODE_LAYA:
        out["sufficiency_judge"] = judge_selection.MODE_OFF
    return out


_MODE_LABELS = {"laya": "on this Mac", "jev": "online with Jev", "off": "off"}


def rerank_chosen(retrieval: Any) -> bool:
    """Whether the person turned Jev's reordering on (and accepted its notice) —
    remembered while another option is chosen, when it does not run."""
    return (
        getattr(retrieval, "sufficiency_jev_rerank", False) is True
        and getattr(retrieval, "sufficiency_jev_rerank_consent", False) is True
    )


def switched_message(mode: str, retrieval: Any) -> str:
    """What the person is told after choosing ``mode`` — including, plainly,
    that Jev's reordering pauses while another option is chosen."""
    text = f"Answer check is now {_MODE_LABELS.get(mode, mode)}. Only one option runs at a time."
    if mode != judge_selection.MODE_JEV and rerank_chosen(retrieval):
        text += (
            " Reordering with Jev is off while it isn't chosen; it comes back "
            "when you choose Jev again."
        )
    return text


# -- the running check ------------------------------------------------------------


def live_engine(app_state: Any) -> Any:
    """The published retrieval engine, or None. Never builds one: a status read
    or a switch must not start an engine just to look at it."""
    engine = getattr(app_state, "engine", None)
    return getattr(engine, "_retrieval_engine", None) if engine is not None else None


def live_judge(app_state: Any) -> Any:
    retrieval_engine = live_engine(app_state)
    return getattr(retrieval_engine, "_sufficiency_judge", None) if retrieval_engine else None


def attach(app_state: Any, retrieval: Any) -> str:
    """Rebuild the running check from ``retrieval``; "off" when there is no engine."""
    retrieval_engine = live_engine(app_state)
    attach_fn = getattr(engine_wiring, "attach_sufficiency_judge", None)
    if retrieval_engine is None or attach_fn is None:
        return judge_selection.MODE_OFF
    try:
        return str(attach_fn(retrieval_engine, retrieval))
    except Exception:  # noqa: BLE001 — logged; enforce() still checks the result
        logger.exception("answer_check: switching the running check failed")
        return judge_selection.MODE_OFF


def force_off(app_state: Any) -> None:
    retrieval_engine = live_engine(app_state)
    if retrieval_engine is None:
        return
    try:
        judge_selection.swap_sufficiency_judge(retrieval_engine, lambda: None)
    except Exception:  # noqa: BLE001 — logged; enforce() reports what is left
        logger.exception("answer_check: taking the running check off failed")


def detach_laya(app_state: Any) -> bool:
    """Stop the on-device check if that is what runs; True when it did.

    Decided by the running judge, not the configured mode: in "auto" the mode
    string never says "laya" while Laya runs. A hot reconfigure can leave a
    second engine holding the same model, so it is stopped in every holder,
    not only the published engine, before any of its files are deleted.
    """
    stopped = False
    if judge_selection.backend_of(live_judge(app_state)) == judge_selection.MODE_LAYA:
        force_off(app_state)
        stopped = True
    return judge_selection.stop_on_device_check() or stopped


def _permitted(retrieval: Any, judge: Any) -> bool:
    if judge_selection.backend_of(judge) != judge_selection.MODE_JEV:
        return True
    if (
        getattr(retrieval, "sufficiency_judge", "") != judge_selection.MODE_JEV
        or getattr(retrieval, "sufficiency_jev_consent", False) is not True
    ):
        return False
    running_k = getattr(judge, "rerank_k", 0)
    running_k = running_k if isinstance(running_k, int) else 0
    return running_k <= judge_selection.jev_rerank_k(retrieval)


def enforce(app_state: Any, retrieval: Any) -> bool:
    """Hold the running check to the saved settings. True once it complies.

    The hosted check may run only while it is chosen and consented to, and may
    reorder only as many memories as its own consent allows. Anything more is
    taken off here — the person said no, so "the switch did not take" is not
    an acceptable reason for memories to keep leaving the machine.
    """
    if _permitted(retrieval, live_judge(app_state)):
        return True
    logger.warning("answer_check: the running online check outlived its consent; stopping it")
    force_off(app_state)
    return _permitted(retrieval, live_judge(app_state))


# -- status -----------------------------------------------------------------------


def key_state(provider: str) -> dict[str, Any]:
    """Whether a key is saved, its last four characters, or what is wrong with it."""
    store = judge_keys.JudgeKeyStore()
    try:
        return {
            "has_key": bool(store.has_key(provider)),
            "key_hint": store.masked(provider),
            "key_problem": "",
        }
    except judge_keys.JudgeKeyStoreError:
        logger.warning("answer_check: the saved %s key file failed its safety check", provider)
        return {"has_key": False, "key_hint": "", "key_problem": KEY_PROBLEM}


def _resolved(retrieval: Any, laya_status: Any) -> str:
    """What the config would run, when no engine is up to ask. Cached briefly."""
    key = (
        tuple(sorted(answer_check_state.snapshot(retrieval).items())),
        laya_status.state,
        laya_status.python,
        laya_status.model_path,
    )
    now = time.monotonic()
    with _resolve_lock:
        hit = _resolve_cache.get(key)
        if hit is not None and hit[0] > now:
            return hit[1]
    value = _resolve_uncached(retrieval, laya_status)
    with _resolve_lock:
        _resolve_cache.clear()  # one entry is all a status poll ever needs
        _resolve_cache[key] = (now + _RESOLVE_TTL_S, value)
    return value


def _resolve_uncached(retrieval: Any, laya_status: Any) -> str:
    resolve_fn = getattr(engine_wiring, "resolve_judge_mode", None)
    if resolve_fn is not None:
        try:
            return str(resolve_fn(retrieval))
        except Exception:  # noqa: BLE001
            logger.exception("answer_check: working out the active check failed")
    mode = str(getattr(retrieval, "sufficiency_judge", "off") or "off").lower()
    if mode in ("laya", "jev", "off"):
        return mode
    # "auto" never resolves to the hosted check.
    return "laya" if laya_status.state == laya_runtime.STATE_READY else "off"


def active_backend(app_state: Any, retrieval: Any, laya_status: Any) -> str:
    """What actually runs: the live judge when an engine is up."""
    if live_engine(app_state) is not None:
        return judge_selection.backend_of(live_judge(app_state))
    return _resolved(retrieval, laya_status)


def rerank_status(retrieval: Any, active: str, judge: Any) -> dict[str, Any]:
    from superlocalmemory.retrieval.jev_rerank import clamp_rerank_k

    enabled = rerank_chosen(retrieval)
    running = judge_selection.jev_rerank_k(retrieval) > 0 and active == judge_selection.MODE_JEV
    if running and judge is not None:
        running_k = getattr(judge, "rerank_k", 0)
        running = isinstance(running_k, int) and running_k > 0
    return {
        "enabled": enabled,
        "active": running,
        "k": clamp_rerank_k(getattr(retrieval, "sufficiency_jev_rerank_k", None)),
    }


def calibration_status(active: str, reordering: bool) -> str | None:
    """How the running check's confidence was measured; None when none runs."""
    if active == judge_selection.MODE_LAYA:
        return judge_recipe.calibration_or_unmeasured("laya").status
    if active == judge_selection.MODE_JEV:
        backend = "jev-listwise" if reordering else "jev"
        return judge_recipe.calibration_or_unmeasured(backend).status
    return None


def adopt_status() -> dict[str, Any]:
    job = laya_runtime.LayaAdoptJob.instance()
    return {"running": job.running, **job.status().to_dict()}


# -- "Test" results: kept, so a reload still shows when it last worked -------------

_TESTS_FILE = "answer-check-tests.json"
_tests_lock = threading.Lock()


def _tests_path() -> Path:
    return state_dir() / _TESTS_FILE


def record_test(kind: str, ok: bool, message: str, seconds: float | None) -> dict[str, Any]:
    """Save the outcome of a Test ("laya" or "jev"); returns what was saved.
    ``message`` is plain language and never contains a key."""
    import json
    from datetime import datetime, timezone

    result = {
        "ok": bool(ok),
        "message": str(message)[:300],
        "seconds": None if seconds is None else round(float(seconds), 1),
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with _tests_lock:
        data = read_tests()
        data[kind] = result
        path = _tests_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(f".tmp-{os.getpid()}")
            tmp.write_text(json.dumps(data), encoding="utf-8")
            os.replace(tmp, path)
        except OSError:
            logger.warning("answer_check: couldn't save the test result")
    return result


def read_tests() -> dict[str, Any]:
    import json

    try:
        data = json.loads(_tests_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return (
        {k: v for k, v in data.items() if k in ("laya", "jev") and isinstance(v, dict)}
        if isinstance(data, dict)
        else {}
    )


def forget_test(kind: str) -> None:
    with _tests_lock:
        data = read_tests()
        if data.pop(kind, None) is None:
            return
        try:
            _tests_path().write_text(__import__("json").dumps(data), encoding="utf-8")
        except OSError:
            logger.warning("answer_check: couldn't clear the test result")


def build_status(app_state: Any) -> dict[str, Any]:
    """The dashboard's view. Reads only: writes nothing, starts nothing."""
    retrieval = effective_retrieval()
    laya_status = laya_runtime.detect(retrieval)
    judge = live_judge(app_state)
    active = active_backend(app_state, retrieval, laya_status)
    provider = retrieval.sufficiency_jev_provider
    rerank = rerank_status(retrieval, active, judge)
    from superlocalmemory.server.answer_check_execution import execution_status

    execution = execution_status(judge, live_engine(app_state))
    laya_view = laya_status.to_dict()
    if active == "laya" and execution["state"] == "unavailable":
        laya_view = {
            **laya_view,
            "state": "needs_check",
            "action": "check",
            "error": execution["detail"],
            "message": execution["detail"],
        }
    job = getattr(app_state, "answer_calibration_job", None)
    return {
        "mode": retrieval.sufficiency_judge,
        "active": active,
        "laya": laya_view,
        "execution": execution,
        "calibration": job.status()
        if job is not None
        else {"state": "idle", "done": 0, "total": 0},
        "adopt": adopt_status(),
        "setup_running": laya_runtime.LayaInstallJob.instance().running,
        "laya_test_running": laya_runtime.LayaTestJob.instance().running,
        "tests": read_tests(),
        "jev": {
            "provider": provider,
            **key_state(provider),
            "consent": retrieval.sufficiency_jev_consent is True,
            "rerank": rerank,
            "ladder": ladder_state(provider),
        },
        "apple_silicon": sufficiency.laya_supported(),
        "calibration_status": calibration_status(active, rerank["active"]),
    }


def ladder_state(provider: str) -> dict:
    """The background judges' consent (``judge_ladder_hosted``) and whether they
    would use it now: only with ``judge.ladder`` on (W2, D-20, D-27)."""
    from superlocalmemory.core import consent_store
    from superlocalmemory.core.feature_flags import FeatureFlags
    from superlocalmemory.infra.data_root import canonical_data_root

    home = canonical_data_root()
    now = consent_store.state(home, "judge_ladder_hosted")
    try:
        flag = FeatureFlags(home).state("judge.ladder").value
    except Exception:  # noqa: BLE001 - an unreadable flag is shown as off
        flag = "off"
    return {"consent": bool(now["given"]) and now["provider"] in (None, provider),
            "since": now["at"], "flag": flag,
            "notice": consent_store.notice("judge_ladder_hosted", provider)}


__all__ = [
    "KEY_PROBLEM",
    "MUTATION_LOCK",
    "NOT_CONFIRMED",
    "active_backend",
    "adopt_status",
    "attach",
    "build_status",
    "calibration_status",
    "detach_laya",
    "effective_retrieval",
    "enforce",
    "forget_managed_install",
    "forget_test",
    "read_tests",
    "record_test",
    "key_state",
    "ladder_state",
    "laya_paths",
    "live_engine",
    "live_judge",
    "save",
    "state_dir",
    "without_rerank",
]
