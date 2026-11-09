# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Which answer check runs, and the one place it is started, switched and stopped.

The rule this module exists to keep: exactly one answer check runs. Switch on
the hosted one (Jev) and the local one (Laya) does not run; switch on Laya and
Jev does not run. Two at once would mean two models resident, two verdicts
that disagree, and memory text leaving the machine while the person believes
it stays local.

"auto" means Laya only when an install was set up and passed its check (the
dashboard or setup path, ``laya_runtime.detect`` reporting it ready) —
otherwise nothing. A ``laya_mlx`` that merely happens to be importable from
SLM's own environment (``pip install "superlocalmemory[full]"``) does not turn
it on; choosing "laya" by name does. "auto" never means Jev: the question and
the top memories leave the machine only when someone chose that, stored a key
and accepted it.

Weights named by repo id are pinned to the revision the threshold was measured
on; when that revision is not in the cache, the check stays off rather than
load whatever revision happens to be there.

Reordering the results is an option of the hosted check only, with its own
consent (``jev_rerank_k``). Laya is never built to reorder, and "auto" never
reaches the branch that could.

One model per process, not just per engine. The daemon's hot reconfigure
builds the new engine before it closes the old one, so two engines briefly
coexist. Through here they share one Laya worker (and the old engine's close
does not stop it), and starting a different check stops the running one
first — before the new one is even constructed.

A switch is process-wide. It may target the engine that is still published
while a reconfigure is building the next one; every engine carrying the
running check (``register_engine``) gets the new one — or none, when the
switch is to off — so no engine is ever left holding a stopped check while
the process runs none.

Nothing here runs on the recall path. A recall reads the engine's judge once;
switching takes it off the engine before stopping it, so a recall that read
the old judge gets None from it, never an exception.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
import weakref
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

MODE_OFF = "off"
MODE_AUTO = "auto"
MODE_LAYA = "laya"
MODE_JEV = "jev"

_JUDGE_ATTR = "_sufficiency_judge"
#: Set on a retrieval engine when its owner closes it. A switch that arrives
#: afterwards must not start a model nobody will ever stop.
_CLOSED_ATTR = "_sufficiency_judge_closed"

_DEFAULT_TIMEOUT_S = 1.5
_DEFAULT_JEV_TIMEOUT_S = 2.0

#: Serialises every start, switch and stop. Re-entrant because a switch calls
#: the builder, which takes it again. The recall path never takes it.
_LOCK = threading.RLock()


@dataclass(frozen=True)
class _LayaPlan:
    python: str
    model: str
    hf_home: str
    timeout_s: float

    @property
    def key(self) -> tuple:
        return (MODE_LAYA, self.python, self.model, self.hf_home, self.timeout_s)


@dataclass(frozen=True)
class _Live:
    """The one answer check this process runs, and how many engines hold it."""

    judge: Any
    key: tuple | None   # None = never shared (a hosted check is cheap to rebuild)
    holders: int
    #: References to the engines carrying it, so a switch reaches all of them.
    carriers: tuple = ()


_live: _Live | None = None
#: One warning per process for a malformed reorder count, not one per recall.
_rerank_k_warned = False


# -- small readers -----------------------------------------------------------

def _mode(retrieval_config: Any) -> str:
    raw = getattr(retrieval_config, "sufficiency_judge", MODE_OFF) or MODE_OFF
    return str(raw).strip().lower()


def _positive_float(value: object, default: float) -> float:
    """A hand-edited config must not stop the daemon from starting."""
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return number if number > 0 else default


def backend_of(judge: Any) -> str:
    """"laya", "jev", or "off" when there is no judge."""
    if judge is None:
        return MODE_OFF
    backend = getattr(judge, "backend", "")
    return backend if isinstance(backend, str) and backend else "unknown"


def interpreter_has_module(python: str, module: str) -> bool:
    """Whether ``python`` can locate ``module``, found without importing it.

    A daemon start must not pay for loading MLX just to learn it is there.
    """
    try:
        done = subprocess.run(
            [python, "-c",
             "import importlib.util, sys; "
             f"sys.exit(0 if importlib.util.find_spec({module!r}) else 1)"],
            capture_output=True, timeout=15, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return done.returncode == 0


# -- where Laya is -----------------------------------------------------------

def _managed_runtime(retrieval_config: Any) -> Any | None:
    """The SLM-managed Laya install when it is ready, else None. Never raises."""
    try:
        from superlocalmemory.core import laya_runtime

        status = laya_runtime.detect(retrieval_config)
    except NotImplementedError:
        return None
    except Exception as exc:  # noqa: BLE001 — a broken check is "not installed"
        logger.warning("Laya runtime check failed (%s); treating Laya as not installed",
                       type(exc).__name__)
        return None
    if getattr(status, "state", "") != laya_runtime.STATE_READY:
        return None
    python = getattr(status, "python", "")
    return status if isinstance(python, str) and python else None


def find_laya(retrieval_config: Any) -> _LayaPlan | None:
    """Interpreter, weights and cache for Laya, or None when it cannot run here.

    "auto": only an install that was set up and passed its check.
    "laya" (chosen by name), in order: an interpreter someone named; else the
    checked install; else SLM's own interpreter, if it has laya_mlx. A named
    interpreter that lacks laya_mlx is reported and NOT replaced by another
    one — running something other than what was configured would hide the
    broken setting. Weights named by repo id load the measured revision only.
    """
    from superlocalmemory.retrieval import sufficiency

    if not sufficiency.laya_supported():
        logger.info("Answer check (Laya) off: it needs Apple Silicon")
        return None
    model = str(getattr(retrieval_config, "sufficiency_model", "") or sufficiency.DEFAULT_MODEL)
    hf_home = str(getattr(retrieval_config, "sufficiency_hf_home", "") or "")
    timeout_s = _positive_float(getattr(retrieval_config, "sufficiency_timeout_s", None),
                                _DEFAULT_TIMEOUT_S)
    if _mode(retrieval_config) == MODE_AUTO:
        return _checked_install_plan(retrieval_config, model, hf_home, timeout_s)
    explicit = str(getattr(retrieval_config, "sufficiency_python", "") or "").strip()
    if explicit:
        if _refused(explicit):
            return None
        if interpreter_has_module(explicit, "laya_mlx"):
            return _pinned_plan(explicit, model, hf_home, timeout_s)
        logger.info("Answer check (Laya) off: %s cannot find laya_mlx "
                    "(fix retrieval.sufficiency_python, or clear it)", explicit)
        return None
    plan = _checked_install_plan(retrieval_config, model, hf_home, timeout_s, quiet=True)
    if plan is not None:
        return plan
    if not _refused(sys.executable) and interpreter_has_module(sys.executable, "laya_mlx"):
        return _pinned_plan(sys.executable, model, hf_home, timeout_s)
    logger.info("Answer check (Laya) off: Laya is not installed")
    return None


def _refused(python: str) -> bool:
    """An interpreter other accounts could change is never run — not even to
    ask it whether laya_mlx is there (``laya_runtime.check_interpreter``)."""
    from superlocalmemory.core import laya_runtime

    reason = laya_runtime.check_interpreter(python, strict_location=False)
    if reason:
        logger.info("Answer check (Laya) off: %s", reason)
    return bool(reason)


def _checked_install_plan(retrieval_config: Any, model: str, hf_home: str,
                          timeout_s: float, *, quiet: bool = False) -> _LayaPlan | None:
    managed = _managed_runtime(retrieval_config)
    if managed is None:
        if not quiet:
            logger.info("Answer check (Laya) off: \"auto\" turns it on once it is set up "
                        "and checked (Settings, Answer check)")
        return None
    if managed.model_path:
        return _LayaPlan(managed.python, managed.model_path,
                         managed.hf_home or hf_home, timeout_s)
    return _pinned_plan(managed.python, model, managed.hf_home or hf_home, timeout_s)


def _pinned_plan(python: str, model: str, hf_home: str, timeout_s: float) -> _LayaPlan | None:
    weights = pinned_weights(model, hf_home)
    return None if weights is None else _LayaPlan(python, weights, hf_home, timeout_s)


def pinned_weights(model: str, hf_home: str) -> str | None:
    """The weights to load: a folder as named, or the measured snapshot of the
    repo id. None when a repo id cannot be pinned — loading it by name would
    load whatever revision is cached, under a threshold measured on another."""
    from superlocalmemory.core.laya_runtime import LAYA_MODEL_REPO, LAYA_MODEL_REVISION

    text = str(model or "").strip()
    # A folder: absolute on this OS (C:\... on Windows), or ~ / . relative.
    if os.path.isabs(text) or text.startswith(("/", "~", ".")):
        return text
    if text != LAYA_MODEL_REPO:
        logger.info("Answer check (Laya) off: %r is named by repo id, which loads whatever "
                    "revision is cached; name its snapshot folder instead", text[:80])
        return None
    folder = "models--" + text.replace("/", "--")
    for hub in _hub_dirs(hf_home):
        snapshot = hub / folder / "snapshots" / LAYA_MODEL_REVISION
        if snapshot.is_dir():
            return str(snapshot)
    logger.info("Answer check (Laya) off: the measured weights (%s@%s) are not in the "
                "model cache", text, LAYA_MODEL_REVISION[:12])
    return None


def _hub_dirs(hf_home: str) -> list[Path]:
    """Where a Hugging Face cache keeps repos: <HF_HOME>/hub, or the folder itself
    when it was used as the hub directory."""
    if hf_home:
        base = Path(hf_home).expanduser()
        return [base / "hub", base]
    if os.environ.get("HF_HUB_CACHE"):
        return [Path(os.environ["HF_HUB_CACHE"]).expanduser()]
    home = os.environ.get("HF_HOME")
    base = Path(home).expanduser() if home else Path.home() / ".cache" / "huggingface"
    return [base / "hub"]


# -- whether Jev may run -----------------------------------------------------

def _jev_target(retrieval_config: Any) -> tuple[str, Any] | None:
    """(provider, key store) when the hosted check may run, else None.

    Consent must be the boolean True: a hand-edited "true" is not someone
    accepting that memory text leaves the machine.
    """
    if getattr(retrieval_config, "sufficiency_jev_consent", False) is not True:
        logger.info("Answer check (Jev) off: sending memories to the provider was not accepted")
        return None
    try:
        from superlocalmemory.core import judge_keys

        provider = str(getattr(retrieval_config, "sufficiency_jev_provider", "") or "")
        provider = provider.strip().lower()
        if provider not in judge_keys.PROVIDERS:
            logger.info("Answer check (Jev) off: unknown provider %r", provider[:40])
            return None
        store = judge_keys.JudgeKeyStore()
        if store.has_key(provider) is not True:
            logger.info("Answer check (Jev) off: no key stored for %s", provider)
            return None
    except NotImplementedError:
        logger.info("Answer check (Jev) off: key storage is not available in this build")
        return None
    except Exception as exc:  # noqa: BLE001 — never log the message: it may name a key
        logger.warning("Answer check (Jev) off: key check failed (%s)", type(exc).__name__)
        return None
    return provider, store


def jev_rerank_k(retrieval_config: Any) -> int:
    """How many memories the hosted check may reorder per recall; 0 = it may not.

    Both the toggle and its own consent must be the boolean True: it sends more
    memories than the answer check alone, so the answer check's consent does
    not cover it. Reordering is part of Jev: while any other option is chosen
    it is 0, so nothing is sent for it — the person's choice is kept and
    counts again once Jev is chosen. Whether Jev itself may run is
    ``_jev_target``; Laya never reads this.
    """
    global _rerank_k_warned
    if _mode(retrieval_config) != MODE_JEV:
        return 0
    on = getattr(retrieval_config, "sufficiency_jev_rerank", False)
    consent = getattr(retrieval_config, "sufficiency_jev_rerank_consent", False)
    if on is not True or consent is not True:
        return 0
    from superlocalmemory.retrieval.jev_rerank import parse_rerank_k

    k = parse_rerank_k(getattr(retrieval_config, "sufficiency_jev_rerank_k", None))
    if k is None:
        # Malformed means OFF: the failure must never be "send more memories".
        # The key is named; its value is not (it is whatever was typed there).
        if not _rerank_k_warned:
            logger.warning("Answer check reordering off: retrieval.sufficiency_jev_rerank_k "
                           "is not a whole number of at least 1")
            _rerank_k_warned = True
        return 0
    _rerank_k_warned = False
    return k


# -- the one live judge ------------------------------------------------------

def _stop(judge: Any) -> None:
    """Stop ``judge`` and forget every verdict it gave (one judge, one memo):
    a switch of provider, a setting or Off never lets an old verdict answer."""
    try:
        shutdown = getattr(judge, "shutdown", None)
        if callable(shutdown):
            shutdown()
    except Exception as exc:  # noqa: BLE001 — a stuck judge must not block a switch
        logger.warning("Answer check (%s) did not stop cleanly: %s",
                       backend_of(judge), type(exc).__name__)
    finally:
        from superlocalmemory.core import answer_check_memo

        answer_check_memo.clear(judge)


def _ref(engine: Any) -> Callable[[], Any]:
    try:
        return weakref.ref(engine)
    except TypeError:  # an object that cannot be weakly referenced (test doubles)
        return lambda: engine


def _carrying(live: _Live) -> list:
    """The engines that still exist and still carry this judge."""
    engines = (ref() for ref in live.carriers)
    return [e for e in engines if e is not None and getattr(e, _JUDGE_ATTR, None) is live.judge]


def _plus_carrier(live: _Live, engine: Any) -> _Live:
    if any(ref() is engine for ref in live.carriers):
        return live
    return replace(live, carriers=live.carriers + (_ref(engine),))


def _minus_carrier(live: _Live, engine: Any) -> _Live:
    return replace(live, carriers=tuple(r for r in live.carriers
                                        if r() is not None and r() is not engine))


def _acquire(key: tuple | None, factory: Callable[[], Any]) -> Any:
    """The process's judge for ``key``: shared when the same Laya is already
    running, otherwise built — after whatever was running has been stopped.
    Every engine that carried the stopped one is handed the new one."""
    global _live
    with _LOCK:
        live = _live
        if (live is not None and key is not None and live.key == key
                and getattr(live.judge, "closed", False) is not True):
            _live = replace(live, holders=live.holders + 1)
            return live.judge
        carriers: list = []
        if live is not None:
            _live = None
            carriers = _carrying(live)
            for engine in carriers:
                setattr(engine, _JUDGE_ATTR, None)
            logger.info("Answer check: stopping %s before starting another",
                        backend_of(live.judge))
            _stop(live.judge)
        judge = factory()
        if judge is not None:
            moved = [e for e in carriers if getattr(e, _CLOSED_ATTR, False) is not True]
            for engine in moved:
                setattr(engine, _JUDGE_ATTR, judge)
            _live = _Live(judge, key, 1 + len(moved), tuple(_ref(e) for e in moved))
        return judge


def _release(judge: Any, engine: Any = None) -> None:
    """One holder lets go. The model stops when the last one does; a judge
    that never came through here is simply stopped."""
    global _live
    with _LOCK:
        live = _live
        if live is not None and live.judge is judge:
            if engine is not None:
                live = _minus_carrier(live, engine)
            # Another engine still carrying it keeps it running, whatever the
            # count says: stopping it would leave that engine a stopped check.
            if live.holders > 1 or _carrying(live):
                _live = replace(live, holders=max(1, live.holders - 1))
                return
            _live = None
        _stop(judge)


def _stop_everywhere() -> None:
    """The process's check is now off: off every engine carrying it, then stopped."""
    global _live
    live = _live
    if live is None:
        return
    _live = None
    for engine in _carrying(live):
        setattr(engine, _JUDGE_ATTR, None)
    _stop(live.judge)


def _construct_laya(plan: _LayaPlan) -> Any:
    from superlocalmemory.retrieval import sufficiency

    try:
        # Built idle: the model loads on the first recall that asks (or when the
        # daemon warms it up at start), never because a one-shot command
        # happened to build an engine.
        return sufficiency.LayaSufficiencyJudge(
            python=plan.python, model=plan.model, hf_home=plan.hf_home,
            timeout_s=plan.timeout_s, start=False)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Answer check (Laya) off: it could not start (%s)", type(exc).__name__)
        return None


def _construct_jev(provider: str, store: Any, timeout_s: float, rerank_k: int) -> Any:
    try:
        from superlocalmemory.infra.data_root import canonical_data_root
        from superlocalmemory.retrieval import jev_judge

        # The stored choice is re-read before every request, so consent
        # withdrawn in any process stops this one too.
        return jev_judge.JevSufficiencyJudge(provider=provider, key_store=store,
                                             timeout_s=timeout_s, rerank_k=rerank_k,
                                             state_dir=canonical_data_root())
    except NotImplementedError:
        logger.info("Answer check (Jev) off: the hosted check is not available in this build")
    except Exception as exc:  # noqa: BLE001 — never log the message: it may name a key
        logger.warning("Answer check (Jev) off: the hosted check could not start (%s)",
                       type(exc).__name__)
    return None


# -- public ------------------------------------------------------------------

def resolve_judge_mode(retrieval_config: Any) -> str:
    """What would run: "laya", "jev" or "off". Detects; starts nothing."""
    mode = _mode(retrieval_config)
    if mode in (MODE_LAYA, MODE_AUTO):
        return MODE_LAYA if find_laya(retrieval_config) is not None else MODE_OFF
    if mode == MODE_JEV:
        return MODE_JEV if _jev_target(retrieval_config) is not None else MODE_OFF
    return MODE_OFF


def build_sufficiency_judge(retrieval_config: Any) -> Any:
    """The judge the config asks for, or None — with one log line saying why."""
    mode = _mode(retrieval_config)
    if mode == MODE_OFF:
        return None
    if mode in (MODE_LAYA, MODE_AUTO):  # "auto" never reaches the Jev branch
        plan = find_laya(retrieval_config)
        if plan is None:
            return None
        return _acquire(plan.key, lambda: _construct_laya(plan))
    if mode == MODE_JEV:
        target = _jev_target(retrieval_config)
        if target is None:
            return None
        timeout_s = _positive_float(getattr(retrieval_config, "sufficiency_jev_timeout_s", None),
                                    _DEFAULT_JEV_TIMEOUT_S)
        rerank_k = jev_rerank_k(retrieval_config)
        return _acquire(None, lambda: _construct_jev(target[0], target[1], timeout_s, rerank_k))
    logger.warning("Answer check off: retrieval.sufficiency_judge=%r is not one of "
                   "off, auto, laya, jev", mode[:40])
    return None


def swap_sufficiency_judge(retrieval_engine: Any, build: Callable[[], Any]) -> str:
    """Replace the engine's judge with ``build()``. Returns the backend now active.

    Order matters and is the point: off the engine first (a recall from here on
    sees no judge), then stopped, then the new one built and attached. Two
    models are never alive at once, and an engine already closed gets nothing.
    The switch is process-wide: every engine carrying the old check gets the
    new one, or none when the switch is to off.
    """
    with _LOCK:
        if getattr(retrieval_engine, _CLOSED_ATTR, False) is True:
            return MODE_OFF
        old = getattr(retrieval_engine, _JUDGE_ATTR, None)
        setattr(retrieval_engine, _JUDGE_ATTR, None)
        if old is not None:
            _release(old, retrieval_engine)
        try:
            new = build()
        except Exception as exc:  # noqa: BLE001 — a failed switch leaves the check off
            logger.warning("Answer check off: switching failed (%s)", type(exc).__name__)
            new = None
        if new is None:
            _stop_everywhere()
        else:
            setattr(retrieval_engine, _JUDGE_ATTR, new)
            _carry(retrieval_engine, new)
            _start_loading(new)
        return backend_of(new)


def _start_loading(judge: Any) -> None:
    """A switch runs in the long-running daemon: load the new check now, so the
    next recall can be judged (it is built idle for one-shot commands)."""
    start = getattr(judge, "start_warmup", None)
    if callable(start):
        try:
            start()
        except Exception as exc:  # noqa: BLE001 — loading later is the fallback
            logger.debug("Answer check warm-up not started: %s", type(exc).__name__)


def stop_on_device_check() -> bool:
    """Stop the on-device check in EVERY engine that holds it. True if one ran.

    For removing the on-device install: during a hot reconfigure two engines
    can share the one Laya worker, and detaching only the published engine
    would leave the other holding a running model whose files are being
    deleted. An online check, if that is what runs, is left alone.
    """
    with _LOCK:
        live = _live
        if live is None or backend_of(live.judge) != MODE_LAYA:
            return False
        _stop_everywhere()
        return True


def _carry(engine: Any, judge: Any) -> None:
    global _live
    live = _live
    if live is not None and live.judge is judge:
        _live = _plus_carrier(live, engine)


def register_engine(retrieval_engine: Any) -> None:
    """Note that this engine carries the process's check, so a switch reaches it.

    Called once an engine has been built. A switch that ran while it was being
    built may already have stopped the judge it was given: it then gets the
    check now running, or none if the switch was to off.
    """
    global _live
    with _LOCK:
        judge = getattr(retrieval_engine, _JUDGE_ATTR, None)
        if judge is None:
            return
        live = _live
        if live is not None and live.judge is judge:
            _live = _plus_carrier(live, retrieval_engine)
            return
        if getattr(judge, "closed", False) is not True:
            return  # built outside this module: not ours to track
        if live is None or getattr(retrieval_engine, _CLOSED_ATTR, False) is True:
            setattr(retrieval_engine, _JUDGE_ATTR, None)
            return
        setattr(retrieval_engine, _JUDGE_ATTR, live.judge)
        _live = replace(_plus_carrier(live, retrieval_engine), holders=live.holders + 1)


def release_sufficiency_judge(retrieval_engine: Any, *, final: bool = False) -> None:
    """Take the engine's judge off it and let go of it. ``final`` = the engine
    is closing, so no later switch may give it another one."""
    with _LOCK:
        if final:
            setattr(retrieval_engine, _CLOSED_ATTR, True)
        old = getattr(retrieval_engine, _JUDGE_ATTR, None)
        setattr(retrieval_engine, _JUDGE_ATTR, None)
        if old is not None:
            _release(old, retrieval_engine)


__all__ = [
    "MODE_AUTO",
    "MODE_JEV",
    "MODE_LAYA",
    "MODE_OFF",
    "backend_of",
    "build_sufficiency_judge",
    "find_laya",
    "interpreter_has_module",
    "jev_rerank_k",
    "pinned_weights",
    "register_engine",
    "release_sufficiency_judge",
    "resolve_judge_mode",
    "stop_on_device_check",
    "swap_sufficiency_judge",
]
