# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Central component / dependency capability registry (v3.8.2).

Single source of truth for "what SLM needs, and whether it is present" —
consumed by the self-heal thread (``server.unified_daemon._self_heal``),
``slm doctor``, the ``GET /api/v3/components`` route, and the dashboard
System-Health panel.

Design principles
-----------------
* **Cheap, side-effect-free probes.** Probing NEVER loads torch into this
  process and NEVER downloads anything: it uses import-spec checks, a
  HuggingFace-cache lookup, and a short Ollama HTTP ping. Repair (model
  download / pip install) is the self-heal thread's job — see
  ``core.component_healer`` and ``unified_daemon._self_heal`` Step 0/0.5.
* **Honest, actionable reporting.** Every component carries a status plus,
  when missing, a plain-language fix command and whether SLM can auto-fix
  it (``auto_fixable``). A non-technical user reads the report; the daemon
  acts on ``auto_fixable`` items on their behalf.
* **Live repair visibility.** A tiny thread-safe transient overlay lets the
  self-heal thread mark a component ``retrying`` while a download/install is
  in flight, so the dashboard shows progress. Overlays are advisory and are
  cleared as soon as a fresh probe confirms the component is present.
* **Immutability.** :class:`Component` is a frozen dataclass; snapshots
  return freshly-built lists/dicts and are never mutated in place.
"""

from __future__ import annotations

import importlib.util
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------
# Status + category vocabulary
# --------------------------------------------------------------------------

STATUS_OK = "ok"            # present and usable
STATUS_MISSING = "missing"  # absent — recall/feature degraded until fixed
STATUS_DEGRADED = "degraded"  # present but not fully functional / unverifiable
STATUS_RETRYING = "retrying"  # self-heal is actively downloading/installing

CATEGORY_REQUIRED = "required"        # core recall breaks without it
CATEGORY_RECOMMENDED = "recommended"  # semantic quality degrades without it
CATEGORY_OPTIONAL = "optional"        # only used by an opt-in feature

# Model repo ids — kept in sync with cli.setup_wizard (single definition here
# would be ideal, but importing the wizard pulls its CLI surface; these three
# strings are stable and asserted equal by tests).
_EMBED_MODEL = "nomic-ai/nomic-embed-text-v1.5"
_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"
_COMPRESSOR_MODEL = "microsoft/llmlingua-2-xlm-roberta-large-meetingbank"


@dataclass(frozen=True)
class Component:
    """One capability SLM depends on, and its current health.

    Frozen: probes build new instances; the transient overlay uses
    :func:`dataclasses.replace` to derive a modified copy rather than
    mutating shared state.
    """

    key: str
    label: str
    category: str
    status: str
    detail: str = ""
    fix_cmd: str = ""
    auto_fixable: bool = False
    last_checked: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "category": self.category,
            "status": self.status,
            "detail": self.detail,
            "fix_cmd": self.fix_cmd,
            "auto_fixable": self.auto_fixable,
            "last_checked": self.last_checked,
        }


# --------------------------------------------------------------------------
# Environment helpers
# --------------------------------------------------------------------------

def pip_is_user_writable() -> bool:
    """True when a plain ``pip install`` into this interpreter is expected to
    succeed — i.e. the interpreter is NOT PEP-668 externally-managed.

    Mirrors the doctor's PEP-668 check. Used to decide whether SLM may
    auto-``pip install`` a pure-python component (sqlite-vec) on the user's
    behalf, or must instead hand them a pipx/uv fix command.
    """
    try:
        import sysconfig

        stdlib = sysconfig.get_path("stdlib")
        if not stdlib:
            return True
        return not (Path(stdlib) / "EXTERNALLY-MANAGED").exists()
    except Exception:
        # Unknown → be conservative and do NOT auto-install.
        return False


def _module_present(module: str) -> bool:
    """True if ``module`` can be imported, WITHOUT importing it.

    ``find_spec`` only resolves the loader; it never executes the module, so
    this stays cheap and never pulls torch/lancedb/etc. into the daemon.
    """
    try:
        return importlib.util.find_spec(module) is not None
    except Exception:
        # A broken/partial install can raise inside find_spec — treat as absent.
        return False


def _hf_model_cached(repo_id: str) -> bool | None:
    """Heuristic: is ``repo_id`` already in the local HuggingFace cache?

    Returns True/False, or None when it cannot be determined (huggingface_hub
    not installed). Checks for the model's ``config.json`` — present for every
    HF model — without loading torch/sentence-transformers. This is a fast
    presence heuristic; the authoritative load-test lives in the healer's real
    download call, which no-ops when the model is genuinely present.
    """
    try:
        from huggingface_hub import try_to_load_from_cache
    except Exception:
        return None
    try:
        hit = try_to_load_from_cache(repo_id, "config.json")
    except Exception:
        return None
    # try_to_load_from_cache returns a filesystem path (str) only when the file
    # is cached; None when unknown, or a _CACHED_NO_EXIST sentinel object when
    # known-absent. Testing for str is version-proof across huggingface_hub
    # releases (the sentinel's import location has moved between versions).
    return isinstance(hit, str)


def _embedding_is_remote(config: Any) -> bool:
    """True when embeddings come from a remote endpoint (no local model needed).

    Delegates to the wizard's canonical detector; conservative False on any
    import/attribute error so a local embedder is still probed.
    """
    try:
        from superlocalmemory.cli.setup_wizard import _embedding_is_remote as _r

        return bool(_r(config))
    except Exception:
        return False


# --------------------------------------------------------------------------
# Individual probes — each returns a Component (never raises)
# --------------------------------------------------------------------------

def probe_python() -> Component:
    import sys

    v = sys.version_info
    ok = v >= (3, 11)
    return Component(
        key="python",
        label="Python runtime",
        category=CATEGORY_REQUIRED,
        status=STATUS_OK if ok else STATUS_MISSING,
        detail=f"{v.major}.{v.minor}.{v.micro}" + ("" if ok else " (need >= 3.11)"),
        fix_cmd="" if ok else "Install Python 3.11+ from https://python.org/downloads/",
        auto_fixable=False,
        last_checked=time.time(),
    )


def probe_search_deps() -> Component:
    """sentence-transformers / torch / scikit-learn — semantic recall channel."""
    missing = [m for m in ("sentence_transformers", "torch", "sklearn")
               if not _module_present(m)]
    ok = not missing
    return Component(
        key="search_deps",
        label="Semantic search dependencies",
        category=CATEGORY_RECOMMENDED,
        status=STATUS_OK if ok else STATUS_MISSING,
        detail="sentence-transformers, torch, scikit-learn"
        if ok else "missing: " + ", ".join(missing),
        fix_cmd="" if ok else "pip install 'superlocalmemory[search]'",
        # NOT background-auto-installed: torch is ~2GB — a silent multi-GB
        # download is a surprise the user must consent to. Fix command guides
        # them; the daemon does not pull it unasked.
        auto_fixable=False,
        last_checked=time.time(),
    )


def _probe_hf_model(key: str, label: str, repo_id: str, *,
                    category: str, auto_fixable: bool,
                    fix_cmd: str) -> Component:
    cached = _hf_model_cached(repo_id)
    if cached is True:
        status, detail, fix = STATUS_OK, repo_id, ""
    elif cached is False:
        status, detail, fix = STATUS_MISSING, f"{repo_id} not in local cache", fix_cmd
    else:
        # huggingface_hub unavailable → can't verify; report degraded, not a
        # false "missing" alarm.
        status, detail, fix = (
            STATUS_DEGRADED, "cannot verify (huggingface_hub unavailable)", "",
        )
    # Downloading model weights only helps when the runtime that loads them
    # (sentence-transformers/torch) is already present. If search deps are
    # missing, the fix is to install those first — not to pull weights.
    can_auto = (
        auto_fixable
        and status == STATUS_MISSING
        and _module_present("sentence_transformers")
    )
    return Component(
        key=key, label=label, category=category,
        status=status, detail=detail, fix_cmd=fix,
        auto_fixable=can_auto,
        last_checked=time.time(),
    )


def probe_embedder_model(config: Any = None) -> Component:
    if config is not None and _embedding_is_remote(config):
        return Component(
            key="embedder_model", label="Embedding model",
            category=CATEGORY_REQUIRED, status=STATUS_OK,
            detail="remote embedding endpoint (no local model required)",
            last_checked=time.time(),
        )
    return _probe_hf_model(
        "embedder_model", "Embedding model", _EMBED_MODEL,
        category=CATEGORY_REQUIRED, auto_fixable=True,
        fix_cmd="slm doctor --fix   (or: slm warmup)",
    )


def probe_reranker_model(config: Any = None) -> Component:
    # Only auto-heal / flag the reranker when the user actually has it enabled.
    enabled = True
    try:
        enabled = bool(getattr(config.retrieval, "use_cross_encoder", True)) \
            if config is not None else True
    except Exception:
        enabled = True
    comp = _probe_hf_model(
        "reranker_model", "Reranker model", _RERANKER_MODEL,
        category=CATEGORY_RECOMMENDED if enabled else CATEGORY_OPTIONAL,
        auto_fixable=enabled,
        fix_cmd="slm doctor --fix",
    )
    if not enabled and comp.status == STATUS_MISSING:
        # Not enabled → absent is expected, not a problem.
        return replace(comp, status=STATUS_OK,
                       detail="disabled (retrieval.use_cross_encoder=false)",
                       fix_cmd="", auto_fixable=False)
    return comp


def probe_compressor_model() -> Component:
    """LLMLingua-2 prose compressor (~560MB).

    NEVER auto-downloaded by self-heal: it is large and only used by opt-in
    aggressive optimize compression, where it lazy-downloads on first real
    use. Reported as optional so the dashboard shows it without alarm.
    """
    comp = _probe_hf_model(
        "compressor_model", "Prose compression model", _COMPRESSOR_MODEL,
        category=CATEGORY_OPTIONAL, auto_fixable=False,
        fix_cmd="Downloads automatically on first use of aggressive compression",
    )
    if comp.status == STATUS_MISSING:
        return replace(comp, detail=comp.detail + " (lazy — downloads on first use)")
    return comp


def probe_sqlite_vec() -> Component:
    present = _module_present("sqlite_vec")
    writable = pip_is_user_writable()
    if present:
        status, detail, fix, auto = STATUS_OK, "sqlite-vec installed", "", False
    elif writable:
        status, detail, fix, auto = (
            STATUS_MISSING, "sqlite-vec not installed (vector index)",
            "slm doctor --fix   (or: pip install sqlite-vec)", True,
        )
    else:
        status, detail, fix, auto = (
            STATUS_MISSING, "sqlite-vec not installed; interpreter externally managed",
            "pipx inject superlocalmemory sqlite-vec   (or use a uv/venv install)",
            False,
        )
    return Component(
        key="sqlite_vec", label="Vector index (sqlite-vec)",
        category=CATEGORY_RECOMMENDED, status=status, detail=detail,
        fix_cmd=fix, auto_fixable=auto, last_checked=time.time(),
    )


def _probe_optional_pkg(key: str, label: str, module: str, pip_name: str,
                        note: str) -> Component:
    present = _module_present(module)
    return Component(
        key=key, label=label, category=CATEGORY_OPTIONAL,
        status=STATUS_OK if present else STATUS_MISSING,
        detail=f"{pip_name} installed" if present else f"{note}",
        fix_cmd="" if present else f"pip install {pip_name}",
        auto_fixable=False,  # opt-in scale/feature backends — staged consent.
        last_checked=time.time(),
    )


def probe_lancedb() -> Component:
    return _probe_optional_pkg(
        "lancedb", "LanceDB scale backend", "lancedb", "lancedb",
        "not installed (optional large-scale vector backend)",
    )


def probe_cozo() -> Component:
    return _probe_optional_pkg(
        "cozo", "CozoDB graph backend", "pycozo", "pycozo",
        "not installed (optional large-scale graph backend)",
    )


def probe_llmlingua() -> Component:
    return _probe_optional_pkg(
        "llmlingua", "LLMLingua compressor lib", "llmlingua", "llmlingua",
        "not installed (optional prose compression)",
    )


def probe_ollama(config: Any = None) -> Component:
    """Ollama reachability — only meaningful in Mode B.

    Never auto-fixed: installing the Ollama binary or pulling a model is a
    surprise network/disk action, so it is always a manual fix command.
    """
    mode = ""
    api_base = "http://localhost:11434"
    model = ""
    try:
        if config is not None:
            mode = getattr(getattr(config, "mode", None), "value", "") or ""
            api_base = getattr(getattr(config, "llm", None), "api_base", api_base) or api_base
            model = getattr(getattr(config, "llm", None), "model", "") or ""
    except Exception:
        pass
    if mode != "b":
        return Component(
            key="ollama", label="Ollama (local LLM)",
            category=CATEGORY_OPTIONAL, status=STATUS_OK,
            detail="not in use (Mode B only)", last_checked=time.time(),
        )
    try:
        import httpx

        resp = httpx.get(f"{api_base}/api/tags", timeout=5.0)
        if resp.status_code == 200:
            names = [m.get("name", "").split(":")[0]
                     for m in resp.json().get("models", [])]
            if model and model.split(":")[0] not in names:
                return Component(
                    key="ollama", label="Ollama (local LLM)",
                    category=CATEGORY_RECOMMENDED, status=STATUS_DEGRADED,
                    detail=f"running but '{model}' not pulled",
                    fix_cmd=f"ollama pull {model}", auto_fixable=False,
                    last_checked=time.time(),
                )
            return Component(
                key="ollama", label="Ollama (local LLM)",
                category=CATEGORY_RECOMMENDED, status=STATUS_OK,
                detail=f"running, {len(names)} models", last_checked=time.time(),
            )
        status_detail = f"HTTP {resp.status_code}"
    except Exception:
        status_detail = f"not reachable at {api_base}"
    return Component(
        key="ollama", label="Ollama (local LLM)",
        category=CATEGORY_RECOMMENDED, status=STATUS_MISSING,
        detail=status_detail,
        fix_cmd="Install/start Ollama: https://ollama.com  then: brew services start ollama",
        auto_fixable=False, last_checked=time.time(),
    )


# --------------------------------------------------------------------------
# Transient overlay — live self-heal progress
# --------------------------------------------------------------------------

_transient_lock = threading.Lock()
_transient: dict[str, tuple[str, str]] = {}  # key -> (status, detail)


def mark_transient(key: str, status: str, detail: str = "") -> None:
    """Mark a component's live repair state (e.g. RETRYING while downloading)."""
    with _transient_lock:
        _transient[key] = (status, detail)


def clear_transient(key: str) -> None:
    with _transient_lock:
        _transient.pop(key, None)


def _apply_transient(comp: Component) -> Component:
    with _transient_lock:
        override = _transient.get(comp.key)
    if override is None:
        return comp
    status, detail = override
    # A confirmed-present component supersedes a stale "retrying" marker.
    if comp.status == STATUS_OK:
        return comp
    return replace(comp, status=status, detail=detail or comp.detail)


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------

def probe_all(config: Any = None) -> list[Component]:
    """Probe every component once and return a fresh list (transient overlaid)."""
    probes = [
        probe_python(),
        probe_search_deps(),
        probe_embedder_model(config),
        probe_reranker_model(config),
        probe_compressor_model(),
        probe_sqlite_vec(),
        probe_llmlingua(),
        probe_lancedb(),
        probe_cozo(),
        probe_ollama(config),
    ]
    return [_apply_transient(c) for c in probes]


def auto_fixable_missing(config: Any = None) -> list[Component]:
    """The subset SLM may repair without asking — drives self-heal + doctor --fix."""
    return [c for c in probe_all(config)
            if c.status == STATUS_MISSING and c.auto_fixable]


def snapshot(config: Any = None) -> dict[str, Any]:
    """Full registry snapshot for /api/v3/components, doctor --json, dashboard."""
    comps = probe_all(config)
    counts = {STATUS_OK: 0, STATUS_MISSING: 0,
              STATUS_DEGRADED: 0, STATUS_RETRYING: 0}
    for c in comps:
        counts[c.status] = counts.get(c.status, 0) + 1
    missing_needed = [c for c in comps
                      if c.status == STATUS_MISSING
                      and c.category in (CATEGORY_REQUIRED, CATEGORY_RECOMMENDED)]
    healthy = not missing_needed and counts[STATUS_RETRYING] == 0
    return {
        "components": [c.as_dict() for c in comps],
        "summary": {
            "ok": counts[STATUS_OK],
            "missing": counts[STATUS_MISSING],
            "degraded": counts[STATUS_DEGRADED],
            "retrying": counts[STATUS_RETRYING],
            "auto_fixable_missing": sum(
                1 for c in comps
                if c.status == STATUS_MISSING and c.auto_fixable
            ),
            "healthy": healthy,
        },
        "generated_at": time.time(),
    }
