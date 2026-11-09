# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Installs, finds, checks and removes the local answer-check model (Laya).

Laya runs in its own Python environment under ``runtime_dir()``, never inside
SLM's: SLM's dependency pins must not move at runtime, and an environment SLM
manages survives every way SLM itself is upgraded (pip, pipx, uv, npm).

Three install sources can exist at once; ``detect()`` (below) picks between
them and explains, in its own docstring, exactly when each counts as READY.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from superlocalmemory.core import laya_interpreter, laya_paths, laya_process
from superlocalmemory.infra.data_root import canonical_data_root

try:
    import fcntl
except ImportError:  # Windows: Laya is Apple-Silicon only; detect() already
    fcntl = None  # type: ignore[assignment]  # refuses before any lock is taken.

logger = logging.getLogger(__name__)

LAYA_MLX_REQUIREMENT = "laya-mlx==0.2.0"
LAYA_MODEL_REPO = "aac6fef/laya-mlx"
#: The weights the shipped threshold was measured on (807 MB).
LAYA_MODEL_REVISION = "20aed815fc6acde75733882e7ec0e3f28aeb9717"

STATE_UNSUPPORTED = "unsupported"      # not Apple Silicon
STATE_NOT_INSTALLED = "not_installed"
STATE_INSTALLING = "installing"
STATE_READY = "ready"                  # installed AND the canary check passed
STATE_FAILED = "failed"

#: What fixes a FAILED install — the dashboard offers exactly this action.
#: Advice that points at the wrong fix strands people: an install SLM did not
#: make was once told to "choose Set up again", which started an 800 MB
#: download on a network that blocks it, instead of simply checking it.
ACTION_SETUP = "setup"   # SLM's own install: Repair (set up again; it resumes)
ACTION_CHECK = "check"   # an install made elsewhere: check it; downloads nothing

_REPAIR_STEP = "Needs repair — choose Repair."
_CHECK_STEP = "Needs a check — choose Check this install."
UNCHECKED = "SLM found this install but hasn't checked it yet. Checking downloads nothing."
UNFINISHED = ("The last setup stopped before it finished. Choose Repair to finish it "
              "(what was downloaded is kept), or Remove to delete it.")
_MANAGED_MISSING = "Parts of the install are missing. Choose Repair, or Remove."
_MANAGED_UNCHECKED = "The install didn't pass its check. Choose Repair, or Remove."
_MOVED = ("The install can't be found — it may have moved. Enter where it is now "
          "under Advanced, or choose Forget.")

#: ~807 MB — see LAYA_MODEL_REVISION. Used only to estimate install progress.
_EXPECTED_DOWNLOAD_BYTES = 807 * 1024 * 1024
_EXPECTED_MB = _EXPECTED_DOWNLOAD_BYTES // (1024 * 1024)
_STALL_S = laya_process.DEFAULT_STALL_S
_MIN_FREE_BYTES = int(1.5 * 1024 ** 3)

_VENV_TIMEOUT_S = 120.0
_PIP_TIMEOUT_S = 600.0
_DOWNLOAD_TIMEOUT_S = 1800.0

_NETWORK_MESSAGE = (
    "Can't reach the model download server. If your network blocks it, choose "
    "'Use an install you already have'; otherwise choose Repair to try again."
)
_STALLED_MESSAGE = (
    "Can't reach the model download server — the download stopped making progress, so "
    "it was stopped. If your network blocks it, choose 'Use an install you already "
    "have'; otherwise choose Repair to try again."
)
_CANCELLED_MESSAGE = ("Setup was cancelled. Choose Repair to continue — what was "
                      "already downloaded is kept.")
_NETWORK_HINTS = (
    "timeout", "timed out", "connection", "network", "resolve", "unreachable",
    "ssl", "certificate", "name or service not known",
    "temporary failure in name resolution", "max retries exceeded",
    "connection refused", "connection reset", "no route to host",
)


@dataclass(frozen=True)
class LayaRuntimeStatus:
    state: str
    managed: bool = False      # True = installed by SLM under runtime_dir()
    python: str = ""
    hf_home: str = ""
    model_path: str = ""       # local snapshot folder handed to laya_mlx.load
    model_revision: str = ""
    progress: float = 0.0      # 0..1 while installing
    step: str = ""             # plain-language current step, for the dashboard
    error: str = ""            # plain-language reason; never contains secrets
    action: str = ""           # FAILED only: ACTION_SETUP or ACTION_CHECK — the fix

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


ProgressFn = Callable[[float, str], None]

WORKER_PATH = Path(__file__).resolve().parent / "laya_worker.py"


# Small stdlib-only helpers (path / json / time) — no third-party imports so
# this module stays importable even before Laya's own venv exists.

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _apple_silicon() -> bool:
    return sys.platform == "darwin" and platform.machine() == "arm64"


def runtime_dir(slm_home: Path | None = None) -> Path:
    """``<SLM_HOME>/runtimes/laya``."""
    base = Path(slm_home) if slm_home is not None else canonical_data_root()
    return base / "runtimes" / "laya"


def check_interpreter(python: str, *, strict_location: bool = True) -> str:
    """Why ``python`` must not be run for Laya, or "" when it may be.

    The rule is in core/laya_interpreter.py. ``strict_location`` (a path a
    person typed) also refuses temporary and shared folders; a path SLM saved
    earlier is re-checked without it, since its ownership checks already catch
    anyone else changing it.
    """
    moved = laya_paths.venv_refusal(Path(python).parent.parent)
    if moved:
        return moved
    return laya_interpreter.refusal(python, strict_location=strict_location,
                                    in_temp_dir=lambda path: _is_in_temp_dir(path))


# verify() lives in core/laya_verify.py; re-exported here, where callers
# (and tests that replace it) have always found it.
from superlocalmemory.core.laya_verify import VERIFY_BUSY, verify  # noqa: E402

# detect() — fast, offline, never raises.

def _record_matches(record: dict[str, Any] | None, python: str, model_path: str) -> bool:
    return (bool(record) and record.get("python") == python
            and record.get("model_path") == model_path)


def _status_from_record(record: dict[str, Any], *, managed: bool,
                          hf_home_override: str = "") -> LayaRuntimeStatus:
    if record.get("_path_error"):
        return _needs_fix(managed=True, python=str(record.get("python", "")),
                          error=str(record["_path_error"]))
    python = str(record.get("python", ""))
    model_path = str(record.get("model_path", ""))
    hf_home = hf_home_override or str(record.get("hf_home", ""))
    revision = str(record.get("revision", ""))
    verified = bool(record.get("verified"))
    files_ok = (bool(python) and Path(python).exists()
                and bool(model_path) and Path(model_path).is_dir())

    refused = check_interpreter(python, strict_location=False) if files_ok else ""
    if verified and files_ok and not refused:
        return LayaRuntimeStatus(
            state=STATE_READY, managed=managed, python=python, hf_home=hf_home,
            model_path=model_path, model_revision=revision, progress=1.0, step="Ready",
        )
    if refused:
        error = refused
    elif not files_ok:
        error = _MANAGED_MISSING if managed else _MOVED
    else:  # present, never passed its check: say why it last failed, when known
        error = str(record.get("reason") or "") or (_MANAGED_UNCHECKED if managed else UNCHECKED)
    return _needs_fix(managed=managed, python=python, hf_home=hf_home,
                      model_path=model_path, model_revision=revision, error=error)


def _needs_fix(*, managed: bool, error: str, **paths: str) -> LayaRuntimeStatus:
    """FAILED, with the one action that fixes it: SLM's own install is
    repaired; an install made elsewhere is checked (nothing is downloaded)."""
    return LayaRuntimeStatus(
        state=STATE_FAILED, managed=managed, error=error,
        step=_REPAIR_STEP if managed else _CHECK_STEP,
        action=ACTION_SETUP if managed else ACTION_CHECK, **paths)


def _inside(path: str, root: Path) -> bool:
    roots = {str(root), str(root.resolve())}
    return any(path == r or path.startswith(r + os.sep) for r in roots)


def _on_disk(retrieval_config: Any, run_dir: Path) -> LayaRuntimeStatus:
    managed_record = _read_json(run_dir / ".slm-managed")
    adopted_record = _read_json(run_dir / "adopted.json")
    if managed_record is not None:
        managed_record = laya_paths.managed_record(managed_record, run_dir)
        if managed_record.get("_path_error") and adopted_record is None:
            return _status_from_record(managed_record, managed=True)

    cfg_python = str(getattr(retrieval_config, "sufficiency_python", "") or "").strip()
    cfg_model = str(getattr(retrieval_config, "sufficiency_model", "") or "").strip()
    cfg_hf_home = str(getattr(retrieval_config, "sufficiency_hf_home", "") or "").strip()

    if cfg_python and cfg_model and Path(cfg_model).is_absolute():
        for record, managed in ((adopted_record, False), (managed_record, True)):
            if _record_matches(record, cfg_python, cfg_model):
                if managed and cfg_hf_home:
                    record = laya_paths.managed_record({**record, "hf_home": cfg_hf_home}, run_dir)
                return _status_from_record(
                    record, managed=managed, hf_home_override="" if managed else cfg_hf_home)
        managed = _inside(cfg_python, run_dir)
        return _needs_fix(managed=managed, python=cfg_python, hf_home=cfg_hf_home,
                          model_path=cfg_model, error=UNFINISHED if managed else UNCHECKED)

    if adopted_record is not None:
        return _status_from_record(adopted_record, managed=False)

    if managed_record is not None:
        return _status_from_record(managed_record, managed=True)

    if _progress_stamp_path(run_dir).exists():
        # A setup that never reached its check: the process stopped mid-way
        # (a crash, a restart, a closed lid). Resumable, so never "not installed".
        return _needs_fix(managed=True, error=UNFINISHED)

    return LayaRuntimeStatus(state=STATE_NOT_INSTALLED)


def _with_last_failure(status: LayaRuntimeStatus) -> LayaRuntimeStatus:
    """Why the last setup in this process failed, when the disk alone can't say.

    Without this, a download that failed on a blocked network came back as
    "not checked yet" — true of the files, silent about what happened.
    """
    if status.state not in (STATE_FAILED, STATE_NOT_INSTALLED):
        return status
    if status.state == STATE_FAILED and not status.managed:
        return status  # an install made elsewhere: its own record says why
    last = LayaInstallJob.instance().status()
    if last.state != STATE_FAILED or not last.error:
        return status
    return dataclasses.replace(status, state=STATE_FAILED, managed=True, error=last.error,
                               step=_REPAIR_STEP, action=ACTION_SETUP)


def detect(retrieval_config: Any = None) -> LayaRuntimeStatus:
    """Where Laya is, if anywhere. No network, never raises, under a second.

    INSTALLING only while a job in this process is actually running — never
    from files a stopped setup left behind; those read as FAILED with
    ``action`` naming the fix.
    """
    try:
        if not _apple_silicon():
            return LayaRuntimeStatus(state=STATE_UNSUPPORTED)

        for job in (LayaInstallJob.instance(), LayaAdoptJob.instance()):
            job_status = job.status()
            if job_status.state == STATE_INSTALLING:
                return job_status

        return _with_last_failure(_on_disk(retrieval_config, runtime_dir()))
    except Exception as exc:  # noqa: BLE001 — detect() must never raise.
        logger.warning("Laya detect() failed unexpectedly: %s", exc)
        return LayaRuntimeStatus(state=STATE_NOT_INSTALLED, error="Couldn't check the install.")


# install() support — one small function per step, so tests can replace any
# one of them without touching the others.

def _venv_python_path(venv_dir: Path) -> Path:
    return venv_dir / "bin" / "python"


def _hf_cache_model_dir(hf_cache: Path, repo: str) -> Path:
    """Where a repo lives under an HF_HOME: the library's default cache is
    ``HF_HOME/hub``, and the worker runs with HF_HOME = this folder."""
    return hf_cache / "hub" / _repo_folder(repo)


def _repo_folder(repo: str) -> str:
    return "models--" + repo.replace("/", "--")


def _managed_snapshot(hf_cache: Path) -> Path:
    return _hf_cache_model_dir(hf_cache, LAYA_MODEL_REPO) / "snapshots" / LAYA_MODEL_REVISION


#: What the on-device model's loader refuses to start without. An interrupted
#: download leaves the snapshot folder in place with some of these absent, or
#: present only as a link to a file that never arrived.
_REQUIRED_WEIGHT_FILES = ("model.safetensors", "rl_agent_config.json",
                          "encoder/config.json", "mlx_config.json",
                          "tokenizer/tokenizer.json")


def _weights_complete(snapshot: Path) -> bool:
    """Whether every file the loader needs is there and readable (links followed)."""
    return snapshot.is_dir() and all((snapshot / name).is_file()
                                     for name in _REQUIRED_WEIGHT_FILES)


def _move_misplaced_weights(hf_cache: Path) -> None:
    """Earlier 4.1.18 builds downloaded into ``hf-cache/models--…`` (one level
    too high). Move such a download into place rather than fetch 800 MB again.
    The cache's links are relative inside the repo folder, so it moves whole."""
    old = hf_cache / _repo_folder(LAYA_MODEL_REPO)
    new = _hf_cache_model_dir(hf_cache, LAYA_MODEL_REPO)
    if old.is_dir() and not old.is_symlink() and not new.exists():
        new.parent.mkdir(parents=True, exist_ok=True)
        os.replace(old, new)


def _folder_size(path: Path) -> int:
    return laya_process.folder_size(path)


def _check_disk_space(path: Path) -> bool:
    probe = next((p for p in (path, *path.parents) if p.exists()), path)
    try:
        return shutil.disk_usage(probe).free >= _MIN_FREE_BYTES
    except OSError:
        return False


def _classify_subprocess_error(text: str) -> str:
    lowered = (text or "").lower()
    if any(hint in lowered for hint in _NETWORK_HINTS):
        return "network"
    return "other"


def _run_subprocess(cmd: list[str], *, timeout_s: float,
                     env: dict[str, str] | None = None) -> tuple[bool, str, str]:
    """(ok, error_kind, raw_detail-for-logs-only). Never raises; Cancel stops it."""
    return laya_process.run(cmd, timeout_s=timeout_s, env=env,
                            classify=_classify_subprocess_error)


def _create_venv(venv_dir: Path, *, timeout_s: float) -> tuple[bool, str, str]:
    return _run_subprocess([sys.executable, "-m", "venv", str(venv_dir)], timeout_s=timeout_s)


def _pip_install(python: Path, requirement: str, *, timeout_s: float) -> tuple[bool, str, str]:
    env = {**os.environ, "TOKENIZERS_PARALLELISM": "false"}
    return _run_subprocess(
        [str(python), "-m", "pip", "install", "--disable-pip-version-check", requirement],
        timeout_s=timeout_s, env=env,
    )


def _download_weights(python: Path, repo: str, revision: str, cache_dir: Path, *,
                       progress: ProgressFn | None, timeout_s: float) -> tuple[bool, str, str]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    script = (
        "import sys; from huggingface_hub import snapshot_download; "
        "snapshot_download(sys.argv[1], revision=sys.argv[2], cache_dir=sys.argv[3]); "
        "print('OK')"
    )
    # Downloading needs the network; never inherit the worker's offline pins.
    env = {k: v for k, v in os.environ.items()
           if k not in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")}
    env["TOKENIZERS_PARALLELISM"] = "false"
    # stderr goes to a file, never a pipe: the library draws progress bars on
    # stderr, and a pipe nobody reads fills up and freezes the download.
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8", errors="replace") as err:
        try:
            proc = subprocess.Popen(
                [str(python), "-c", script, repo, revision, str(cache_dir)],
                stdout=subprocess.DEVNULL, stderr=err, text=True, env=env,
            )
        except OSError as exc:
            return False, "other", str(exc)

        def _report(size: int) -> None:
            if progress is not None:
                done = min(size / _EXPECTED_DOWNLOAD_BYTES, 1.0)
                progress(0.35 + 0.55 * done, "Downloading the model weights "
                         f"({size // (1024 * 1024)} of {_EXPECTED_MB} MB)")

        stopped = laya_process.watch_download(
            proc, lambda: _folder_size(cache_dir), timeout_s=timeout_s,
            stall_s=_STALL_S, on_size=_report)
        if stopped is not None:
            return stopped
        err.seek(0)
        stderr = err.read()[-2000:]
    if proc.returncode == 0:
        return True, "", ""
    return False, _classify_subprocess_error(stderr), stderr


def _failure_message(kind: str, *, doing: str, do: str) -> str:
    """One plain-language line per step failure. ``doing``/``do`` are a
    gerund and an infinitive phrase for the timeout/generic cases."""
    if kind == "network":
        return _NETWORK_MESSAGE
    if kind == laya_process.KIND_STALLED:
        return _STALLED_MESSAGE
    if kind == laya_process.KIND_CANCELLED:
        return _CANCELLED_MESSAGE
    if kind == "timeout":
        return f"{doing} took too long and timed out. Please try again."
    return f"Couldn't {do}. Please try again."


def _progress_stamp_path(run_dir: Path) -> Path:
    return run_dir / ".install-steps.json"


def _load_stamp(run_dir: Path) -> dict[str, Any]:
    return _read_json(_progress_stamp_path(run_dir)) or {}


def _mark_stamp(run_dir: Path, step: str) -> None:
    data = _load_stamp(run_dir)
    data[step] = True
    _write_json_atomic(_progress_stamp_path(run_dir), data)


def _existing_created_at(run_dir: Path) -> str:
    data = _read_json(run_dir / ".slm-managed")
    return str(data.get("created_at", "")) if data else ""


def _acquire_install_lock(lock_path: Path):
    """Non-blocking cross-process lock. Returns an open file handle, or None
    when another process already holds it (fails fast; never waits)."""
    if fcntl is None:
        return None
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(lock_path, "w", encoding="utf-8")  # noqa: SIM115 — lifetime matches install(), closed explicitly
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        return None
    return fh


def _release_install_lock(fh) -> None:
    if fh is None:
        return
    try:
        if fcntl is not None:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    except OSError:
        pass
    finally:
        fh.close()


_STEP_VENV, _STEP_PIP, _STEP_WEIGHTS = "venv", "pip", "weights"


def _run_resumable_steps(run_dir: Path, venv_dir: Path, venv_python: Path,
                          hf_cache: Path, report: ProgressFn) -> str | None:
    """Run remaining venv/pip/weights steps; return an error or None on success."""
    stamp = _load_stamp(run_dir)

    if laya_paths.venv_refusal(venv_dir):
        shutil.rmtree(venv_dir)
        stamp.pop(_STEP_VENV, None)
    if not (stamp.get(_STEP_VENV) and venv_python.exists()):
        # A new environment is empty: whatever the stamp says was installed
        # into the old one is not in this one.
        stamp.pop(_STEP_PIP, None)
        report(0.0, "Setting up a private Python environment")
        ok, kind, detail = _create_venv(venv_dir, timeout_s=_VENV_TIMEOUT_S)
        if not ok:
            logger.warning("Laya install: venv step failed: %s", detail)
            return _failure_message(kind, doing="Setting up the environment",
                                     do="set up a private Python environment")
        _mark_stamp(run_dir, _STEP_VENV)
    report(0.05, "Python environment ready")

    if not stamp.get(_STEP_PIP):
        report(0.05, "Installing the answer-check package")
        ok, kind, detail = _pip_install(venv_python, LAYA_MLX_REQUIREMENT, timeout_s=_PIP_TIMEOUT_S)
        if not ok:
            logger.warning("Laya install: pip step failed: %s", detail)
            return _failure_message(kind, doing="Installing the answer-check package",
                                     do="install the answer-check package")
        _mark_stamp(run_dir, _STEP_PIP)
    report(0.35, "Package installed")
    if laya_process.CANCEL.is_set():
        return _CANCELLED_MESSAGE

    _move_misplaced_weights(hf_cache)
    # Done only when the weights are complete: the library resumes a partial
    # download, so fetching again costs only what is missing.
    if not (stamp.get(_STEP_WEIGHTS) and _weights_complete(_managed_snapshot(hf_cache))):
        # cache_dir is the hub folder itself: snapshot_download(cache_dir=X)
        # stores the repo at X/models--…, with no hub/ level of its own.
        ok, kind, detail = _download_weights(
            venv_python, LAYA_MODEL_REPO, LAYA_MODEL_REVISION, hf_cache / "hub",
            progress=report, timeout_s=_DOWNLOAD_TIMEOUT_S,
        )
        if not ok:
            logger.warning("Laya install: download step failed: %s", detail)
            return _failure_message(kind, doing="The download", do="download the model weights")
        _mark_stamp(run_dir, _STEP_WEIGHTS)
    report(0.90, "Weights downloaded")
    return None


def _verify_and_record(run_dir: Path, venv_python: Path, hf_cache: Path,
                        model_path: Path) -> tuple[bool, str]:
    """Runs verify() and writes .slm-managed either way — a failed verify
    still leaves evidence on disk, so detect() reports 'needs a check'
    rather than 'not installed'."""
    ok, reason = verify(str(venv_python), str(hf_cache), str(model_path))
    previous = _read_json(run_dir / ".slm-managed")
    if previous is not None:
        previous = laya_paths.managed_record(previous, run_dir)
    if reason == VERIFY_BUSY and previous is not None and _record_matches(
            previous, str(venv_python), str(model_path)):
        return bool(previous.get("verified")), reason
    record = {
        "requirement": LAYA_MLX_REQUIREMENT,
        "repo": LAYA_MODEL_REPO,
        "revision": LAYA_MODEL_REVISION,
        "created_at": _existing_created_at(run_dir) or _now_iso(),
        "verified": ok,
        "verified_at": _now_iso() if ok else "",
        "python": str(venv_python.relative_to(run_dir)),
        "model_path": str(model_path.relative_to(run_dir)),
        "hf_home": str(hf_cache.relative_to(run_dir)),
        "reason": "" if ok else reason,
    }
    _write_json_atomic(run_dir / ".slm-managed", record)
    return ok, reason


def _install_body(run_dir: Path, report: ProgressFn,
                  before_verify: Callable[[], None] | None = None) -> LayaRuntimeStatus:
    """Everything install() does once it holds the lock: disk check, the
    three resumable steps, then verify-and-record."""
    report(0.0, "Checking free disk space")
    if not _check_disk_space(run_dir):
        return LayaRuntimeStatus(
            state=STATE_FAILED, error="Not enough free disk space (needs about 1.5 GB)")

    venv_dir = run_dir / "venv"
    venv_python = _venv_python_path(venv_dir)
    hf_cache = run_dir / "hf-cache"

    error = _run_resumable_steps(run_dir, venv_dir, venv_python, hf_cache, report)
    if error is not None:
        return LayaRuntimeStatus(state=STATE_FAILED, error=error)

    model_path = _managed_snapshot(hf_cache)
    report(0.90, "Checking the install works")
    if before_verify is not None:
        before_verify()
    ok, reason = _verify_and_record(run_dir, venv_python, hf_cache, model_path)

    if not ok:
        return _needs_fix(managed=True, python=str(venv_python), hf_home=str(hf_cache),
                          model_path=str(model_path), model_revision=LAYA_MODEL_REVISION,
                          error=reason)

    report(1.0, "Ready")
    return LayaRuntimeStatus(
        state=STATE_READY, managed=True, python=str(venv_python), hf_home=str(hf_cache),
        model_path=str(model_path), model_revision=LAYA_MODEL_REVISION, progress=1.0, step="Ready",
    )


def install(*, progress: ProgressFn | None = None,
            slm_home: Path | None = None,
            before_verify: Callable[[], None] | None = None) -> LayaRuntimeStatus:
    """Create the environment, install the pinned package, download the pinned
    weights, verify. Blocking, idempotent, resumable. Needs the network.
    ``before_verify`` runs right before the model is loaded for its check."""

    def _report(fraction: float, step: str) -> None:
        if progress is not None:
            try:
                progress(fraction, step)
            except Exception:  # noqa: BLE001 — a bad UI callback can't abort an install
                pass

    if not _apple_silicon():
        return LayaRuntimeStatus(state=STATE_UNSUPPORTED, error="Laya needs Apple Silicon.")

    try:
        run_dir = runtime_dir(slm_home)
        run_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return LayaRuntimeStatus(
            state=STATE_FAILED, error=f"Couldn't set up the install folder: {exc}")

    if not LayaInstallJob.instance().running:
        # Called directly, not from the dashboard's job (which clears it when
        # it starts, so a Cancel pressed right after Set up is never lost).
        laya_process.CANCEL.clear()
    lock_fh = _acquire_install_lock(run_dir / "install.lock")
    if fcntl is not None and lock_fh is None:
        return LayaRuntimeStatus(
            state=STATE_FAILED,
            error="Another install is already running. Please wait for it to finish.",
        )

    try:
        return _install_body(run_dir, _report, before_verify)
    except Exception as exc:  # noqa: BLE001 — install() must report, never raise.
        logger.exception("Laya install() hit an unexpected error")
        return LayaRuntimeStatus(state=STATE_FAILED, error=f"Unexpected install error: {exc}")
    finally:
        _release_install_lock(lock_fh)


# adopt() — use an install that already exists.

def _is_in_temp_dir(path: Path) -> bool:
    try:
        resolved = path.resolve(strict=False)
    except OSError:
        resolved = path
    try:
        temp_root = Path(tempfile.gettempdir()).resolve(strict=False)
        resolved.relative_to(temp_root)
        return True
    except ValueError:
        pass
    text = str(resolved)
    markers = ("/private/var/folders", "/private/tmp", "/var/folders")
    return any(marker in text for marker in markers)


def _find_snapshot(hf_home: Path | None, repo: str, revision: str) -> Path | None:
    """The pinned (or refs/main) snapshot under an HF_HOME — or under the hub
    cache folder itself, which people paste just as often."""
    if hf_home is None:
        return None
    for model_dir in (_hf_cache_model_dir(hf_home, repo), hf_home / _repo_folder(repo)):
        found = _snapshot_in(model_dir, revision)
        if found is not None:
            return found
    return None


def _snapshot_in(model_dir: Path, revision: str) -> Path | None:
    snapshots = model_dir / "snapshots"
    preferred = snapshots / revision
    if preferred.is_dir():
        return preferred
    refs_main = model_dir / "refs" / "main"
    if refs_main.is_file():
        try:
            commit = refs_main.read_text(encoding="utf-8").strip()
        except OSError:
            commit = ""
        if commit:
            candidate = snapshots / commit
            if candidate.is_dir():
                return candidate
    return None


_NOT_A_REAL_INSTALL = "That looks like a temporary location, not a real install."


def _resolve_adopt_target(
    python: str, hf_home: str, model_path: str,
) -> tuple[Path, Path | None, str] | LayaRuntimeStatus:
    """Validates adopt()'s inputs. Returns the resolved (python_path,
    hf_home_path, model_path) on success, or the FAILED status to return."""
    refused = check_interpreter(python, strict_location=True)
    if refused:
        return LayaRuntimeStatus(state=STATE_FAILED, error=refused)
    python_path = Path(python)

    hf_home_path = Path(hf_home) if hf_home else None
    if hf_home_path is not None:
        if not hf_home_path.exists():
            return LayaRuntimeStatus(
                state=STATE_FAILED, error="That model cache folder can't be found.")
        if _is_in_temp_dir(hf_home_path):
            return LayaRuntimeStatus(state=STATE_FAILED, error=_NOT_A_REAL_INSTALL)

    resolved_model_path = model_path.strip() if model_path else ""
    if not resolved_model_path:
        found = _find_snapshot(hf_home_path, LAYA_MODEL_REPO, LAYA_MODEL_REVISION)
        if found is None:
            return LayaRuntimeStatus(
                state=STATE_FAILED, error="Couldn't find the model under that cache folder.")
        resolved_model_path = str(found)

    model_dir = Path(resolved_model_path)
    if not model_dir.is_dir():
        return LayaRuntimeStatus(state=STATE_FAILED, error="That model folder can't be found.")
    if _is_in_temp_dir(model_dir):
        return LayaRuntimeStatus(state=STATE_FAILED, error=_NOT_A_REAL_INSTALL)

    return python_path, hf_home_path, resolved_model_path


def adopt(python: str, hf_home: str, model_path: str = "") -> LayaRuntimeStatus:
    """Use a Laya install that already exists (for networks that block downloads)."""
    resolved = _resolve_adopt_target(python, hf_home, model_path)
    if isinstance(resolved, LayaRuntimeStatus):
        return resolved
    python_path, _hf_home_path, resolved_model_path = resolved

    ok, reason = verify(str(python_path), hf_home, resolved_model_path)
    if reason == VERIFY_BUSY:
        return LayaRuntimeStatus(state=STATE_FAILED, python=str(python_path),
                                 hf_home=hf_home, model_path=resolved_model_path,
                                 error=reason)

    run_dir = runtime_dir()
    run_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "python": str(python_path),
        "hf_home": hf_home,
        "model_path": resolved_model_path,
        "verified": ok,
        "verified_at": _now_iso() if ok else "",
        "reason": reason,
    }
    _write_json_atomic(run_dir / "adopted.json", record)

    if ok:
        return LayaRuntimeStatus(
            state=STATE_READY, managed=False, python=str(python_path),
            hf_home=hf_home, model_path=resolved_model_path,
        )
    return _needs_fix(managed=False, python=str(python_path), hf_home=hf_home,
                      model_path=resolved_model_path, error=reason)


# remove() lives in core/laya_remove.py; re-exported here, where callers
# (and tests that replace it) have always found it.
# The background jobs the dashboard starts and polls live in core/laya_jobs.py;
# re-exported here, where callers have always found them.
from superlocalmemory.core.laya_jobs import (  # noqa: E402
    LayaAdoptJob,
    LayaInstallJob,
)
from superlocalmemory.core.laya_jobs import (  # noqa: E402 - late import, order kept
    LayaTestJob as LayaTestJob,
)
from superlocalmemory.core.laya_remove import (  # noqa: E402
    forget_adopted as forget_adopted,
)
from superlocalmemory.core.laya_remove import (  # noqa: E402 - late import, order kept
    remove as remove,
)
