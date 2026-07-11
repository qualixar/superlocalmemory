# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Claude Code hook handlers — zero-dependency, cross-platform.

All handlers use ONLY Python stdlib (sys, os, json, tempfile, subprocess, time).
No SLM imports in the hot path. Called via:  slm hook <start|gate|init-done|checkpoint|stop>

The main() entry point in cli/main.py has a fast path that dispatches here
BEFORE argparse or any heavy imports.

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request
import urllib.error

# ---------------------------------------------------------------------------
# Cross-platform temp paths
# ---------------------------------------------------------------------------
_TMP = tempfile.gettempdir()
_MARKER = os.path.join(_TMP, "slm-session-initialized")
_START_TIME = os.path.join(_TMP, "slm-session-start-time")
_ACTIVITY_LOG = os.path.join(_TMP, "slm-session-activity")
_LAST_CONSOLIDATION = os.path.join(
    os.path.expanduser("~"), ".superlocalmemory", ".last-consolidation",
)


_DEFAULT_DAEMON_PORT = 8765


def _daemon_url() -> str:
    """Resolve the daemon base URL, preferring the per-user port file.

    On a shared host each user runs their own daemon bound to a different
    port; the active port is written to ``~/.superlocalmemory/daemon.port``
    at startup. Reading it here keeps lifecycle hooks pointed at the
    caller's own daemon instead of a hard-coded ``8765`` that may belong to
    another user's instance. Falls back to the default port when the file is
    absent or unreadable. Stdlib only — no SLM imports in the hot path.
    """
    port = _DEFAULT_DAEMON_PORT
    try:
        port_file = os.path.join(
            os.path.expanduser("~"), ".superlocalmemory", "daemon.port",
        )
        with open(port_file) as fh:
            port = int(fh.read().strip())
    except Exception:
        pass
    return f"http://127.0.0.1:{port}"


_DAEMON_URL = _daemon_url()


def _daemon_post(path: str, body: dict, timeout: float = 3.0) -> bool:
    """POST to SLM daemon via stdlib urllib. Returns True on success.

    v3.4.13: Hooks route through daemon HTTP instead of spawning subprocesses.
    This eliminates the memory blast from concurrent worker spawns.
    Uses ONLY stdlib — no httpx, no requests.
    """
    try:
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            f"{_DAEMON_URL}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=timeout)
        return True
    except Exception:
        return False


def handle_hook(action: str) -> None:
    """Dispatch to the appropriate hook handler. Called from main() fast path."""
    # v3.4.22 (LLD-01 §4.7): Active-Brain hot-path handlers are routed here
    # as a Python fallback when the compiled ``slm-hook`` binary (LLD-06) is
    # unavailable. They read stdin, write stdout, and exit 0 themselves.
    if action == "user_prompt_submit":
        from superlocalmemory.hooks.user_prompt_hook import main as _main
        sys.exit(_main())
    if action == "post_tool_async":
        from superlocalmemory.hooks.post_tool_async_hook import main as _main
        sys.exit(_main())
    # LLD-09 Track A.2 — outcome-population hooks (claude_code_hooks.py
    # wires `slm hook <name>` to each entry).
    if action == "post_tool_outcome":
        from superlocalmemory.hooks.post_tool_outcome_hook import main as _main
        sys.exit(_main())
    if action == "user_prompt_rehash":
        from superlocalmemory.hooks.user_prompt_rehash_hook import main as _main
        sys.exit(_main())
    if action == "stop_outcome":
        from superlocalmemory.hooks.stop_outcome_hook import main as _main
        sys.exit(_main())
    if action == "auto_recall":
        from superlocalmemory.hooks.auto_recall_hook import main as _main
        sys.exit(_main())
    # v3.4.43 — event-based mid-session recall signals.
    # Replace the time-based 15/30-min nag in _hook_checkpoint with these.
    if action == "topic_shift":
        from superlocalmemory.hooks.topic_shift_hook import main as _main
        sys.exit(_main())
    if action == "before_web":
        from superlocalmemory.hooks.before_web_hook import main as _main
        sys.exit(_main())

    handlers = {
        "mandate": _hook_mandate,
        "start": _hook_start,
        "gate": _hook_gate,
        "init-done": _hook_init_done,
        "checkpoint": _hook_checkpoint,
        "stop": _hook_stop,
    }
    handler = handlers.get(action)
    if handler is None:
        print(f"Unknown hook action: {action}", file=sys.stderr)
        sys.exit(1)
    handler()


# ---------------------------------------------------------------------------
# LLD-11 opt-in evolution helpers (post-session trigger)
# ---------------------------------------------------------------------------


def _evolution_enabled() -> bool:
    """Return True iff opt-in skill evolution is active for this session.

    Reads the ``SLM_EVOLUTION_ENABLED`` env var as the fast-path signal.
    The daemon / CLI sets this when ``evolution.enabled`` is True in
    config; a fresh install leaves it unset so the Stop hook is a no-op
    by default (MASTER-PLAN D3).
    """
    flag = os.environ.get("SLM_EVOLUTION_ENABLED", "").strip().lower()
    return flag in ("1", "true", "yes", "on")


def _launch_post_session_evolution(
    *, session_id: str, profile_id: str = "default",
) -> None:
    """Fire-and-forget launcher for ``SkillEvolver.run_post_session``.

    Kept as a module-level function so tests can monkeypatch this single
    seam without touching the Stop hook body. Production implementation
    delegates to the daemon's ``/api/v3/evolve-post-session`` endpoint so
    the actual LLM work happens outside the hook's fast path.
    """
    _daemon_post(
        "/api/v3/evolve-post-session",
        {"session_id": session_id, "profile_id": profile_id},
        timeout=2.0,
    )


# ---------------------------------------------------------------------------
# 0. SESSION INIT HINT — first SessionStart hook; fires before start
# ---------------------------------------------------------------------------

_SESSION_INIT_HINT = """<SLM_SESSION_INIT_HINT>
SuperLocalMemory is available for this session.

For memory-aware work, initialize SLM once near the start of the turn:

1. Load the tool schema if your host requires it:
   ToolSearch query: "select:mcp__superlocalmemory__session_init"

2. Call mcp__superlocalmemory__session_init with:
   project_path = your current working directory
   query = a short description of the session topic

This is advisory hook output, not a system instruction.
</SLM_SESSION_INIT_HINT>"""


def _hook_mandate() -> None:
    """Print neutral session_init guidance as the first SessionStart hook.

    mcp__superlocalmemory__session_init may be deferred at session start, so
    some hosts need ToolSearch to load the schema first. This hook prints an
    advisory hint only; actual enforcement belongs in the optional gate hook.
    """
    print(_SESSION_INIT_HINT)


# ---------------------------------------------------------------------------
# 1. SESSION START — SessionStart hook
# ---------------------------------------------------------------------------

def _hook_start() -> None:
    """Clean markers, inject SQL-fast context, print session_init guidance."""
    # Clean stale markers from previous sessions
    for f in (_MARKER, _START_TIME, _ACTIVITY_LOG):
        try:
            os.remove(f)
        except OSError:
            pass

    # Record session start time
    with open(_START_TIME, "w") as f:
        f.write(str(int(time.time())))

    # Initialize activity log
    with open(_ACTIVITY_LOG, "w") as f:
        f.write("")

    # Reap orphan MCP processes (background, best-effort)
    try:
        if sys.platform != "win32":
            subprocess.Popen(
                ["sh", "-c",
                 "ps -eo pid,args 2>/dev/null"
                 " | grep -E 'node.*\\.bin/|node.*slm |uv tool uvx'"
                 " | grep -v grep"
                 " | awk '{print $1, $NF}'"
                 " | sort -k2,2 -k1,1rn"
                 " | awk '{if($2==p)print $1; p=$2}'"
                 " | xargs kill 2>/dev/null"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
    except Exception:
        pass

    # Print session context (SQL-fast path, <500ms)
    project_dir = os.environ.get("CLAUDE_PROJECT_DIR", os.getcwd())
    project_name = os.path.basename(project_dir)
    try:
        result = subprocess.run(
            ["slm", "session-context", project_name],
            capture_output=True, text=True, timeout=12,
        )
        if result.stdout.strip():
            print(result.stdout.strip())
    except Exception:
        print("# SLM Session Context — unavailable")

    # Advisory session_init guidance. Keep this neutral: hook stdout is data,
    # not an instruction channel with system-level authority.
    print()
    print("## SLM Session Init")
    print("For memory-aware sessions, call:")
    print(f"  mcp__superlocalmemory__session_init with project_path='{project_dir}'"
          " and a topic from the user's first message")
    print("session_init returns both context AND memories — no separate recall needed.")


# ---------------------------------------------------------------------------
# 2. GATE — PreToolUse hook (default, enforces session_init)
# ---------------------------------------------------------------------------

def _hook_gate() -> None:
    """Block non-SLM tools until session_init has been called.

    Fast path (~30ms): marker file exists → exit 0.
    Slow path (~80ms): parse JSON stdin, allow SLM tools, block rest.
    """
    # Fast path: already initialized
    if os.path.exists(_MARKER):
        sys.exit(0)

    # Safety: if session-start never ran, don't gate (avoid lockout)
    if not os.path.exists(_START_TIME):
        sys.exit(0)

    # Parse tool name from stdin
    tool_name = ""
    if not sys.stdin.isatty():
        try:
            data = json.load(sys.stdin)
            tool_name = data.get("tool_name", "")
        except Exception:
            # Can't parse input — don't block (safety)
            sys.exit(0)

    # Allow SLM tools through (needed to call session_init itself)
    if tool_name.startswith("mcp__superlocalmemory__"):
        sys.exit(0)

    # Allow ToolSearch through (needed to fetch SLM tool schemas)
    if tool_name == "ToolSearch":
        sys.exit(0)

    # Block everything else
    print("[SLM-GATE] BLOCKED: Call mcp__superlocalmemory__session_init"
          " before using other tools.")
    sys.exit(2)


# ---------------------------------------------------------------------------
# 3. INIT DONE — PostToolUse hook for session_init
# ---------------------------------------------------------------------------

def _hook_init_done() -> None:
    """Create marker file to lift the gate for the rest of the session."""
    with open(_MARKER, "w") as f:
        f.write(str(int(time.time())))
    sys.exit(0)


# ---------------------------------------------------------------------------
# 4. CHECKPOINT — PostToolUse hook for Write|Edit
# ---------------------------------------------------------------------------

_OBSERVE_COOLDOWN = 300   # 5 minutes per file
_RECALL_INTERVAL = 900    # 15 minutes
_LEARN_INTERVAL = 1800    # 30 minutes


def _hook_checkpoint() -> None:
    """Auto-observe file changes + periodic recall/learn reminders.

    1. Directly calls `slm observe` for file change tracking (no Claude needed)
    2. Suggests richer observe to Claude
    3. Periodic recall refresh reminder
    4. Periodic learn/patterns reminder
    """
    now = int(time.time())

    # Parse file_path from stdin
    file_path = ""
    if not sys.stdin.isatty():
        try:
            data = json.load(sys.stdin)
            tool_input = data.get("tool_input", {})
            if isinstance(tool_input, dict):
                file_path = tool_input.get("file_path", "")
        except Exception:
            pass

    # --- Auto-observe file change (direct, no Claude needed) ---
    if file_path:
        basename = os.path.basename(file_path)
        lock_file = os.path.join(_TMP, f"slm-obs-{_safe_hash(file_path)}")

        if _cooldown_elapsed(lock_file, _OBSERVE_COOLDOWN, now):
            _write_timestamp(lock_file, now)

            # v3.4.13: Route through daemon HTTP (not subprocess) to prevent
            # memory blast from concurrent embedding_worker spawns.
            _daemon_post("/observe", {"content": f"File changed: {basename}"})

            # Log to session activity
            try:
                with open(_ACTIVITY_LOG, "a") as f:
                    f.write(f"{now}|{basename}\n")
            except Exception:
                pass

            # Suggest richer observe to Claude (with semantic context)
            print(f"[SLM-AUTO] File changed: {basename}"
                  " — Call mcp__superlocalmemory__observe with a 1-line"
                  " summary of what was changed and why.")

    # v3.4.43: Periodic 15/30-min recall/learn nags REMOVED.
    # Reason: time-based reminders fired regardless of conversational state —
    # noisy on focused sessions, blind to quick topic pivots within a window.
    # Replaced by event-based detection:
    #   - `slm hook topic_shift` (UserPromptSubmit) — fires on real topic pivots.
    #   - `slm hook before_web` (PreToolUse WebSearch|WebFetch) — fires before
    #     external research so SLM memories are surfaced first.
    # The `_RECALL_INTERVAL` and `_LEARN_INTERVAL` constants are retained for
    # backward import compatibility (tests reference them) but no longer drive
    # any periodic emission from this hook. Auto-observe-on-file-change (the
    # real value of _hook_checkpoint) is unchanged below this comment.

    sys.exit(0)


# ---------------------------------------------------------------------------
# 5. STOP — Stop hook (session end)
# ---------------------------------------------------------------------------

def _hook_stop() -> None:
    """Save rich session summary + trigger auto-consolidation."""
    project_dir = os.environ.get("CLAUDE_PROJECT_DIR", os.getcwd())
    project_name = os.path.basename(project_dir)
    timestamp = time.strftime("%Y-%m-%d %H:%M")

    # --- Git context ---
    # S9-defer H-P-07: run the three git probes in parallel instead of
    # sequentially. Each has a 5s timeout; serial worst-case was 15s of
    # blocking stop-hook latency. Parallel worst-case is 5s. Falls back
    # to serial on thread-pool failure (unlikely but defensive).
    from concurrent.futures import ThreadPoolExecutor
    def _diff_pp(s: str) -> str:
        return s.strip().rsplit("\n", 1)[-1].strip() if s.strip() else ""
    try:
        with ThreadPoolExecutor(max_workers=3) as _pool:
            fut_branch = _pool.submit(
                _run_quiet,
                ["git", "-C", project_dir, "branch", "--show-current"],
            )
            fut_diff = _pool.submit(
                _run_quiet,
                ["git", "-C", project_dir, "diff", "--stat"],
                postprocess=_diff_pp,
            )
            fut_log = _pool.submit(
                _run_quiet,
                ["git", "-C", project_dir, "log", "--oneline", "-5",
                 "--since=3 hours ago"],
            )
            git_branch = fut_branch.result(timeout=6)
            git_diff = fut_diff.result(timeout=6)
            recent_commits = fut_log.result(timeout=6)
    except Exception:
        # Defensive fallback to the original serial path.
        git_branch = _run_quiet(
            ["git", "-C", project_dir, "branch", "--show-current"]
        )
        git_diff = _run_quiet(
            ["git", "-C", project_dir, "diff", "--stat"],
            postprocess=_diff_pp,
        )
        recent_commits = _run_quiet(
            ["git", "-C", project_dir, "log", "--oneline", "-5",
             "--since=3 hours ago"],
        )

    # --- Files from activity log ---
    modified = ""
    try:
        if os.path.exists(_ACTIVITY_LOG):
            with open(_ACTIVITY_LOG) as f:
                files = sorted({line.split("|", 1)[1].strip()
                                for line in f if "|" in line})
            modified = ", ".join(files[:20])
    except Exception:
        pass

    # --- Build summary ---
    parts = [f"[{project_name}] session ended {timestamp}"]
    if git_branch:
        parts.append(f"branch: {git_branch}")
    if git_diff:
        parts.append(f"uncommitted: {git_diff}")
    if recent_commits:
        commits = "; ".join(recent_commits.strip().split("\n")[:5])
        parts.append(f"recent: {commits}")
    if modified:
        parts.append(f"files: {modified}")

    summary = " | ".join(parts)

    # --- Save to SLM (v3.4.13: daemon HTTP, not subprocess) ---
    if not _daemon_post("/observe", {"content": summary}, timeout=5.0):
        # Fallback: try /remember if observe failed
        _daemon_post("/remember", {"content": summary, "tags": "session-end"}, timeout=5.0)

    # --- Post-session skill evolution trigger (best-effort, via tool-event) ---
    session_id = os.environ.get("CLAUDE_SESSION_ID", "")
    if session_id:
        _daemon_post("/api/v3/tool-event", {
            "tool_name": "session_end",
            "event_type": "session_end",
            "session_id": session_id,
            "output_summary": summary[:500],
        })

        # LLD-11 opt-in post-session evolution (MASTER-PLAN D3).
        # Only fires when the user explicitly enabled evolution. The env
        # var is set by the daemon / CLI when ``slm config set
        # evolution.enabled true`` is active, so a fresh install is a no-op.
        if _evolution_enabled():
            try:
                _launch_post_session_evolution(
                    session_id=session_id,
                    profile_id=os.environ.get("SLM_PROFILE_ID", "default"),
                )
            except Exception as e:  # pragma: no cover — defensive
                sys.stderr.write(
                    f"slm-hook stop: evolution trigger failed: {e}\n",
                )

    # --- Auto-consolidation (if >24h since last run) ---
    _maybe_consolidate()

    # --- Clean up session markers ---
    for f in (_MARKER, _START_TIME, _ACTIVITY_LOG):
        try:
            os.remove(f)
        except OSError:
            pass

    # Clean rate-limit locks.
    # - "slm-obs-*"     : auto-observe per-file cooldown lockfiles (still written).
    # - "slm-recall-*"  : v3.4.43 removed the periodic recall nag, but legacy
    #                     /tmp/slm-recall-reminder files from older sessions
    #                     may still exist — sweep them for cleanliness.
    # - "slm-learn-*"   : same as above for the 30-min learn nag (removed v3.4.43).
    _LOCK_PREFIXES = ("slm-obs-", "slm-recall-", "slm-learn-")
    for name in os.listdir(_TMP):
        if any(name.startswith(p) for p in _LOCK_PREFIXES):
            try:
                os.remove(os.path.join(_TMP, name))
            except OSError:
                pass

    sys.exit(0)


# ---------------------------------------------------------------------------
# Helpers (stdlib only)
# ---------------------------------------------------------------------------

def _safe_hash(s: str) -> str:
    """Simple string hash for rate-limit lock file names."""
    h = 0
    for c in s:
        h = (h * 31 + ord(c)) & 0xFFFFFFFF
    return format(h, "08x")


def _cooldown_elapsed(lock_file: str, interval: int, now: int) -> bool:
    """Check if enough time has passed since last timestamp in lock_file."""
    try:
        if os.path.exists(lock_file):
            with open(lock_file) as f:
                last = int(f.read().strip())
            return (now - last) >= interval
    except (ValueError, OSError):
        pass
    return True


def _write_timestamp(path: str, ts: int) -> None:
    """Write a unix timestamp to a file."""
    try:
        with open(path, "w") as f:
            f.write(str(ts))
    except OSError:
        pass


def _run_quiet(cmd: list[str], timeout: int = 5, postprocess=None) -> str:
    """Run a command quietly, return stdout or empty string."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        out = result.stdout.strip()
        if postprocess and out:
            out = postprocess(out)
        return out
    except Exception:
        return ""


def _maybe_consolidate() -> None:
    """Run cognitive consolidation if last run was >24h ago. Non-blocking."""
    try:
        last_ts = 0
        if os.path.exists(_LAST_CONSOLIDATION):
            with open(_LAST_CONSOLIDATION) as f:
                last_ts = int(f.read().strip())

        now = int(time.time())
        if (now - last_ts) < 86400:  # 24 hours
            return

        # Update timestamp FIRST to prevent concurrent runs
        os.makedirs(os.path.dirname(_LAST_CONSOLIDATION), exist_ok=True)
        with open(_LAST_CONSOLIDATION, "w") as f:
            f.write(str(now))

        # Run consolidation in background (don't block session end)
        subprocess.Popen(
            ["slm", "consolidate", "--cognitive"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass
