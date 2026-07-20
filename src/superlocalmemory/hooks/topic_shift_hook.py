# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.43 — Topic-shift detection on UserPromptSubmit

"""Topic-shift detection hook — replaces time-based recall nag.

Replaces the time-based "[SLM] 15+ min since last context refresh" reminder
emitted by _hook_checkpoint with event-based detection. Fires a single-line
recall reminder only when the current prompt's content-word set has zero
overlap with EVERY recent prompt in a 5-prompt sliding window — the strictest
defensible signal for a genuine topic pivot.

Dispatch: `slm hook topic_shift` (UserPromptSubmit).

HOT-PATH CONTRACT
=================
- stdlib-only imports at module load.
- Reads {"session_id", "prompt"} from stdin JSON.
- On topic shift: prints one-line reminder to stdout (Claude Code surfaces
  as system-reminder).
- On no-shift / any error: silent exit 0. Never blocks the prompt.
- Latency budget: <10 ms (regex + set ops on bounded input). Verified
  by the algorithm itself; subprocess startup adds ~30-40 ms but that's
  outside the budget for the Python logic.
- State file per root and session:
  /tmp/slm-topicstate-{root_hash}-{sha256(session_id)[:16]}.json
  Schema: {"window": [[word, ...], ...], "version": 1}.

DESIGN NOTES (NASA-grade — defensible thresholds, e2e-tuned)
============================================================
- N=5 sliding window — spans conversational follow-ups, still detects shifts
  in long sessions.
- Algorithm: per-prompt MAX overlap (NOT jaccard-vs-union). True pivots share
  zero content words with EVERY recent prompt; same-topic follow-ups share
  at least one anchor word with at least ONE recent prompt (often not with
  the union). Per-prompt max captures this; jaccard-vs-union over-fires.
- |current_words| >= 5 — skip short utterances. Trade-off: very short pivots
  ("monsoon forecast Mumbai") miss firing. Bounded cost: one missed reminder;
  Claude self-trigger covers the residual.
- >= 2 prior window entries — don't trigger on prompt 2 (insufficient baseline).
- Word regex drops hyphens vs the topic_signature regex: compound technical
  terms like "varunpratap-website" split into ["varunpratap", "website"] so
  each half independently anchors against the window.
- Extended stopword list (generic temporal connectors: "next", "back",
  "week"...) prevents false-negative bridges across unrelated topics.
- Observability: every decision logged TSV to a per-user log file unless
  SLM_TOPIC_SHIFT_LOG=0 in environment.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import tempfile
import time

# --------------------------------------------------------------------------
# Config — frozen for v3.4.43. Tune via real-conversation log analysis.
# --------------------------------------------------------------------------

_WINDOW_SIZE = 5
_MIN_CURRENT_WORDS = 5
_MIN_WINDOW_ENTRIES = 2
_MAX_PER_PROMPT_OVERLAP = 0
_STATE_MAX_AGE_SEC = 24 * 3600
_MAX_PROMPT_CHARS = 4000

_TMP = tempfile.gettempdir()

_STOPWORDS: frozenset[str] = frozenset({
    "a", "about", "above", "after", "again", "against", "all", "am", "an",
    "and", "any", "are", "as", "at", "be", "because", "been", "before",
    "being", "below", "between", "both", "but", "by", "can", "cannot",
    "could", "did", "do", "does", "doing", "don", "down", "during", "each",
    "few", "for", "from", "further", "had", "has", "have", "having", "he",
    "her", "here", "hers", "herself", "him", "himself", "his", "how", "i",
    "if", "in", "into", "is", "it", "its", "itself", "just", "let", "me",
    "more", "most", "my", "myself", "no", "nor", "not", "now", "of", "off",
    "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves",
    "out", "over", "own", "same", "she", "should", "so", "some", "such",
    "than", "that", "the", "their", "theirs", "them", "themselves", "then",
    "there", "these", "they", "this", "those", "through", "to", "too",
    "under", "until", "up", "use", "using", "very", "was", "we", "were",
    "what", "when", "where", "which", "while", "who", "whom", "why", "will",
    "with", "would", "you", "your", "yours", "yourself", "yourselves",
    "ok", "okay", "yes", "no", "yep", "nope", "thanks", "please", "go",
    "tell", "let's", "lets", "want", "need", "would", "could", "make",
    "also", "still", "really", "actually",
    "next", "back", "here", "there", "now", "then", "again", "today",
    "tomorrow", "yesterday", "week", "month", "year", "day", "time",
    "thing", "things", "stuff", "way", "ways", "case", "cases",
})

# Linear-time non-backtracking word regex. Hyphens excluded so compound
# technical terms split into independently-matchable halves.
_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9']{2,}")

_ACK_RE = re.compile(
    r"^\s*(yes|no|ok|okay|approved|thanks|thank you|go|sure|yep|nope|done|y|n|"
    r"cool|got it|right|correct)([\s]+(yes|no|ok|okay|approved|thanks|done|\d+))*\s*[\.\!\?]?\s*$",
    re.IGNORECASE,
)

_SHIFT_REMINDER = (
    "[SLM] Topic shift detected. Consider calling "
    "mcp__superlocalmemory__recall with the new topic to surface relevant "
    "memories before responding."
)

# Optional test/embedder overrides. Runtime defaults are resolved dynamically
# through the canonical data-root authority so environment switches in a
# long-lived process do not continue logging to the first observed namespace.
_LOG_DIR: str | None = None
_LOG_PATH: str | None = None
_LOG_ENABLED = os.environ.get("SLM_TOPIC_SHIFT_LOG", "1") != "0"
_LOG_PROMPT_PREVIEW_CHARS = 80


# --------------------------------------------------------------------------
# Pure logic — testable without IO.
# --------------------------------------------------------------------------

def extract_content_words(prompt: str) -> list[str]:
    """Tokenize → lowercase → filter stopwords + len<3. Bounded input."""
    if not prompt:
        return []
    if len(prompt) > _MAX_PROMPT_CHARS:
        prompt = prompt[:_MAX_PROMPT_CHARS]
    words = _WORD.findall(prompt.lower())
    return [w for w in words if w not in _STOPWORDS and len(w) >= 3]


def is_substantive(prompt: str) -> bool:
    """Substantive = length >= 10 AND not a pure conversational ack."""
    if not prompt or len(prompt) < 10:
        return False
    if len(prompt) <= 30 and _ACK_RE.match(prompt):
        return False
    return True


def detect_shift(
    current_words: list[str],
    window: list[list[str]],
) -> tuple[bool, int]:
    """Pure decision function.

    Returns (fired, max_overlap_or_-1_when_gated).
    """
    if len(current_words) < _MIN_CURRENT_WORDS:
        return False, -1
    if len(window) < _MIN_WINDOW_ENTRIES:
        return False, -1
    cur = set(current_words)
    max_overlap = max(len(cur & set(wl)) for wl in window)
    return max_overlap <= _MAX_PER_PROMPT_OVERLAP, max_overlap


# --------------------------------------------------------------------------
# IO — state file + stdin parsing + stdout emission.
# --------------------------------------------------------------------------

def state_path(session_id: str) -> str:
    """Hash canonical root and session ID for an isolated safe filename."""
    from superlocalmemory.infra.data_root import canonical_data_root

    root_digest = hashlib.sha256(
        str(canonical_data_root()).encode("utf-8")
    ).hexdigest()[:16]
    session_digest = hashlib.sha256(
        session_id.encode("utf-8")
    ).hexdigest()[:16]
    return os.path.join(
        _TMP,
        f"slm-topicstate-{root_digest}-{session_digest}.json",
    )


def load_state(path: str) -> list[list[str]]:
    """Load window from disk. Empty on any failure or staleness."""
    try:
        st = os.stat(path)
        if (time.time() - st.st_mtime) > _STATE_MAX_AGE_SEC:
            return []
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return []
        if data.get("version") != 1:
            return []
        win = data.get("window", [])
        if not isinstance(win, list):
            return []
        out: list[list[str]] = []
        for entry in win[-_WINDOW_SIZE:]:
            if isinstance(entry, list) and all(isinstance(w, str) for w in entry):
                out.append(entry)
        return out
    except (FileNotFoundError, json.JSONDecodeError, OSError, ValueError):
        return []


def save_state(path: str, window: list[list[str]]) -> None:
    """Persist window. Silent on any IO failure."""
    tmp = f"{path}.{os.getpid()}.{time.time_ns()}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"version": 1, "window": window[-_WINDOW_SIZE:]}, f)
        os.replace(tmp, path)
    except OSError:
        try:
            os.remove(tmp)
        except OSError:
            pass


def _topic_log_paths() -> tuple[str, str]:
    """Return ``(directory, file)`` while retaining explicit test overrides."""
    if _LOG_PATH is not None:
        directory = _LOG_DIR or os.path.dirname(_LOG_PATH)
        return directory, _LOG_PATH
    if _LOG_DIR is not None:
        return _LOG_DIR, os.path.join(_LOG_DIR, "topic-shift.log")

    from superlocalmemory.infra.data_root import state_path as runtime_state_path

    path = str(runtime_state_path("logs", "topic-shift.log"))
    return os.path.dirname(path), path


def _read_input() -> tuple[str, str]:
    """Parse stdin JSON. Returns ('', '') on any failure."""
    try:
        raw = sys.stdin.read()
        if not raw:
            return "", ""
        data = json.loads(raw)
        if not isinstance(data, dict):
            return "", ""
        sid = data.get("session_id", "")
        prompt = data.get("prompt", "")
        if not isinstance(sid, str) or not isinstance(prompt, str):
            return "", ""
        return sid, prompt
    except (json.JSONDecodeError, ValueError, OSError):
        return "", ""


def _log_decision(
    session_id: str,
    current_words: list[str],
    window: list[list[str]],
    max_overlap: int,
    fired: bool,
    prompt: str,
) -> None:
    """Append one decision line for observability. Silent on failure."""
    if not _LOG_ENABLED:
        return
    try:
        log_dir, log_path = _topic_log_paths()
        os.makedirs(log_dir, exist_ok=True)
        ts = time.strftime("%Y-%m-%dT%H:%M:%S")
        sh = hashlib.sha256(session_id.encode()).hexdigest()[:8]
        preview = (prompt[:_LOG_PROMPT_PREVIEW_CHARS]
                   .replace("\t", " ").replace("\n", " "))
        line = (f"{ts}\t{sh}\t{len(current_words)}\t{len(window)}"
                f"\t{max_overlap}\t{int(fired)}\t{preview}\n")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line)
    except OSError:
        pass


def main() -> int:
    """Entry point. Always returns 0 — fail-open contract."""
    try:
        session_id, prompt = _read_input()
        if not session_id or not prompt:
            return 0
        if not is_substantive(prompt):
            return 0

        current = extract_content_words(prompt)
        path = state_path(session_id)
        window = load_state(path)

        fired, max_overlap = detect_shift(current, window)

        if fired:
            print(_SHIFT_REMINDER)

        _log_decision(session_id, current, window, max_overlap, fired, prompt)

        window.append(current)
        save_state(path, window)
    except Exception:  # noqa: BLE001 — fail-open contract
        pass
    return 0
