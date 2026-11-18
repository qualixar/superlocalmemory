# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The need gate: should this prompt get memories at all? (WS-4, M-16)

Per-prompt recall puts memories into the agent's context before it answers.
Many prompts need none ("yes", "2+2", "translate this"), and every needless
injection costs the agent attention and tokens. The gate decides per turn,
in two steps:

1. **Rules** skip the obvious turns (an acknowledgement, a greeting, pure
   arithmetic, a translation request, a slash command). They are fixed, so
   they cannot adapt to look good; their miss rate is measured and reported.
2. **A score** ranks the remaining prompts by how likely memory is to help
   (cues such as "last time", "we decided", names, paths, questions). A
   prompt is skipped when its score is below a threshold ``lam``.

**The threshold is never hand-picked (M-16).** It is set by conformal risk
control (Angelopoulos et al., 2022) over logged turns, for the loss "this
turn needed memory and was skipped" (monotone in ``lam``):

    lam = the largest grid score with
          (sum_i w_i * needed_i * skipped_i(lam) + w_max) / (sum_i w_i + w_max) <= NEED_ALPHA

Labels exist only for turns that were recalled, so a skipped turn is recalled
anyway with logged probability ``EXPLORE`` and counts with weight
``1 / EXPLORE`` (importance-weighted conformal, Tibshirani et al., 2019; the
unseen next turn takes the largest weight, which is conservative). A turn
"needed" memory when an agent used a memory delivered for it within the
outcome window (the M-12 ``used`` signals) or pulled memory itself (an
explicit recall in the same conversation before its next prompt). With too
few labels no ``lam`` passes and the score skips nothing: the gate starts
from "recall every prompt the rules let through" and earns the right to skip.

Caveat, stated in the report: usage detection finds some uses, not all
(M-20), so "needed" is a lower bound and the risk is measured against
detected need.
"""

from __future__ import annotations

import logging
import math
import re
import sqlite3
import threading
import time
import uuid
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

#: Policy (awaits the phase review, D-47): the share of needed turns the gate
#: may skip, at the conformal guarantee's level.
NEED_ALPHA = 0.10
#: The logged chance that a turn the gate would skip is recalled anyway. Any
#: value keeps the estimate unbiased; it sets only how fast labels arrive.
EXPLORE = 0.05
#: Calibration window: turns from the last 30 days whose outcome window closed.
WINDOW_DAYS = 30
#: M-12's outcome window.
OUTCOME_HOURS = 24
#: Routing only: how long a computed threshold is reused before recomputing.
CACHE_S = 600.0
#: Routing only: the longest prompt the score reads.
MAX_CHARS = 4000

RULE_EMPTY = "empty"
RULE_ACK = "ack"
RULE_GREETING = "greeting"
RULE_ARITHMETIC = "arithmetic"
RULE_TRANSLATION = "translation"
RULE_COMMAND = "command"
RULES = (RULE_EMPTY, RULE_ACK, RULE_GREETING, RULE_ARITHMETIC, RULE_TRANSLATION, RULE_COMMAND)

_ACK_WORDS = (r"(?:yes|no|ok|okay|approved|thanks|thank you|go|go ahead|ahead|sure|yep|nope|done|"
              r"y|n|cool|got it|right|correct|continue|proceed|perfect|great|nice|please|\d+)")
_ACK = re.compile(rf"^\s*{_ACK_WORDS}(?:[\s,]+{_ACK_WORDS})*\s*[.!?]*\s*$", re.IGNORECASE)
_GREETING = re.compile(
    r"^\s*(?:hi|hello|hey|hiya|good (?:morning|afternoon|evening|night)|gm|namaste)"
    r"(?:[\s,]+(?:there|claude|all|team))?\s*[.!?]*\s*$", re.IGNORECASE)
_ARITHMETIC = re.compile(r"^\s*(?:what\s+is\s+|calc(?:ulate)?\s+)?[\d\s.,+\-*/x×÷^%()=]+\??\s*$",
                         re.IGNORECASE)
_TRANSLATION = re.compile(r"^\s*translate\b.{0,400}\b(?:to|into)\s+\w+", re.IGNORECASE | re.DOTALL)
_COMMAND = re.compile(r"^\s*/[a-z][\w:-]*(?:\s|$)", re.IGNORECASE)

_WORD = re.compile(r"[^\W_]+(?:['’][^\W_]+)?", re.UNICODE)
_CUES = re.compile(
    r"\b(?:remember|recall|last time|earlier|before|previous(?:ly)?|yesterday|again|"
    r"we (?:decided|agreed|said|discussed|built|chose|did)|did (?:we|i|you)|"
    r"what (?:was|were|did)|continue|resume|pick (?:it )?up|status|where (?:were|are) we|"
    r"checkpoint|my|our|usual|as always|like before)\b", re.IGNORECASE)
_QUESTION = re.compile(r"\?|^\s*(?:who|what|when|where|why|how|which|is|are|do|does|can|should)\b",
                       re.IGNORECASE)
_NAMES = re.compile(r"(?:\b[A-Z][a-z]+[A-Z]\w*|\b\w+_\w+\b|[\w.-]+/[\w./-]+|\bv?\d+\.\d+(?:\.\d+)?\b"
                    r"|(?<=[a-z,;] )[A-Z][a-z]{2,})")


def rule(prompt: str) -> str | None:
    """The rule that skips this prompt, or None."""
    text = (prompt or "").strip()
    if not text:
        return RULE_EMPTY
    if len(text) <= 40 and _ACK.match(text):
        return RULE_ACK
    if len(text) <= 40 and _GREETING.match(text):
        return RULE_GREETING
    if len(text) <= 80 and re.search(r"\d", text) and _ARITHMETIC.match(text):
        return RULE_ARITHMETIC
    if _TRANSLATION.match(text):
        return RULE_TRANSLATION
    if _COMMAND.match(text):
        return RULE_COMMAND
    return None


def score(prompt: str) -> float:
    """How likely memory is to help, in [0, 1]. A ranking only: the conformal
    threshold makes any ranking safe; a better one just skips more."""
    text = (prompt or "")[:MAX_CHARS]
    n = len(_WORD.findall(text))
    s = 0.35 * min(1.0, math.log1p(n) / math.log1p(40))
    if _CUES.search(text):
        s += 0.35
    if _QUESTION.search(text):
        s += 0.10
    if _NAMES.search(text):
        s += 0.20
    return round(min(1.0, s), 6)


def _skipped(rule_: str | None, score_: float, lam: float | None) -> bool:
    return rule_ is not None or (lam is not None and score_ < lam)


@dataclass(frozen=True)
class Labelled:
    """One recalled turn whose outcome window closed."""

    rule: str | None
    score: float
    needed: bool
    weight: float          # 1 / P(this turn was recalled)


def bound(rows: Sequence[Labelled], lam: float | None) -> float:
    """Conformal upper bound on the share of needed turns skipped at ``lam``."""
    if not rows:
        return 1.0
    w_max = max(r.weight for r in rows)
    total = sum(r.weight for r in rows)
    loss = sum(r.weight for r in rows if r.needed and _skipped(r.rule, r.score, lam))
    return (loss + w_max) / (total + w_max)


def calibrate(rows: Sequence[Labelled], alpha: float = NEED_ALPHA) -> float | None:
    """The largest score threshold whose bound is within ``alpha``, or None
    (the score skips nothing). The bound only grows with ``lam``, so the grid
    is walked upward and stops at the first failure."""
    if bound(rows, None) > alpha:
        return None
    best = None
    for lam in sorted({r.score for r in rows if r.rule is None}):
        nxt = math.nextafter(lam, math.inf)  # skip every score <= lam
        if bound(rows, nxt) > alpha:
            break
        best = nxt
    return best


@dataclass(frozen=True)
class Decision:
    turn_id: str
    recall: bool
    rule: str | None
    score: float
    threshold: float | None
    explored: bool
    #: P(recall) for this turn under the logged policy: 1 when the gate let it
    #: through, EXPLORE when it was skipped and drawn, 0 when skipped (and no
    #: draw could be made, or the draw said no).
    propensity: float
    at: datetime
    #: False when ``threshold`` was only logged (shadow), never applied.
    acting: bool = True


def flag_state(slm_home: Path | None) -> str | None:
    """``"on"`` or ``"shadow"``, or None when per-prompt recall is off (the
    flag off, or the kill switch). Fails closed: an unreadable flag is off."""
    try:
        from superlocalmemory.core.feature_flags import FeatureFlags

        flags = FeatureFlags(slm_home)
        if flags.killed:
            return None
        state = flags.state("delivery.need_gate").value
    except Exception:  # noqa: BLE001
        return None
    return state if state in ("on", "shadow") else None


def _draw(key: bytes | None, turn_id: str) -> float | None:
    if not key:
        return None
    from superlocalmemory.outcomes.randomiser import draw

    return draw(key, turn_id, "need_explore")


def decide(prompt: str, *, threshold: float | None, key: bytes | None,
           now: datetime | None = None) -> Decision:
    """One turn's decision. ``key`` is the receipt key (no key: no exploration)."""
    at = now or datetime.now(UTC)
    turn_id = f"turn_{uuid.uuid4().hex}"
    rule_ = rule(prompt)
    s = score(prompt) if rule_ is None else 0.0
    if not _skipped(rule_, s, threshold):
        return Decision(turn_id, True, None, s, threshold, False, 1.0, at)
    u = _draw(key, turn_id)
    if u is None:
        return Decision(turn_id, False, rule_, s, threshold, False, 0.0, at)
    explored = u < EXPLORE
    return Decision(turn_id, explored, rule_, s, threshold, explored,
                    EXPLORE if explored else 1.0 - EXPLORE, at)


_INSERT = ("INSERT OR IGNORE INTO need_turns (turn_id, profile_id, session_id, at, prompt_digest,"
           " rule, score, threshold, acting, recalled, explored, propensity, recall_query_id)"
           " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)")


def log(db_path: Path, decision: Decision, *, profile_id: str, session_id: str | None,
        prompt_digest: str | None, recall_query_id: str | None = None) -> bool:
    """Store one turn. ``propensity`` is stored as P(recalled) for recalled
    turns and P(skipped) for skipped ones (the probability of what happened)."""
    from superlocalmemory.core.session_identity import is_conversation

    sid = session_id if is_conversation(session_id, profile_id) else None
    try:
        conn = sqlite3.connect(str(db_path), timeout=0.25)
        try:
            conn.execute(_INSERT, (
                decision.turn_id, profile_id, sid, decision.at.isoformat(), prompt_digest,
                decision.rule, decision.score, decision.threshold, int(decision.acting),
                int(decision.recall),
                int(decision.explored), decision.propensity, recall_query_id))
            conn.commit()
        finally:
            conn.close()
        return True
    except sqlite3.Error as exc:
        logger.info("need-gate turn not logged: %s", exc)
        return False


def log_later(db_path: Path, decision: Decision, **kw) -> None:
    """``log`` on a short-lived thread, so a prompt never waits for the write."""
    threading.Thread(target=log, args=(db_path, decision), kwargs=kw, daemon=True,
                     name="slm-need-log").start()


#: Recalled turns whose window closed, with their next prompt in the same
#: conversation (the end of the explicit-pull window).
_LABELLED = """
WITH t AS (
  SELECT turn_id, session_id, at, rule, score, recalled, propensity, recall_query_id,
         LEAD(at) OVER (PARTITION BY session_id ORDER BY at) AS next_at
    FROM need_turns
   WHERE profile_id = ? AND at >= ?
)
SELECT t.rule, t.score, t.propensity,
       EXISTS (SELECT 1 FROM injection_deliveries d
                 JOIN injection_usage u ON u.delivery_id = d.delivery_id
                WHERE d.profile_id = ? AND t.recall_query_id IS NOT NULL
                  AND d.recall_query_id = t.recall_query_id
                  AND julianday(u.observed_at) <= julianday(t.at) + ? / 24.0) AS used,
       EXISTS (SELECT 1 FROM injection_deliveries p
                WHERE p.profile_id = ? AND t.session_id IS NOT NULL
                  AND p.session_id = t.session_id
                  AND p.path IN ('mcp_recall', 'cli_recall', 'http_recall')
                  AND julianday(p.delivered_at) > julianday(t.at)
                  AND (t.next_at IS NULL OR julianday(p.delivered_at) < julianday(t.next_at))
                  AND julianday(p.delivered_at) <= julianday(t.at) + ? / 24.0) AS pulled
  FROM t
 WHERE t.at <= ? AND t.recalled = 1
"""


def labelled(conn: sqlite3.Connection, profile_id: str, *,
             now: datetime | None = None) -> list[Labelled]:
    """The calibration sample. Missing tables mean no sample."""
    now = now or datetime.now(UTC)
    since = (now - timedelta(days=WINDOW_DAYS)).isoformat()
    closed = (now - timedelta(hours=OUTCOME_HOURS)).isoformat()
    try:
        rows = conn.execute(_LABELLED, (profile_id, since, profile_id, OUTCOME_HOURS,
                                        profile_id, OUTCOME_HOURS, closed)).fetchall()
    except sqlite3.Error:
        return []
    out = []
    for rule_, s, p, used, pulled in rows:
        p = float(p or 0)
        if p <= 0:
            continue
        out.append(Labelled(rule_, float(s or 0), bool(used or pulled), 1.0 / p))
    return out


class Gate:
    """The daemon's need gate: one cached threshold per profile."""

    def __init__(self, slm_home: Path, db_path: Path) -> None:
        self._home = Path(slm_home)
        self._db = Path(db_path)
        self._cache: dict[str, tuple[float, float | None]] = {}
        self._lock = threading.Lock()

    def threshold(self, profile_id: str) -> float | None:
        now = time.monotonic()
        with self._lock:
            hit = self._cache.get(profile_id)
            if hit is not None and now - hit[0] < CACHE_S:
                return hit[1]
        lam = None
        try:
            from superlocalmemory.truth.invariants import open_read_only

            conn = open_read_only(str(self._db))
            try:
                lam = calibrate(labelled(conn, profile_id))
            finally:
                conn.close()
        except Exception as exc:  # noqa: BLE001 - no threshold: the score skips nothing
            logger.info("need-gate threshold unavailable: %s", exc)
        with self._lock:
            self._cache[profile_id] = (now, lam)
        return lam

    def _key(self) -> bytes | None:
        from superlocalmemory.core.receipt_key import KEY_FILE

        try:
            key = (self._home / "keys" / KEY_FILE).read_bytes()
        except OSError:
            return None
        return key if len(key) >= 32 else None

    def decide(self, prompt: str, profile_id: str) -> Decision | None:
        """This turn's decision, or None when per-prompt recall is off.

        ``delivery.need_gate`` (D-47): ``off`` (the default, so 4.2.30 ships
        dark) and the kill switch mean no per-prompt recall at all, as before
        4.2.30; ``shadow`` recalls every prompt the rules let through and logs
        the threshold it would use; ``on`` skips by score below it."""
        state = flag_state(self._home)
        if state is None:
            return None
        lam = self.threshold(profile_id)
        d = decide(prompt, threshold=lam if state == "on" else None, key=self._key())
        if state != "on" and lam is not None:
            d = Decision(d.turn_id, d.recall, d.rule, d.score, lam, d.explored, d.propensity,
                         d.at, acting=False)
        return d

    def record(self, decision: Decision, *, profile_id: str, session_id: str | None,
               prompt: str, recall_query_id: str | None = None) -> None:
        from superlocalmemory.outcomes.usage import query_digest

        log_later(self._db, decision, profile_id=profile_id, session_id=session_id,
                  prompt_digest=query_digest(self._home, prompt),
                  recall_query_id=recall_query_id)


_SHOWN = """
SELECT DISTINCT r.fact_id FROM injection_deliveries d
  JOIN injection_receipts r ON r.delivery_id = d.delivery_id
 WHERE d.profile_id = ? AND d.session_id = ? AND r.delivered = 1
"""


def shown_in_session(db_path: Path, profile_id: str, session_id: str | None) -> frozenset[str]:
    """Fact ids already delivered to this conversation (any path). Read-only;
    an unidentified session has no history, so nothing is filtered."""
    from superlocalmemory.core.session_identity import is_conversation
    from superlocalmemory.truth.invariants import open_read_only

    if not is_conversation(session_id, profile_id):
        return frozenset()
    conn = open_read_only(str(db_path))
    try:
        return frozenset(r[0] for r in conn.execute(_SHOWN, (profile_id, session_id)))
    except sqlite3.OperationalError:
        return frozenset()
    finally:
        conn.close()


def report(conn: sqlite3.Connection, profile_id: str, *, now: datetime | None = None) -> dict:
    """The gate's state for the Memory Health report: counts, the threshold it
    would use now, the rules' measured miss bound, and the caveat."""
    now = now or datetime.now(UTC)
    since = (now - timedelta(days=WINDOW_DAYS)).isoformat()
    try:
        turns, skipped, explored = conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(1 - recalled), 0), COALESCE(SUM(explored), 0)"
            " FROM need_turns WHERE profile_id = ? AND at >= ?", (profile_id, since)).fetchone()
    except sqlite3.Error:
        return {"turns": None}
    rows = labelled(conn, profile_id, now=now)
    return {
        "turns": int(turns), "skipped": int(skipped), "explored": int(explored),
        "labelled": len(rows), "alpha": NEED_ALPHA, "explore": EXPLORE,
        "threshold": calibrate(rows),
        "rules_bound": round(bound(rows, None), 4) if rows else None,
        "caveat": "needed = detected use or an explicit recall; detection misses some uses",
    }


def describe(section: dict) -> Iterable[str]:
    if not section or section.get("turns") is None:
        return ["Need gate: no per-prompt turns logged yet."]
    lam = section.get("threshold")
    lines = [f"Need gate (last {WINDOW_DAYS} days): {section['turns']} prompts, "
             f"{section['skipped']} skipped, {section['explored']} recalled to learn, "
             f"{section['labelled']} with a known outcome."]
    if lam is None:
        lines.append(f"  The score skips nothing yet: no threshold keeps needed turns skipped "
                     f"under {NEED_ALPHA:.0%} with the labels so far (rules still skip obvious turns).")
    else:
        lines.append(f"  Skipping prompts scored under {lam:.3f}; needed turns skipped "
                     f"<= {NEED_ALPHA:.0%} (conformal bound).")
    if section.get("rules_bound") is not None:
        lines.append(f"  Rules alone: needed turns skipped <= {section['rules_bound']:.1%}.")
    lines.append(f"  ({section['caveat']}.)")
    return lines


__all__ = ["CACHE_S", "EXPLORE", "NEED_ALPHA", "OUTCOME_HOURS", "RULES", "WINDOW_DAYS",
           "Decision", "Gate", "Labelled", "bound", "calibrate", "decide", "describe",
           "labelled", "log", "log_later", "report", "rule", "score", "shown_in_session"]
