# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The need gate: should this prompt get memories at all? (WS-4, M-16)

Per-prompt recall puts memories into the agent's context before it answers.
Many prompts need none ("yes", "2+2", "translate this"), and every needless
injection costs the agent attention and tokens. The gate decides per turn,
in two steps:

1. **Rules** skip the obvious turns (an acknowledgement, a greeting, pure
   arithmetic, a translation request, a slash command). They are fixed, so
   they cannot adapt to look good; each rule's miss share is measured, and a
   rule whose measured bound exceeds ``NEED_ALPHA`` stops skipping (A-18).
   "continue", "proceed", "resume" and "go ahead" are not acknowledgements: they ask
   for the last context, so they are never skipped by a rule.
2. **A score** ranks the remaining prompts by how likely memory is to help
   (cues such as "last time", "we decided", names, paths, questions). A
   prompt is skipped when its score is below a threshold ``lam``.

**Estimand (M-16, one sentence).** The gate bounds the *share of the turns that
needed memory which it skips*, ``R(lam) = P(skipped | needed)``, at
``NEED_ALPHA``. It does not bound ``P(needed and skipped)`` over all turns: when
few turns need memory that quantity is small even for a gate that skips
everything (A-02).

**The threshold is never hand-picked.** It is set by weighted conformal risk
control (Angelopoulos et al., 2022) over the logged turns that *needed* memory,
for the loss "this needed turn was skipped" (0/1, monotone in ``lam``):

    lam = the largest grid score with
          (sum_{i needed} w_i * skipped_i(lam) + W) / (sum_{i needed} w_i + W) <= NEED_ALPHA

where ``skipped_i`` is "an active rule fires or the score is below lam". The
calibration sample is the labelled needed turns only, so the guarantee is
conditional on need.

Labels exist only for turns that were recalled, so a skipped turn is recalled
anyway with logged probability ``EXPLORE`` and counts with weight
``1 / EXPLORE`` (importance weighting, Tibshirani et al., 2019); every row has
``w_i = 1 / P(recalled)`` under the policy at its time. The unseen next turn that
the gate would skip has ``P(recalled) = EXPLORE``, the smallest probability any
turn can have, so ``W = 1 / EXPLORE`` is its largest possible weight, not the
largest one observed (A-03). With ``W = 20`` and ``alpha = 0.1`` nothing can be
skipped by score until the labelled needed turns weigh at least
``W (1 - alpha) / alpha = 180``; with fewer, or with a low need rate (8% of 1,000
turns is 80), the score skips nothing: the gate starts from "recall every prompt
the rules let through" and earns the right to skip.

Statement made, in these words: *approximately* ``P(skipped | needed) <= alpha``
for a new needed turn, exchangeable with the calibration window (30 days). It is
approximate, not exact: the threshold moves as labels arrive, so rows logged at
different times carry different weights (each exact for its own era), and the
finite-sample theorem assumes weights fixed in advance. It is re-calibrated on a
rolling window.

**What "needed" means.** A turn "needed" memory when an agent used a memory
delivered for it within the outcome window (the M-12 ``used`` signals) or pulled
memory itself (an explicit recall in the same conversation before its next
prompt). Usage detection finds some uses, not all (M-20), so "needed" is a lower
bound and the bound is on the share of *detected* need.

**Rules** are measured the same way, per rule: ``bound_r`` is the share of needed
turns that rule r would skip, over labelled turns the rule matched (rule-skipped
turns reach the sample only through exploration, or while the rule is switched off).
With no labelled turn for a rule, or too few needed turns to ever show ``bound_r <=
alpha``, the rule is "not measured yet" and the report says so, claiming nothing.

**Modes (``delivery.need_gate``).** ``off``: no per-prompt recall. ``shadow``:
per-prompt recall runs, the rules skip and explore as in ``on``, but the score
never skips and the threshold is only logged; memories ARE injected on turns the
rules let through, because labels need real deliveries (this differs from the
glossary sense of shadow, "changes nothing"; see the report). ``on``: rules and
score skip.
"""

from __future__ import annotations

import logging
import math
import os
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

#: Policy (awaits the phase review, D-47): the share of the turns that needed
#: memory which the gate may skip, P(skipped | needed), at the conformal level.
NEED_ALPHA = 0.10
#: The logged chance that a turn the gate would skip is recalled anyway. Any
#: value keeps the estimate unbiased; it sets only how fast labels arrive.
EXPLORE = 0.05
#: The largest weight any turn can have, 1 / P(recalled), reached by a turn the gate
#: skips and the exploration draw recalled. The unseen next turn is given this (A-03).
W_TEST = 1.0 / EXPLORE
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

# "go ahead", "continue", "proceed", "resume" are NOT here (A-18): they ask the agent to carry
# on, which is exactly the prompt that needs the last session's context. A bare "go" stays: it
# answers a question ("shall I run it?") and the hook's own acknowledgement list has it too.
_ACK_WORDS = (r"(?:yes|no|ok|okay|approved|thanks|thank you|go|sure|yep|nope|done|"
              r"y|n|cool|got it|right|correct|perfect|great|nice|please|\d+)")
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


def _skipped(rule_: str | None, score_: float, lam: float | None,
             suspended: frozenset[str] = frozenset()) -> bool:
    """Would the gate skip this turn: an active (not switched-off) rule fires, or the
    score is below the threshold."""
    return (rule_ is not None and rule_ not in suspended) or (lam is not None and score_ < lam)


@dataclass(frozen=True)
class Labelled:
    """One recalled turn whose outcome window closed."""

    rule: str | None
    score: float
    needed: bool
    weight: float          # 1 / P(this turn was recalled)


def min_needed_weight(alpha: float = NEED_ALPHA) -> float:
    """Needed-turn weight below which no bound <= alpha exists even with no miss:
    W / (n + W) <= alpha  <=>  n >= W (1 - alpha) / alpha (180 at the defaults)."""
    return W_TEST * (1.0 - alpha) / alpha


def needed_weight(rows: Sequence[Labelled]) -> float:
    """Total weight of the labelled turns that needed memory."""
    return sum(r.weight for r in rows if r.needed)


def bound(rows: Sequence[Labelled], lam: float | None,
          suspended: frozenset[str] = frozenset()) -> float:
    """Conformal upper bound on P(skipped | needed) at ``lam`` (A-02).

    Over the labelled turns that needed memory only: skipped weight plus the
    unseen turn's largest possible weight ``W_TEST``, over total weight plus ``W_TEST``.
    With no needed turn labelled it is 1."""
    needed = [r for r in rows if r.needed]
    loss = sum(r.weight for r in needed if _skipped(r.rule, r.score, lam, suspended))
    return (loss + W_TEST) / (sum(r.weight for r in needed) + W_TEST)


def calibrate(rows: Sequence[Labelled], alpha: float = NEED_ALPHA,
              suspended: frozenset[str] = frozenset()) -> float | None:
    """The largest score threshold whose bound is within ``alpha``, or None
    (the score skips nothing). The bound only grows with ``lam``, so the scores are
    walked upward in one sorted pass (running needed weight) and stop at the first
    failure. The active rules' misses are in ``loss`` from the start, so a rule set that
    already exceeds ``alpha`` fails at the first score level and nothing is skipped."""
    needed_total = needed_weight(rows) + W_TEST
    loss = sum(r.weight for r in rows if r.needed and _skipped(r.rule, r.score, None, suspended))
    open_rows = sorted((r for r in rows if not _skipped(r.rule, r.score, None, suspended)),
                       key=lambda r: r.score)
    best = None
    i = 0
    while i < len(open_rows):
        level = open_rows[i].score
        while i < len(open_rows) and open_rows[i].score == level:
            loss += open_rows[i].weight if open_rows[i].needed else 0.0
            i += 1
        if (loss + W_TEST) / needed_total > alpha:  # skipping every score <= level is too costly
            break
        best = math.nextafter(level, math.inf)
    return best


@dataclass(frozen=True)
class RuleBound:
    """One rule's measured miss share: the share of needed turns it would skip."""

    rule: str
    labelled: int                 # labelled turns the rule matched
    bound: float | None           # None: not measured yet
    state: str                    # not_measured | ok | exceeded


def rule_bounds(rows: Sequence[Labelled], alpha: float = NEED_ALPHA) -> dict[str, RuleBound]:
    """Per rule: bound on P(rule fires | needed), from labelled turns it matched.

    ``not_measured`` with no labelled turn for the rule, or while the needed turns
    weigh too little for any bound under ``alpha`` to exist (A-03); ``exceeded``
    switches the rule off (A-18)."""
    needed_total = needed_weight(rows)
    out = {}
    for name in RULES:
        matched = [r for r in rows if r.rule == name]
        if not matched:
            continue
        loss = sum(r.weight for r in matched if r.needed)
        value = (loss + W_TEST) / (needed_total + W_TEST)
        if needed_total < min_needed_weight(alpha):
            out[name] = RuleBound(name, len(matched), None, "not_measured")
        else:
            out[name] = RuleBound(name, len(matched), value,
                                  "ok" if value <= alpha else "exceeded")
    return out


def suspended_rules(rows: Sequence[Labelled], alpha: float = NEED_ALPHA) -> frozenset[str]:
    """Rules whose measured bound is above ``alpha``: they stop skipping."""
    return frozenset(n for n, b in rule_bounds(rows, alpha).items() if b.state == "exceeded")


@dataclass(frozen=True)
class Decision:
    turn_id: str
    recall: bool
    rule: str | None
    score: float
    threshold: float | None
    explored: bool
    #: P(what happened) for this turn under the logged policy: 1.0 when the gate
    #: let it through; when it skipped the turn, EXPLORE if the exploration draw
    #: recalled it anyway and 1 - EXPLORE (0.95) if the draw said no; 0.0 only when
    #: no draw could be made (no key), which keeps the turn out of every label.
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
           now: datetime | None = None,
           suspended: frozenset[str] = frozenset()) -> Decision:
    """One turn's decision. ``key`` is the receipt key (no key: no exploration).

    A rule in ``suspended`` (its measured miss bound is above alpha) does not skip:
    the turn is recalled with probability 1 and logged with the rule that matched, so
    the rule keeps being measured and can be reinstated. The score is always computed
    and logged, a rule turn's included."""
    at = now or datetime.now(UTC)
    turn_id = f"turn_{uuid.uuid4().hex}"
    rule_ = rule(prompt)
    s = score(prompt)
    if not _skipped(rule_, s, threshold, suspended):
        return Decision(turn_id, True, rule_, s, threshold, False, 1.0, at)
    u = _draw(key, turn_id)
    if u is None:
        return Decision(turn_id, False, rule_, s, threshold, False, 0.0, at)
    explored = u < EXPLORE
    return Decision(turn_id, explored, rule_, s, threshold, explored,
                    EXPLORE if explored else 1.0 - EXPLORE, at)


_INSERT = ("INSERT OR IGNORE INTO need_turns (turn_id, profile_id, session_id, at, prompt_digest,"
           " rule, score, threshold, acting, recalled, explored, propensity, recall_query_id)"
           " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)")


#: One line per prompt that could not be logged, next to the store. Appended with
#: O_APPEND, so counting a drop never waits for the database or its writer (A-19).
DROPPED_FILE = "need_turns_dropped.log"
#: Routing only: the most of the drop file read back (its newest lines).
_DROPPED_READ_BYTES = 1_000_000


def _note_drop(db_path: Path, profile_id: str, at: datetime) -> None:
    """Record that one turn was not logged. Never raises and never takes a lock."""
    try:
        fd = os.open(Path(db_path).parent / DROPPED_FILE,
                     os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        try:
            os.write(fd, f"{at.isoformat()}\t{profile_id}\n".encode())
        finally:
            os.close(fd)
    except OSError as exc:
        logger.info("need-gate drop not counted: %s", exc)


def dropped_since(db_path: Path | str, since: datetime, profile_id: str | None = None) -> int:
    """Turns dropped at or after ``since`` (all profiles unless one is named)."""
    try:
        path = Path(db_path).parent / DROPPED_FILE
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            fh.seek(max(fh.tell() - _DROPPED_READ_BYTES, 0))
            lines = fh.read().decode("utf-8", "replace").splitlines()
    except OSError:
        return 0
    cut = since.isoformat()
    count = 0
    for line in lines:
        stamp, _, who = line.partition("\t")
        if stamp >= cut and (profile_id is None or who == profile_id):
            count += 1
    return count


def log(db_path: Path, decision: Decision, *, profile_id: str, session_id: str | None,
        prompt_digest: str | None, recall_query_id: str | None = None) -> bool:
    """Store one turn. ``propensity`` is stored as P(what happened): P(recalled) for
    a turn the gate let through, EXPLORE or 1 - EXPLORE for a skipped one.

    Runs on its own thread (``log_later``), never on the hook path. A turn that cannot
    be written (the store is busy) is counted in ``DROPPED_FILE``, not lost silently;
    the report prints the count (A-19)."""
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
        _note_drop(Path(db_path), profile_id, decision.at)
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


@dataclass(frozen=True)
class Calibration:
    """What the labels say now: the score threshold and the rules switched off."""

    threshold: float | None = None
    suspended: frozenset[str] = frozenset()


class Gate:
    """The daemon's need gate: one cached calibration per profile."""

    def __init__(self, slm_home: Path, db_path: Path) -> None:
        self._home = Path(slm_home)
        self._db = Path(db_path)
        self._cache: dict[str, tuple[float, Calibration]] = {}
        self._lock = threading.Lock()

    def calibration(self, profile_id: str) -> Calibration:
        now = time.monotonic()
        with self._lock:
            hit = self._cache.get(profile_id)
            if hit is not None and now - hit[0] < CACHE_S:
                return hit[1]
        cal = Calibration()
        try:
            from superlocalmemory.truth.invariants import open_read_only

            conn = open_read_only(str(self._db))
            try:
                rows = labelled(conn, profile_id)
                off = suspended_rules(rows)
                cal = Calibration(calibrate(rows, suspended=off), off)
            finally:
                conn.close()
        except Exception as exc:  # noqa: BLE001 - no threshold: the score skips nothing
            logger.info("need-gate threshold unavailable: %s", exc)
        with self._lock:
            self._cache[profile_id] = (now, cal)
        return cal

    def threshold(self, profile_id: str) -> float | None:
        return self.calibration(profile_id).threshold

    def suspended(self, profile_id: str) -> frozenset[str]:
        return self.calibration(profile_id).suspended

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
        the threshold it would use (the score never skips: memories ARE injected,
        because labels need real deliveries); ``on`` skips by score below it."""
        state = flag_state(self._home)
        if state is None:
            return None
        lam = self.threshold(profile_id)
        d = decide(prompt, threshold=lam if state == "on" else None, key=self._key(),
                   suspended=self.suspended(profile_id))
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


ESTIMAND = ("the share of the turns that needed memory which the gate skips, "
            "P(skipped | needed)")


def _db_file(conn: sqlite3.Connection) -> Path | None:
    try:
        name = conn.execute("PRAGMA database_list").fetchone()[2]
    except (sqlite3.Error, TypeError, IndexError):
        return None
    return Path(name) if name else None


def report(conn: sqlite3.Connection, profile_id: str, *, now: datetime | None = None) -> dict:
    """The gate's state for the Memory Health report: counts, the threshold it
    would use now, each rule's measured bound, the drops, and the caveat."""
    now = now or datetime.now(UTC)
    since = (now - timedelta(days=WINDOW_DAYS)).isoformat()
    try:
        turns, skipped, explored = conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(1 - recalled), 0), COALESCE(SUM(explored), 0)"
            " FROM need_turns WHERE profile_id = ? AND at >= ?", (profile_id, since)).fetchone()
    except sqlite3.Error:
        return {"turns": None}
    rows = labelled(conn, profile_id, now=now)
    rules = rule_bounds(rows)
    off = frozenset(n for n, b in rules.items() if b.state == "exceeded")
    active = [b for b in rules.values() if b.state != "exceeded" and b.bound is not None]
    db = _db_file(conn)
    return {
        "turns": int(turns), "skipped": int(skipped), "explored": int(explored),
        "labelled": len(rows), "needed_labelled": sum(1 for r in rows if r.needed),
        "needed_weight": round(needed_weight(rows), 2),
        "min_needed_weight": round(min_needed_weight(), 2),
        "alpha": NEED_ALPHA, "explore": EXPLORE, "estimand": ESTIMAND,
        "threshold": calibrate(rows, suspended=off),
        "suspended_rules": sorted(off),
        # None = not measured yet: no rule-matched turn has a seen outcome, or too few
        # turns needed memory for any bound under alpha to exist (A-03).
        "rules_bound": (round(max(b.bound for b in active), 4) if active else None),
        "rules": {n: {"labelled": b.labelled, "bound": None if b.bound is None
                      else round(b.bound, 4), "state": b.state} for n, b in rules.items()},
        "turns_dropped": 0 if db is None else dropped_since(db, now - timedelta(days=WINDOW_DAYS),
                                                            profile_id),
        "caveat": "needed = detected use or an explicit recall; detection misses some uses",
    }


def describe(section: dict) -> Iterable[str]:
    if not section or section.get("turns") is None:
        return ["Need gate: no per-prompt turns logged yet."]
    lam = section.get("threshold")
    alpha = section.get("alpha", NEED_ALPHA)
    lines = [f"Need gate (last {WINDOW_DAYS} days): {section['turns']} prompts, "
             f"{section['skipped']} skipped, {section['explored']} recalled to learn, "
             f"{section['labelled']} with a known outcome, "
             f"{section.get('needed_labelled', 0)} of those needed memory."]
    if lam is None:
        lines.append(
            f"  The score skips nothing yet: it needs labelled turns that needed memory to "
            f"weigh at least {section.get('min_needed_weight', min_needed_weight()):.0f} "
            f"(now {section.get('needed_weight', 0):.0f}) before any threshold can keep the "
            f"share of needed turns it skips under {alpha:.0%} (rules still skip obvious turns).")
    else:
        lines.append(
            f"  Skipping prompts scored under {lam:.3f}. Among the turns that needed memory, "
            f"at most {alpha:.0%} are skipped (conformal bound on the share of needed turns "
            f"skipped, not on all turns).")
    rb = section.get("rules_bound")
    if rb is None:
        lines.append("  Rules: not measured yet (no rule-skipped prompt has had its outcome seen "
                     "in enough needed turns).")
    else:
        lines.append(f"  Rules: the share of needed turns a rule skips is at most {rb:.1%} "
                     f"(worst rule).")
    if section.get("suspended_rules"):
        lines.append("  Switched off because that bound is above "
                     f"{alpha:.0%}: {', '.join(section['suspended_rules'])}.")
    if section.get("turns_dropped"):
        lines.append(f"  {section['turns_dropped']} prompts could not be logged (the store was "
                     f"busy); the numbers above do not include them.")
    lines.append(f"  ({section['caveat']}.)")
    return lines


__all__ = ["CACHE_S", "DROPPED_FILE", "ESTIMAND", "EXPLORE", "NEED_ALPHA", "OUTCOME_HOURS",
           "RULES", "WINDOW_DAYS", "W_TEST", "Calibration", "Decision", "Gate", "Labelled",
           "RuleBound", "bound", "calibrate", "decide", "describe", "dropped_since", "labelled",
           "log", "log_later", "min_needed_weight", "needed_weight", "report", "rule",
           "rule_bounds", "score", "shown_in_session", "suspended_rules"]
