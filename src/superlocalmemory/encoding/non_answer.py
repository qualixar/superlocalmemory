# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later
"""Conservative receipt/stub detection, independent of fact length.

A fact is a stub only when the WHOLE text is machine bookkeeping: a label
followed by a machine-shaped id (8+ hex characters with a letter a-f, a canonical UUID or a
ULID; a plain number needs 12+ digits, so a receipt number or a date stamp is
never one), or session boilerplate with its timestamp. Anything a person could
mean ("use port 8765", "memory id: grandma-recipe") stays answer material.
A user's decision ("this is an answer") is stored and always wins.

The hooks' own lines are stubs too, matched only as the WHOLE text and in the
writer's exact shape (hooks/hook_handlers.py and the agents' slm-checkpoint):
the Stop hook's "[project] session ended <date> <time>" with its optional
"| files: ..." tail, the Write/Edit hook's "File changed: <name>", a bare
"files: a.py, b.md" list (what a stored summary splits into) and the
Codex/agents "[agent] session ended (reason) at <date> <time> in <project>."
A summary that carries commits ("| recent: ...") or the agent's last action is
real content and is not matched. One that adds only the branch name and the
``git diff --stat`` total ("| branch: main | uncommitted: 2 files changed, ...")
is still bookkeeping (4.2.16: 100 such rows on one store, three of them shown
at a session start), and so are the copies ingestion derives from it: the bare
diff total and "the user's session ended <date> <time>".

Measured on one real store, about 1,056 of 5,443 facts written in 7 days were
such lines, and they answered "what am I working on?" with bookkeeping.
"""

from __future__ import annotations

import re

_EMPTY = re.compile(r'^[\s\[\]{}(),;:]*$')
_DATE = r'\d{4}-\d{2}-\d{2}(?:[T ]\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?)?'
_SESSION = re.compile(r'^(?:\[\]\s*)?(?:the\s+)?(?:user[\'\u2019]s\s+)?session\s+(?:ended|closed)'
                      r'(?:\s+at)?\s+' + _DATE + r'[.!]?$', re.I)
# The last line of ``git diff --stat``: "2 files changed, 0 insertions(+), 0 deletions(-)".
_DIFF_STAT = (r'\d+\s+files?\s+changed(?:,\s*\d+\s+insertions?\(\+\))?'
              r'(?:,\s*\d+\s+deletions?\(-\))?')
_GIT_STAT = re.compile(r'^(?:uncommitted:\s*)?' + _DIFF_STAT + r'$', re.I)
# "[Agentic_official] session ended 2026-10-13 09:54 | files: NOTICE, formats.md",
# with the Stop hook's optional "| branch: <name>" and "| uncommitted: <diff
# total>" in the order it writes them. A "| recent:" segment (commits) is real
# content, so it is not in the pattern. The "[project]" is optional: ingestion
# also derives the same line without it, and that copy was unflagged; it is
# "[]" when the project folder's name is empty.
_HOOK_SESSION = re.compile(
    r'^(?:\[[^\]\n]{0,200}\]\s*)?session\s+ended\s+' + _DATE
    + r'(?:\s*\|\s*branch:\s*[^\s|]+)?'
    + r'(?:\s*\|\s*uncommitted:\s*' + _DIFF_STAT + r')?'
    + r'(?:\s*\|\s*files:\s*[^|\n]+)?$', re.I)
# "[codex] session ended (exit) at 2026-10-13 11:01 in my project." plus the id
# segments the checkpoint appends; a "last_action:" segment is real content.
# The "[agent]" is optional for the same reason as above: ingestion derives
# "session ended (exit) at 2026-11-07 13:49 in config." without it (4.2.14).
_AGENT_CHECKPOINT = re.compile(
    r'^(?:\[[^\]\n]{1,200}\]\s+)?session\s+ended\s+\([^)\n]{0,100}\)\s+at\s+' + _DATE
    + r'\s+in\s+[^|\n]{1,300}?\.(?:\s*\|\s*(?:transcript|session_id):\s*[^|\n]+)*$', re.I)
# The hook writes a basename; a name with a space is left as an answer (harmless).
_BASENAME = r'[\w@+~#%=.-]+'
_FILE_CHANGED = re.compile(r'^File changed:\s*' + _BASENAME + '$')
_FILE_LIST = re.compile(r'^files:\s*(?=[^\n]*\.\w)' + _BASENAME + r'(?:,\s*' + _BASENAME + ')*$')
_MACHINE_ID = (r'(?:[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}'
               r'|(?=[0-9]*[a-f])[0-9a-f]{8,}'  # hex: a letter a-f, so not a plain number
               r'|[0-9]{12,}'
               r'|[0-7][0-9a-hjkmnp-tv-z]{25})')
_RECEIPT = re.compile(r'^(?:receipt|operation|session|fact|memory|run)(?:[ _-]?id)?'
                      r'\s*[:=]\s*' + _MACHINE_ID + r'[.!]?$', re.I)
_USER_COLUMN = 'non_answer_user'


def is_empty_structure(text: str) -> bool:
    """Only brackets, braces and separators: nothing to remember at all."""
    return bool(_EMPTY.fullmatch(text or ''))


def is_stub(text: str) -> bool:
    value = (text or '').strip()
    return bool(value and (_EMPTY.fullmatch(value) or _SESSION.fullmatch(value)
                           or _RECEIPT.fullmatch(value) or _hook_line(value)))


def _hook_line(value: str) -> bool:
    return bool(_HOOK_SESSION.fullmatch(value) or _AGENT_CHECKPOINT.fullmatch(value)
                or _FILE_CHANGED.fullmatch(value) or _FILE_LIST.fullmatch(value)
                or _GIT_STAT.fullmatch(value))


def _columns(db) -> frozenset:
    cached = getattr(db, '_non_answer_columns', None)
    if cached:
        return cached
    found = frozenset(r['name'] for r in db.execute('PRAGMA table_info(atomic_facts)')
                      if r['name'] in ('non_answer', _USER_COLUMN))
    if found == {'non_answer', _USER_COLUMN}:
        try:  # cached once complete; re-probed while a later migration is pending
            db._non_answer_columns = found
        except AttributeError:
            pass
    return found


def has_marker(db) -> bool:
    return 'non_answer' in _columns(db)


def user_decided(db, fact_id, profile_id) -> bool:
    """True when the user already said whether this fact is an answer."""
    if _USER_COLUMN not in _columns(db):
        return False
    rows = db.execute(f'SELECT {_USER_COLUMN} FROM atomic_facts WHERE fact_id=? '
                      'AND profile_id=?', (fact_id, profile_id))
    return bool(rows) and rows[0][_USER_COLUMN] is not None


def mark(db, fact_id, profile_id, content):
    """Flag a stub automatically, never over the user's own decision."""
    columns = _columns(db)
    if 'non_answer' not in columns:
        return
    guard = f' AND {_USER_COLUMN} IS NULL' if _USER_COLUMN in columns else ''
    db.execute(f'UPDATE atomic_facts SET non_answer=? WHERE fact_id=? AND profile_id=?{guard}',
               (int(is_stub(content)), fact_id, profile_id))


def user_unmark(db, fact_id, profile_id):
    """The user says this fact is an answer: clear the flag and remember why."""
    columns = _columns(db)
    if 'non_answer' not in columns:
        return
    extra = f', {_USER_COLUMN}=0' if _USER_COLUMN in columns else ''
    db.execute(f'UPDATE atomic_facts SET non_answer=0{extra} WHERE fact_id=? AND profile_id=?',
               (fact_id, profile_id))


def user_unmark_on_connection(connection, fact_id, profile_id):
    """``user_unmark`` inside an open transaction (a correction being applied)."""
    columns = {row[1] for row in connection.execute('PRAGMA table_info(atomic_facts)')}
    if 'non_answer' not in columns:
        return
    extra = f', {_USER_COLUMN}=0' if _USER_COLUMN in columns else ''
    connection.execute(
        f'UPDATE atomic_facts SET non_answer=0{extra} WHERE fact_id=? AND profile_id=?',
        (fact_id, profile_id))
