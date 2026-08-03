# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V4 | https://qualixar.com | https://varunpratap.com

"""Temporal utility helpers for bi-temporal retrieval (Phase 4b).

STORED TIMESTAMP FORMAT (empirically verified, Phase 4b):
    The fact_temporal_validity table stores system_expired_at and valid_until
    via datetime.now(timezone.utc).isoformat() which produces:
        "YYYY-MM-DDTHH:MM:SS.microseconds+00:00"  (Python isoformat, +00:00 suffix)

    normalize_as_of() outputs "YYYY-MM-DDTHH:MM:SS+00:00" (no microseconds, same
    +00:00 suffix) so that SQL string comparisons ("system_expired_at <= as_of")
    are lexicographically correct:

        "2026-03-01T00:00:00.123456+00:00" <= "2026-06-01T00:00:00+00:00"
        → first 10 chars "2026-03-01" < "2026-06-01" → TRUE  ✓

    At sub-second boundary (same second, stored has micros):
        "2024-01-01T12:00:00.123456+00:00" <= "2024-01-01T12:00:00+00:00"
        → at pos 19: '.' (46) > '+' (43) → FALSE
        Interpretation: stored supersession is 0.123456s AFTER as_of → fact
        was still valid at that exact second. Semantically correct.

    Contrast with SQLite strftime('%Y-%m-%dT%H:%M:%SZ', 'now') → "Z" suffix.
    If as_of used "Z" suffix and stored uses "+00:00":
        "2024-01-01T12:00:00.123456+00:00" <= "2024-01-01T12:00:00Z"
        → at pos 19: '.' (46) < 'Z' (90) → TRUE
        This would WRONGLY demote a fact whose supersession happened 0.123456s
        AFTER as_of. Using "+00:00" avoids this error.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional


def normalize_as_of(s: object) -> Optional[str]:
    """Parse and normalize an as_of timestamp string to UTC ISO 8601.

    Accepts:
      - "2024-01-01T12:00:00Z"           → "2024-01-01T12:00:00+00:00"
      - "2024-01-01T12:00:00+05:30"      → "2024-01-01T06:30:00+00:00"
      - "2024-01-01"                      → "2024-01-01T00:00:00+00:00"
      - "2024-01-01T12:00:00"            → "2024-01-01T12:00:00+00:00" (naive → UTC)
      - "2024-01-01T12:00:00+00:00"      → "2024-01-01T12:00:00+00:00"

    Returns None (not empty string) on invalid input so callers can
    distinguish "no as_of" from "bad as_of" and reject or ignore accordingly.

    Output format: "YYYY-MM-DDTHH:MM:SS+00:00"
        - No microseconds (user input seldom has sub-second precision)
        - +00:00 suffix matches the format produced by datetime.now(UTC).isoformat()
          (the stored format in fact_temporal_validity), ensuring SQL string
          comparisons are correct (see module docstring for analysis).

    Requirements:
        Python >=3.11 (fromisoformat handles "Z" suffix natively). The
        fallback .replace("Z", "+00:00") is kept for defensive compatibility.

    Args:
        s: Raw as_of string from user/HTTP/MCP/CLI input. May be None or
           non-string; both return None (treated as "no as_of" by callers).

    Returns:
        Normalized UTC string "YYYY-MM-DDTHH:MM:SS+00:00", or None on error.
    """
    if not s or not isinstance(s, str):
        return None
    s = s.strip()
    if not s:
        return None

    # Try ISO 8601 parse. Python 3.11 handles "Z" natively; the replace()
    # makes this safe on any 3.x that might be used in tests.
    for candidate in (s, s.replace("Z", "+00:00")):
        try:
            dt = datetime.fromisoformat(candidate)
            # Naive datetime → assume UTC (documented assumption).
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = dt.astimezone(timezone.utc)
            return dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")
        except ValueError:
            continue

    # Date-only "YYYY-MM-DD" — treat as midnight UTC.
    try:
        dt = datetime.strptime(s[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
        return dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")
    except ValueError:
        return None


__all__ = ("normalize_as_of",)
