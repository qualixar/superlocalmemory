# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Pure, stdlib-only extraction of structured facets from text.

Every pattern is linear (no nested quantifiers) and only the first
``MAX_SCAN_CHARS`` characters are scanned, so the cost per save is small and
bounded. URLs are kept without credentials, query strings or fragments, so a
facet never repeats a token carried in a link.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import date
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)

EXTRACTED_KEY = "_slm_extracted"
MAX_SCAN_CHARS = 24_000
MAX_ITEMS = 16
MAX_ITEM_CHARS = 200

_URL = re.compile(r"https?://[^\s<>\"'`]+")
_TRAIL = ".,;:!?)]}'\""
_PATH = re.compile(
    r"(?<![\w/.:~\\-])/(?:[\w.@+-]+/)+[\w.@+-]+"
    r"|(?<![\w/~])~/[\w.@+-]+(?:/[\w.@+-]+)*"
    r"|(?<!\w)[A-Za-z]:\\(?:[^\\/:*?\"<>|\s]+\\)*[^\\/:*?\"<>|\s]+"
)
_DATE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_VERSION = re.compile(
    r"(?<![\w.\-/])v?\d{1,4}\.\d{1,4}(?:\.\d{1,4})?"
    r"(?:-[A-Za-z][0-9A-Za-z]*(?:\.[0-9A-Za-z]+)*)?(?![\w]|\.\d|-\d)"
)
_BACKTICK = re.compile(r"`([^`\n]{1,120})`")
_IDENT = re.compile(r"[A-Za-z_]\w*(?:(?:\.|::)[A-Za-z_]\w*)*")
_CODE_SHAPE = re.compile(r"[_.]|::|[a-z][A-Z]")
_HASHTAG = re.compile(r"(?<![\w&#/])#([A-Za-z][\w/-]{0,48})")
_HEX_COLOUR = re.compile(r"(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})")


@dataclass(frozen=True)
class Extracted:
    """Facets found in one text; each is deduplicated, ordered and bounded."""

    urls: tuple[str, ...] = ()
    paths: tuple[str, ...] = ()
    dates: tuple[str, ...] = ()
    versions: tuple[str, ...] = ()
    code_ids: tuple[str, ...] = ()
    hashtags: tuple[str, ...] = ()

    def is_empty(self) -> bool:
        return not (self.urls or self.paths or self.dates
                    or self.versions or self.code_ids or self.hashtags)

    def as_metadata(self) -> dict[str, list[str]]:
        fields = (("urls", self.urls), ("paths", self.paths), ("dates", self.dates),
                  ("versions", self.versions), ("code_ids", self.code_ids),
                  ("hashtags", self.hashtags))
        return {name: list(values) for name, values in fields if values}


def _bounded(items) -> tuple[str, ...]:
    seen: dict[str, None] = {}
    for item in items:
        if item and len(item) <= MAX_ITEM_CHARS and item not in seen:
            seen[item] = None
            if len(seen) == MAX_ITEMS:
                break
    return tuple(seen)


def _clean_url(raw: str) -> str | None:
    while raw and raw[-1] in _TRAIL:
        if raw[-1] == ")" and raw.count("(") >= raw.count(")"):
            break
        raw = raw[:-1]
    try:
        parts = urlsplit(raw)
        host = (parts.hostname or "").lower()
        port = parts.port
    except ValueError:
        return None
    if parts.scheme not in ("http", "https") or not host:
        return None
    netloc = f"[{host}]" if ":" in host else host
    if port:
        netloc += f":{port}"
    return f"{parts.scheme}://{netloc}{parts.path}"


def _valid_date(match: re.Match) -> str | None:
    try:
        date(int(match[1]), int(match[2]), int(match[3]))
    except ValueError:
        return None
    return match[0]


def _code_ids(text: str):
    for match in _BACKTICK.finditer(text):
        token = match[1]
        if _IDENT.fullmatch(token) and _CODE_SHAPE.search(token):
            yield token


def _hashtags(text: str):
    for match in _HASHTAG.finditer(text):
        tag = match[1].rstrip("/-")
        if tag and not _HEX_COLOUR.fullmatch(tag):
            yield tag.lower()


def extract_deterministic(text: str) -> Extracted:
    """Extract facets from ``text``. Pure; raises ``TypeError`` for non-str."""
    if not isinstance(text, str):
        raise TypeError("text must be a str")
    text = text[:MAX_SCAN_CHARS]
    urls = _bounded(_clean_url(m[0]) for m in _URL.finditer(text))
    rest = _URL.sub(" ", text)
    return Extracted(
        urls=urls,
        paths=_bounded(m[0].rstrip(".,;:") for m in _PATH.finditer(rest)),
        dates=_bounded(_valid_date(m) for m in _DATE.finditer(rest)),
        versions=_bounded(m[0] for m in _VERSION.finditer(rest)),
        code_ids=_bounded(_code_ids(rest)),
        hashtags=_bounded(_hashtags(rest)),
    )


def add_extracted(metadata: dict, text: str) -> None:
    """Set ``metadata[EXTRACTED_KEY]`` from ``text`` when any facet is found.

    Replaces any existing value and never raises: a facet must not fail a save.
    """
    try:
        metadata.pop(EXTRACTED_KEY, None)
        found = extract_deterministic(text)
        if not found.is_empty():
            metadata[EXTRACTED_KEY] = found.as_metadata()
    except Exception as exc:  # noqa: BLE001 - facets are best effort
        metadata.pop(EXTRACTED_KEY, None)
        logger.debug("facet extraction skipped: %s", type(exc).__name__)
