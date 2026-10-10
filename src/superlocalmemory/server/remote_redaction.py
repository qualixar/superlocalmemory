# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V4 | https://qualixar.com | https://varunpratap.com

"""Keep details of the SLM computer out of tool answers to remote callers.

A tool answer for a caller on this computer is unchanged. For a remote caller,
before the answer leaves the daemon:

* fields that describe the host (``base_dir``, ``db_path``, ``pid``, ``cwd``,
  ``env``, anything ending in ``_path``/``_dir``/``_root``/``_file`` ...) become
  :data:`REDACTED`;
* in every other string except memory text, absolute paths become
  :data:`HOST_PATH`, and the home directory, the SLM data folder and the
  account name are replaced too.

Memory text (``content`` and the like) is returned exactly as stored: it is the
user's data, and a path the user wrote down is not a host detail. A remote key
can read the full text of every memory in the one profile it is bound to
(:mod:`server.remote_profile_binding`); that binding is what limits exposure.

Diagnostics are not memory text. Inside an error, a traceback, a warning, a
note or a receipt - and in the whole of a failed tool result or a JSON-RPC
error - nothing is passed through as written: a host path in an exception
message is withheld like any other host detail.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path
from typing import Any

from superlocalmemory.media.upload_links import UPLOAD_BASE_URL

REDACTED = "[host detail withheld]"
HOST_PATH = "[host path]"

#: Field names that always describe the host, matched case-insensitively.
_HOST_KEYS = frozenset({
    "base_dir", "db_path", "data_dir", "data_root", "path", "paths", "file", "files",
    "filename", "cwd", "home", "home_dir", "pid", "ppid", "env", "environ",
    "environment", "executable", "python", "python_path", "sys_path", "hostname",
    "username", "user_name", "os_user", "socket", "log_file", "log_dir", "logs_dir",
    "install_dir", "repo_path", "project_path", "config_path", "argv", "cmdline",
})
_HOST_SUFFIXES = ("_path", "_dir", "_root", "_file", "_paths", "_dirs")
#: Fields that carry memory text, returned as written (outside diagnostics).
_CONTENT_KEYS = frozenset({
    "content", "text", "original_content", "fact", "memory", "summary_text", "query",
    "answer", "snippet", "excerpt", "observation", "title", "body",
})
#: Fields that carry diagnostics. Their whole subtree is redacted, memory-text
#: field names included: an exception message is not the user's data.
_DIAGNOSTIC_KEYS = frozenset({
    "error", "errors", "exception", "exceptions", "traceback", "stack", "stack_trace",
    "stderr", "stdout", "warning", "warnings", "detail", "details", "diagnostic",
    "diagnostics", "hint", "reason", "failure", "failures", "note", "notes",
    "receipt", "receipts", "message", "messages",
})
_DIAGNOSTIC_SUFFIXES = ("_error", "_errors", "_exception", "_traceback", "_warning",
                        "_warnings", "_note", "_notes", "_reason", "_receipt", "_receipts",
                        "_detail", "_details", "_message", "_hint")

_POSIX_PATH = re.compile(r"(?<![\w.~:/-])/(?:[^\s/\"'`<>|:;,()\[\]{}]+/)+[^\s/\"'`<>|:;,()\[\]{}]*")
_WINDOWS_PATH = re.compile(r"\b[A-Za-z]:\\(?:[^\s\\\"'<>|:;,]+\\)*[^\s\\\"'<>|:;,]*")
_TILDE_PATH = re.compile(r"(?<![\w])~/(?:[^\s\"'`<>|:;,()\[\]{}]+)")


@lru_cache(maxsize=1)
def _host_strings() -> tuple[str, ...]:
    values: list[str] = []
    try:
        values.append(str(Path.home()))
    except Exception:  # noqa: BLE001
        pass
    try:
        from superlocalmemory.infra.data_root import canonical_data_root

        values.append(str(canonical_data_root()))
    except Exception:  # noqa: BLE001
        pass
    try:
        from superlocalmemory.core.platform_utils import current_user_name

        user = current_user_name()
        if len(user) >= 3:
            values.append(user)
    except Exception:  # noqa: BLE001
        pass
    for name in ("HOSTNAME", "COMPUTERNAME"):
        if len(os.environ.get(name, "")) >= 3:
            values.append(os.environ[name])
    # Longest first so a data root inside home is replaced whole.
    return tuple(sorted({v for v in values if v}, key=len, reverse=True))


def clear_cache() -> None:
    _host_strings.cache_clear()


def _is_host_key(key: str) -> bool:
    lowered = key.lower()
    return lowered in _HOST_KEYS or lowered.endswith(_HOST_SUFFIXES)


def _is_diagnostic_key(key: str) -> bool:
    lowered = key.lower()
    return lowered in _DIAGNOSTIC_KEYS or lowered.endswith(_DIAGNOSTIC_SUFFIXES)


#: A one-time upload link made by ``media_upload_link``. It carries no host detail, and
#: rewriting a stretch of its token that happens to equal the account or computer name
#: would break it, so it is passed through whole.
_UPLOAD_LINK = re.compile(re.escape(UPLOAD_BASE_URL) + r"/u/[0-9a-f]{32}/[A-Za-z0-9_-]{43}")


def redact_text(text: str) -> str:
    """Host details in a non-memory string; an upload link in it is kept as it is."""
    pieces: list[str] = []
    last = 0
    for found in _UPLOAD_LINK.finditer(text):
        pieces += [_redact_plain(text[last:found.start()]), found.group(0)]
        last = found.end()
    pieces.append(_redact_plain(text[last:]))
    return "".join(pieces)


def _redact_plain(text: str) -> str:
    out = _WINDOWS_PATH.sub(HOST_PATH, text)
    out = _POSIX_PATH.sub(HOST_PATH, out)
    out = _TILDE_PATH.sub(HOST_PATH, out)
    for value in _host_strings():
        out = out.replace(value, HOST_PATH if os.sep in value else REDACTED)
    return out


def redact_value(value: Any, key: str | None = None, *, diagnostic: bool = False) -> Any:
    """A redacted copy (inputs are never mutated).

    ``diagnostic`` is set inside an error/receipt subtree, where memory-text
    field names lose their pass-through.
    """
    if key is not None:
        if _is_diagnostic_key(key):
            diagnostic = True
        elif key.lower() in _CONTENT_KEYS and not diagnostic:
            return value
        if _is_host_key(key):
            return REDACTED if value not in (None, "", [], {}) else value
    if isinstance(value, dict):
        return {k: redact_value(v, str(k), diagnostic=diagnostic) for k, v in value.items()}
    if isinstance(value, list):
        return [redact_value(v, key, diagnostic=diagnostic) for v in value]
    if isinstance(value, str):
        return redact_text(value)
    return value


def _redact_block_text(text: str, diagnostic: bool) -> str:
    try:
        parsed = json.loads(text)
    except ValueError:
        return redact_text(text)
    if isinstance(parsed, (dict, list)):
        return json.dumps(redact_value(parsed, diagnostic=diagnostic),
                          separators=(",", ":"), ensure_ascii=False, default=str)
    return redact_text(text)


def redact_tool_result(result: dict[str, Any]) -> dict[str, Any]:
    """A ``tools/call`` result with host details removed.

    A failed result (``isError``) is diagnostics throughout.
    """
    diagnostic = result.get("isError") is True
    out = dict(result)
    blocks: Iterable[Any] = result.get("content") or []
    out["content"] = [
        dict(b, text=_redact_block_text(b["text"], diagnostic))
        if isinstance(b, dict) and isinstance(b.get("text"), str) else b
        for b in blocks
    ]
    if isinstance(result.get("structuredContent"), (dict, list)):
        out["structuredContent"] = redact_value(result["structuredContent"],
                                                diagnostic=diagnostic)
    return out


def redact_rpc_error(error: Any) -> Any:
    """A JSON-RPC ``error`` object with host details removed (all diagnostics)."""
    return redact_value(error, diagnostic=True)


__all__ = ["HOST_PATH", "REDACTED", "clear_cache", "redact_rpc_error", "redact_text",
           "redact_tool_result", "redact_value"]
