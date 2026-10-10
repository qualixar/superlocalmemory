# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""HTTP proxy that lets MCP processes use the daemon as their worker.

Without this, every MCP process (one per IDE) would spawn its own
``recall_worker`` subprocess through ``WorkerPool.shared()`` and load
the ONNX embedder into that subprocess. With N IDEs open the total
RSS was approximately N x 1.6 GB — the exact failure Path B was built
to avoid.

With this proxy, the MCP process opens an HTTP connection to the
single long-lived daemon (already running for dashboard / mesh /
health) and forwards ``recall`` and ``store`` calls there. Heavy
engine state exists in exactly one process: the daemon.
"""
from __future__ import annotations

import logging
import urllib.parse
from typing import Any

logger = logging.getLogger(__name__)

_OPAQUE_UNAVAILABLE = "DAEMON_UNAVAILABLE: owned daemon is unavailable; retry later."


def daemon_unavailable_error() -> str:
    """Return a one-line, *diagnosed* daemon-unavailable message.

    The opaque wording this replaces described a stopped daemon, a recycled
    PID, an unreachable port and an identity mismatch identically (issue #104).
    Diagnosis is best effort: if it fails for any reason the caller still gets
    the original, retryable message rather than an exception.
    """
    try:
        from superlocalmemory.cli.daemon import describe_daemon_unavailability

        diagnosis = describe_daemon_unavailability()
        return (
            f"DAEMON_UNAVAILABLE ({diagnosis['reason']}): "
            f"{diagnosis['message']} {diagnosis['hint']}"
        )
    except Exception:  # noqa: BLE001 - diagnosis must never mask the failure
        return _OPAQUE_UNAVAILABLE


class DaemonPoolProxy:
    """:class:`WorkerPool`-shaped facade that talks to the daemon over HTTP.

    The shape matches ``WorkerPool.recall`` / ``WorkerPool.store`` so that
    the existing pool adapter in ``mcp/_pool_adapter.py`` can swap between
    a local subprocess pool and the daemon proxy without any adapter
    change. Errors are returned as ``{"ok": False, "error": "..."}``
    envelopes — the adapter is responsible for surfacing those.
    """

    def __init__(
        self,
        port: int | None,
        *,
        timeout_s: float = 30.0,
        unavailable: bool = False,
    ) -> None:
        # v3.4.59: 8s→30s — dense graph recall can exceed the old timeout.
        self._port = port
        self._timeout = timeout_s
        self._unavailable = unavailable

    @staticmethod
    def _unavailable_response() -> dict[str, Any]:
        return {
            "ok": False,
            "code": "DAEMON_UNAVAILABLE",
            "retryable": True,
            "error": daemon_unavailable_error(),
        }

    def recall(
        self, query: str, limit: int = 10, session_id: str = "",
        fast: bool | None = None,
        include_global: bool | None = None,
        include_shared: bool | None = None,
        window: str | None = None,
        as_of: str | None = None,
        known_as_of: str | None = None,
        valid_at: str | None = None,
        include_unknown: bool = False,
        profile_id: str = "",
        answer_check: bool | str = True,
        project: str = "",
        saved_by: str = "",
        about: str = "",
        kind: str = "",
        prefer_project: str = "",
        tags: "str | list[str] | None" = None,
        tags_match: str = "all",
        project_strict: bool = False,
        budget_s: float | None = None,
    ) -> dict[str, Any]:
        if self._unavailable:
            return self._unavailable_response()
        _params: dict[str, Any] = {
            "q": query,
            "limit": limit,
            "session_id": session_id or "",
        }
        # Per-request profile routing (spec section 3/5): the anchor is only
        # serialized when the caller set it — an unset profile_id keeps the
        # legacy query string byte-identical, exactly like the scope flags
        # above. The daemon serves this one recall against that profile.
        # 4.1.14 audit: stripped (whitespace-only is legacy).
        if (profile_id or "").strip():
            _params["profile_id"] = profile_id.strip()
        # v3.8.2 client-driven agentic: only send ``fast`` when the caller set it
        # explicitly. Unset (None) lets the daemon resolve the configured
        # client-driven-agentic default — the same way scope flags are handled.
        if fast is not None:
            _params["fast"] = "true" if fast else "false"
        # v3.6.15 multi-scope: only send the scope flags when explicitly set, so
        # an unset value lets the daemon resolve the configured default (shared
        # is opt-in). "None" must NOT become the string "none" on the wire.
        if include_global is not None:
            _params["include_global"] = "true" if include_global else "false"
        if include_shared is not None:
            _params["include_shared"] = "true" if include_shared else "false"
        if window:
            _params["window"] = window
        # 4.1.19 facets, sent only when set.
        for _facet_name, _facet_value in (("project", project), ("saved_by", saved_by),
                                          ("about", about),
                                          ("prefer_project", prefer_project)):
            if (_facet_value or "").strip():
                _params[_facet_name] = _facet_value.strip()
        # 4.1.22: sent only when set, so every other recall's query string is
        # unchanged.
        if project_strict and (project or "").strip():
            _params["project_strict"] = "true"
        # 4.1.19 WP8: sent only when set, so an unset kind keeps the query
        # string byte-identical to before this filter existed.
        if (kind or "").strip():
            _params["kind"] = kind.strip()
        # 4.1.22: sent only when set. A string is sent as-is (the
        # daemon splits a single ``?tags=a,b`` on the comma); a real list is
        # sent as a REPEATED param (``doseq=True`` below) so a label
        # containing a comma survives the round trip untouched.
        if isinstance(tags, str) and tags.strip():
            _params["tags"] = tags.strip()
        elif isinstance(tags, (list, tuple)) and tags:
            _params["tags"] = [str(t) for t in tags if str(t).strip()]
        if _params.get("tags") and (tags_match or "").strip().lower() == "any":
            _params["tags_match"] = "any"
        if as_of:
            _params["as_of"] = as_of
        # A relayed call's remaining time: sent only when the request carried a
        # deadline, so every other query string is unchanged.
        if budget_s is not None:
            _params["budget_s"] = f"{float(budget_s):.3f}"
        if known_as_of:
            _params["known_as_of"] = known_as_of
        if valid_at:
            _params["valid_at"] = valid_at
        if include_unknown:
            _params["include_unknown"] = "true"
        # A remote caller's recall tells the daemon so, to hide what a remote app
        # may not see (pictures, pages, folders). Only ever added, never for a
        # caller on this computer, so its query string is unchanged.
        from superlocalmemory.mcp.remote_visibility import current_view
        from superlocalmemory.retrieval.remote_view import VIEW_PARAM

        if current_view():
            _params[VIEW_PARAM] = current_view()
        # S-M2: a recall that loads context (session start, auto-injection) is
        # not a question, so the daemon must not judge it or send it anywhere.
        # Sent only when asked — explicitly, or from inside
        # ``skip_answer_check()`` — so every other query string is unchanged.
        from superlocalmemory.core.answer_check_scope import (
            ANSWER_CHECK_PARAM,
            ANSWER_CHECK_SKIP,
            answer_check_skipped,
        )
        # ``answer_check`` is True (ask as usual), False (skip), or one of the
        # daemon's request words — the bounded-loop gate sends "no_reorder":
        # it needs the verdict, never the reorder, and pays for one per lap.
        from superlocalmemory.retrieval.answer_check_status import REQUEST_NO_REORDER

        if answer_check is False or answer_check_skipped() or (
                isinstance(answer_check, str) and answer_check == ANSWER_CHECK_SKIP):
            _params[ANSWER_CHECK_PARAM] = ANSWER_CHECK_SKIP
        elif answer_check == REQUEST_NO_REORDER:
            _params[ANSWER_CHECK_PARAM] = REQUEST_NO_REORDER
        # doseq=True: safe for every existing plain-string param (urlencode
        # special-cases str/bytes even with doseq on — only a genuine list
        # value, like `tags` above, expands to repeated `key=value` pairs).
        params = urllib.parse.urlencode(_params, doseq=True)
        try:
            from superlocalmemory.cli.daemon import daemon_request

            data = daemon_request(
                "GET",
                f"/recall?{params}",
                timeout_seconds=self._timeout,
                preserve_not_found=True,
            )
        except Exception as exc:
            # 4.1.14 audit: a live daemon refusing an unknown profile is
            # an answer, not an outage — surface it instead of collapsing
            # to DAEMON_UNAVAILABLE. Never retried: 404s don't heal.
            # Discriminated by shape, not by name, so a failed import can
            # never hit an unbound reference here.
            if type(exc).__name__ == "DaemonNotFound" and hasattr(exc, "code"):
                return {
                    "ok": False,
                    "success": False,
                    "code": getattr(exc, "code"),
                    "retryable": False,
                    "error": getattr(exc, "message", "daemon returned 404"),
                }
            logger.warning("daemon /recall failed: %s", exc)
            return self._unavailable_response()
        if not isinstance(data, dict):
            return self._unavailable_response()
        data.setdefault("ok", True)
        return data

    def store(
        self, content: str, metadata: dict | None = None, *, kind: str = "",
    ) -> dict[str, Any]:
        """``kind`` is a kind the caller declared, already validated. It is
        sent as the request's own ``kind`` field - the only place the daemon
        accepts one from. A ``_slm_memory_kind`` inside ``metadata`` is never
        promoted to it: the daemon strips that key, and lifting it here would
        let any metadata a client forwards confirm a kind."""
        if self._unavailable:
            return self._unavailable_response()
        meta = dict(metadata or {})
        # ``replaces``, ``scope``, ``shared_with`` and ``session_date`` are
        # fields of the /remember request, not memory metadata (L3-24): the
        # daemon-owned path (tools_core.remember's main branch) sends them as
        # top-level body fields, never nested inside "metadata". Before this
        # fix they stayed inside ``meta`` here, so RememberRequest never saw
        # them as scope/shared_with/session_date at all -- a remember() call
        # that fell back to this proxy silently lost its scope, its
        # shared_with list, and its "when this memory is about" date.
        replaces = meta.pop("replaces", None)
        scope = meta.pop("scope", None)
        shared_with = meta.pop("shared_with", None)
        session_date = meta.pop("session_date", "")
        tags = meta.get("tags", "")
        if isinstance(tags, (list, tuple, set)):
            tags = ",".join(str(tag) for tag in tags)
        body = {
            "content": content,
            "tags": tags,
            "metadata": meta,
            "scope": scope,
            "shared_with": shared_with,
            "session_id": meta.get("session_id", ""),
            "session_date": session_date,
            "idempotency_key": meta.get("idempotency_key") or None,
            "profile_id": meta.get("profile_id", ""),
        }
        # A 422 (e.g. an idempotency key already used for another request)
        # is a refusal of this request: surfaced, never read as an outage.
        flags = {"preserve_conflict": True, "preserve_not_found": True,
                 "preserve_unprocessable": True}
        if replaces is not None:
            body["replaces"] = replaces
        from superlocalmemory.mcp._remember_kind import with_declared_kind

        body, flags = with_declared_kind(body, flags, (kind or "").strip())
        # One identity-aware daemon client owns descriptor validation,
        # capability delivery, and exact-instance targeting. A raw urllib POST
        # here previously became unauthenticated when /remember was hardened
        # and could also attach to a stale/foreign port.
        try:
            from superlocalmemory.cli.daemon import (
                DaemonConflict,
                DaemonRefused,
                DaemonUnprocessable,
                daemon_request,
            )
        except Exception as exc:
            logger.warning("daemon client import failed: %s", exc)
            return self._unavailable_response()
        try:
            data = daemon_request("POST", "/remember", body, **flags)
        except DaemonConflict as exc:
            return {
                "ok": False,
                "code": "PROFILE_MISMATCH",
                "retryable": False,
                "error": str(exc),
            }
        except DaemonUnprocessable as exc:
            # The daemon refused the request itself (e.g. a ``replaces`` it
            # cannot honour). Retrying cannot change that answer.
            return {
                "ok": False,
                "code": exc.code,
                "retryable": False,
                "error": exc.message,
            }
        except DaemonRefused as exc:
            # 401/403 is an answer, not an outage; retrying cannot change it.
            return {
                "ok": False,
                "code": "NOT_AUTHORIZED",
                "retryable": False,
                "error": str(exc),
            }
        except Exception as exc:
            # 4.1.14 audit: surface a live daemon's unknown-profile 404
            # instead of collapsing it to DAEMON_UNAVAILABLE. Shaped check
            # (see recall above) — never retried, 404s don't heal.
            if type(exc).__name__ == "DaemonNotFound" and hasattr(exc, "code"):
                return {
                    "ok": False,
                    "success": False,
                    "code": getattr(exc, "code"),
                    "retryable": False,
                    "error": getattr(exc, "message", "daemon returned 404"),
                }
            logger.warning("daemon /remember failed: %s", exc)
            return self._unavailable_response()
        if not isinstance(data, dict):
            return self._unavailable_response()
        data.setdefault("ok", True)
        return data


def choose_pool() -> Any:
    """Return the best available pool for this MCP process.

    The daemon is the sole canonical writer. A bounded daemon auto-start is
    attempted for first use; if it cannot become healthy, return a facade that
    reports a retryable ``DAEMON_UNAVAILABLE`` envelope. Never construct a
    process-local ``WorkerPool`` from an MCP client.
    """
    try:
        from superlocalmemory.cli.daemon import (
            _get_port,
            ensure_daemon,
            is_daemon_running,
        )
        if is_daemon_running() or ensure_daemon():
            return DaemonPoolProxy(port=_get_port())
    except Exception as exc:
        logger.warning("daemon probe or bounded start failed: %s", exc)
    return DaemonPoolProxy(port=None, unavailable=True)
