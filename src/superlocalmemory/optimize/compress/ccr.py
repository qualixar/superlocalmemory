# compress/ccr.py
# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later
#
# CCR (Compressed Context Retrieval) concept and pattern:
#   headroom/ccr/ package (Apache-2.0, Headroom contributors)
#   Specifically: batch_store.py (BatchContext dataclass, TTL pattern),
#   tool_injection.py (MCP tool injection pattern)
#   Attribution: See ATTRIBUTION.md.
# Storage: llmcache_ccr table defined in INTERFACE-CONTRACT §1.
# Database access: CacheDB.ccr_put() and CacheDB.ccr_get() per INTERFACE-CONTRACT §1.

"""CCRStore — stores pre-compression originals, provides retrieval tool."""

from __future__ import annotations

import logging
import re
import threading

logger = logging.getLogger("slm.optimize.compress.ccr")

# SEC-C-04: UUID4 format validator — only ^UUID4^ values pass
_UUID4_RE = re.compile(
    r'^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$'
)


class CCRStore:
    """Stores compression originals and retrieves them by ccr_id.

    Thread-safe. Singleton per daemon instance.
    """

    _instance: "CCRStore | None" = None
    _lock: threading.Lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "CCRStore":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        self._db: "CacheDB | None" = None

    def store(
        self,
        original: bytes,
        model: str = "",
        tenant_id: str = "default",
        ttl_seconds: int | None = None,
    ) -> str:
        """Store a pre-compression original. Returns ccr_id (UUID4) or '' on failure.

        B-03: Called BEFORE compression runs.
        RB-05: Only original bytes accepted; compressed bytes stored via update_compressed().
        """
        try:
            import uuid as _uuid_mod
            import time as _time_mod
            ccr_id = str(_uuid_mod.uuid4())
            db = self._get_db()
            ttl_expires = (
                _time_mod.time() + ttl_seconds
                if ttl_seconds is not None
                else None
            )
            db.ccr_put(ccr_id, original, ttl_expires=ttl_expires)
            logger.debug("CCR stored ccr_id=%s orig_bytes=%d", ccr_id, len(original))
            return ccr_id
        except Exception as exc:
            logger.warning("CCRStore.store failed (fail-open): %s", exc)
            return ""

    def update_compressed(self, ccr_id: str, compressed_bytes: bytes) -> None:
        """Update the compressed_hash for an existing CCR row. Non-fatal if fails."""
        try:
            db = self._get_db()
            if hasattr(db, "ccr_update_compressed"):
                db.ccr_update_compressed(ccr_id, compressed_bytes)
        except Exception as exc:
            logger.debug("CCRStore.update_compressed failed (non-fatal): %s", exc)

    def retrieve(self, ccr_id: str) -> bytes | None:
        """Retrieve a CCR original by ccr_id. Returns None if not found or TTL expired."""
        try:
            db = self._get_db()
            return db.ccr_get(ccr_id)
        except Exception as exc:
            logger.warning("CCRStore.retrieve failed (ccr_id=%s): %s", ccr_id, exc)
            return None

    def get_mcp_tool_definition(self) -> dict:
        return {
            "name": "headroom_retrieve",
            "description": (
                "Retrieve the original (pre-compression) text for a compressed content block. "
                "Use this when you need the full, uncompressed version of content that was "
                "compressed by SLM. Provide the ccr_id from the compression stub comment."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "ccr_id": {
                        "type": "string",
                        "description": "The ccr_id from the compression stub comment.",
                    }
                },
                "required": ["ccr_id"],
            },
        }

    async def handle_mcp_call(self, arguments: dict) -> dict:
        ccr_id = arguments.get("ccr_id", "")
        if not ccr_id:
            return {
                "isError": True,
                "content": [{"type": "text", "text": "ccr_id is required"}],
            }
        if not _UUID4_RE.match(ccr_id):
            return {
                "isError": True,
                "content": [{
                    "type": "text",
                    "text": f"ccr_id must be a UUID4, got {ccr_id!r}",
                }],
            }
        original = self.retrieve(ccr_id)
        if original is None:
            return {
                "isError": True,
                "content": [{
                    "type": "text",
                    "text": (
                        f"CCR original not found for ccr_id={ccr_id!r}. "
                        "Possible causes: entry expired, never stored, or ccr_id incorrect."
                    ),
                }],
            }
        try:
            text = original.decode("utf-8")
        except UnicodeDecodeError:
            logger.warning(
                "CCR ccr_id=%s: original bytes not valid UTF-8; falling back to latin-1",
                ccr_id,
            )
            text = original.decode("latin-1")
        return {"content": [{"type": "text", "text": text}]}

    def _get_db(self) -> "CacheDB":
        if self._db is None:
            from superlocalmemory.optimize.storage.db import CacheDB
            self._db = CacheDB()
        return self._db
