# compress/router.py
# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later
#
# Routing pattern adapted from:
#   headroom/transforms/content_router.py (Apache-2.0, Headroom contributors)
#   Specifically: ContentRouter._determine_strategy(), _strategy_from_detection()
#   Attribution: See ATTRIBUTION.md.

"""CompressRouter — implements CompressHook, dispatches to sub-compressors.

INTERFACE-CONTRACT v2.2 §3 defines: compress(ProxyRequest) → ProxyRequest.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from typing import Any

from superlocalmemory.optimize.proxy.lifecycle import ProxyRequest, CompressHook
from superlocalmemory.optimize.config.store import ConfigStore

logger = logging.getLogger("slm.optimize.compress.router")

_MIN_CHARS_FOR_COMPRESSION: int = 500


class CompressRouter:
    """Implements CompressHook Protocol per INTERFACE-CONTRACT §3.

    compress(req: ProxyRequest) → ProxyRequest
    on_compress(before_tokens, after_tokens, lossy) — metrics callback

    Singleton per daemon instance.
    """

    _instance: "CompressRouter | None" = None
    _lock: threading.Lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "CompressRouter":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        self._llmlingua_compressor: "LLMLinguaCompressor | None" = None
        self._ccr_store: "CCRStore | None" = None
        self._aligner: "CacheAligner | None" = None
        self._config_store: ConfigStore | None = None
        self._metrics_counters: Any = None

    # ── CompressHook Protocol (INTERFACE-CONTRACT §3) ──────────────────────

    def compress(self, req: ProxyRequest) -> ProxyRequest:
        """Compress request body. Called by proxy after cache miss.

        CONTRACT §3: compress(req: ProxyRequest) -> ProxyRequest (NOT ctx → CompressResult).
        Returns req unchanged on error / no-op (fail-open).
        """
        try:
            cfg = self._get_config()

            if not cfg.compress_enabled:
                return req

            body = dict(req.body)
            messages = body.get("messages", [])
            if not isinstance(messages, list):
                return req

            # Step 5: CacheAligner on system prompt (detection only)
            system_text = body.get("system", "")
            if isinstance(system_text, str) and system_text:
                aligner = self._get_aligner()
                align_result = aligner.detect(system_text)
                if align_result.findings:
                    logger.warning(
                        "[%s] CacheAligner: %d volatile tokens — "
                        "provider prefix cache may be unstable: %s",
                        req.request_id,
                        len(align_result.findings),
                        [f.label for f in align_result.findings[:5]],
                    )

            # Step 6: compress each content block
            aggressive = cfg.compress_mode == "aggressive"
            protect_recent = cfg.compress_protect_recent
            new_messages, tokens_before, tokens_after, strategy = self._compress_messages(
                messages=messages,
                aggressive=aggressive,
                protect_recent=protect_recent,
                request_id=req.request_id,
                model=body.get("model", ""),
                tenant_id="default",
            )

            # S-01/Stage-9 fix: build new_bytes BEFORE the improvement guard.
            # Layer 1 normalization saves characters (not word-count tokens), so
            # tokens_after == tokens_before for "normalize" strategy. Checking byte
            # length of the serialized body correctly detects Layer 1 savings.
            body["messages"] = new_messages
            new_bytes = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode()

            bytes_saved = len(req.body_bytes) - len(new_bytes)
            if bytes_saved <= 0 and tokens_after >= tokens_before:
                return req  # neither bytes nor tokens improved

            # CONTRACT §3: fire on_compress metrics callback
            lossy = strategy == "llmlingua2_prose"
            self.on_compress(tokens_before, tokens_after, lossy)

            return ProxyRequest(
                provider=req.provider,
                method=req.method,
                path=req.path,
                headers=req.headers,
                body=body,
                body_bytes=new_bytes,
                request_id=req.request_id,
                stream=req.stream,
                has_tools=req.has_tools,
            )

        except Exception as exc:
            logger.warning("[%s] CompressRouter.compress failed (fail-open): %s",
                           req.request_id if hasattr(req, 'request_id') else '?', exc)
            return req

    def on_compress(self, before_tokens: int, after_tokens: int, lossy: bool) -> None:
        """Metrics callback — CONTRACT §3. MUST NOT raise."""
        try:
            saved = max(0, before_tokens - after_tokens)
            if self._metrics_counters is not None:
                # M-02: pass before/after directly (bytes_original, bytes_after contract)
                self._metrics_counters.on_compress(before_tokens, after_tokens)
            logger.debug("on_compress: saved=%d tokens lossy=%s", saved, lossy)
        except Exception as exc:
            logger.debug("on_compress metrics update failed (non-fatal): %s", exc)

    def set_metrics(self, counters: Any) -> None:
        self._metrics_counters = counters

    # ── Internal routing ──────────────────────────────────────────────────

    def _compress_messages(
        self,
        messages: list[dict[str, Any]],
        aggressive: bool,
        protect_recent: int,
        request_id: str,
        model: str,
        tenant_id: str,
    ) -> tuple[list[dict[str, Any]], int, int, str]:
        total_before = 0
        total_after = 0
        primary_strategy = "none"
        new_messages: list[dict[str, Any]] = []

        # K-05: protect last N *user* turns, not last N messages of any role
        user_indices = [i for i, m in enumerate(messages) if m.get("role") == "user"]
        protect_indices: set[int] = set(user_indices[-protect_recent:]) if protect_recent > 0 else set()
        # Always protect the very last message (current turn, any role)
        if messages:
            protect_indices.add(len(messages) - 1)

        for idx, msg in enumerate(messages):
            role = msg.get("role", "")
            is_tool_msg = (
                role == "tool"  # OpenAI tool result messages (plain text, skip entirely)
                # Anthropic tool_result blocks are NOT skipped — _compress_content_block
                # handles type=="tool_result" blocks by compressing their text content
                # while preserving the block structure. This is where the bulk of
                # Claude Code output lives (bash results, file contents, JSON data).
            )
            if idx in protect_indices or is_tool_msg:
                new_messages.append(msg)
                continue

            new_content, before, after, strat = self._compress_content_block(
                content=msg.get("content", ""),
                aggressive=aggressive,
                request_id=request_id,
                model=model,
                tenant_id=tenant_id,
            )
            total_before += before
            total_after += after
            if strat != "none":
                primary_strategy = strat

            new_msg = dict(msg)
            new_msg["content"] = new_content
            new_messages.append(new_msg)

        return new_messages, total_before, total_after, primary_strategy

    def _compress_content_block(
        self,
        content: Any,
        aggressive: bool,
        request_id: str,
        model: str,
        tenant_id: str,
    ) -> tuple[Any, int, int, str]:
        if isinstance(content, str):
            return self._compress_text(content, aggressive, request_id, model, tenant_id)

        if isinstance(content, list):
            new_blocks: list[Any] = []
            total_before = 0
            total_after = 0
            primary = "none"
            for block in content:
                if not isinstance(block, dict):
                    new_blocks.append(block)
                    continue
                block_type = block.get("type", "")
                if block_type in ("text", "tool_result"):
                    text = block.get("text", "") or _tool_result_text(block)
                    if len(text) < _MIN_CHARS_FOR_COMPRESSION:
                        new_blocks.append(block)
                        continue
                    new_text, before, after, strat = self._compress_text(
                        text, aggressive, request_id, model, tenant_id
                    )
                    total_before += before
                    total_after += after
                    if strat != "none":
                        primary = strat
                    new_block = dict(block)
                    if block_type == "text":
                        new_block["text"] = new_text
                    else:
                        new_block = _set_tool_result_text(new_block, new_text)
                    new_blocks.append(new_block)
                else:
                    new_blocks.append(block)
            return new_blocks, total_before, total_after, primary

        return content, 0, 0, "none"

    def _compress_text(
        self,
        text: str,
        aggressive: bool,
        request_id: str,
        model: str,
        tenant_id: str,
    ) -> tuple[str, int, int, str]:
        tokens_before = _token_estimate(text)

        if len(text) < _MIN_CHARS_FOR_COMPRESSION:
            return text, tokens_before, tokens_before, "none"

        # K-01/K-02/K-03: NEVER compress structured content (JSON or code)
        stripped = text.strip()
        if stripped.startswith(("{", "[")):
            # PERF-02: for large content, structural bracket-match avoids O(n) json.loads().
            # Conservative: matching outer brackets → treat as JSON and skip compression.
            # K-01 mandate is safety-first: false-positive (non-JSON treated as JSON) is
            # safe; false-negative (JSON compressed) would be a correctness violation.
            _last = stripped[-1] if stripped else ""
            if len(stripped) > 8192 and (
                (stripped[0] == "{" and _last == "}") or (stripped[0] == "[" and _last == "]")
            ):
                return text, tokens_before, tokens_before, "none"  # large JSON → passthrough
            try:
                json.loads(stripped)
                return text, tokens_before, tokens_before, "none"  # valid JSON → passthrough
            except json.JSONDecodeError:
                pass  # not valid JSON — treat as prose
            except Exception as exc:
                logger.warning("compress: unexpected error probing JSON content: %s", exc)

        if _detect_language(text) is not None:
            return text, tokens_before, tokens_before, "none"  # code → passthrough

        # Layer 1 — lossless whitespace normalization (always-on, safe)
        normalized = self._normalize_whitespace(text)
        tokens_after_l1 = _token_estimate(normalized)

        # Layer 2 — LLMLingua-2 prose compression (aggressive + opt-in only)
        cfg = self._get_config()
        prose_enabled = bool(getattr(cfg, "compress_prose", False))
        if aggressive and prose_enabled:  # pragma: no cover — LLMLingua optional dep
            compressor = self._get_llmlingua_compressor()
            if compressor is not None:
                # B-03: store original BEFORE lossy compression
                ccr_id = self._ccr_store_original(text.encode(), model, tenant_id)
                compressed = compressor.compress(normalized)
                if ccr_id:
                    self._ccr_update_compressed(ccr_id, compressed.encode())
                tokens_after_l2 = _token_estimate(compressed)
                if tokens_after_l2 < tokens_before:
                    logger.info(
                        "[%s] LLMLingua-2 prose compressed rate=%.2f ccr_id=%s (LOSSY)",
                        request_id,
                        tokens_after_l2 / tokens_before if tokens_before else 1.0,
                        ccr_id,
                    )
                    return compressed, tokens_before, tokens_after_l2, "llmlingua2_prose"

        # S-01 fix: compare character length, not word count.
        # _token_estimate() is word-count — whitespace normalization saves characters/bytes
        # but never removes words, so token counts are identical before and after Layer 1.
        # Character comparison correctly detects when normalization reduced the body size.
        if len(normalized) < len(text):
            return normalized, tokens_before, tokens_after_l1, "normalize"

        return text, tokens_before, tokens_before, "none"

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        """Layer 1 lossless: collapse excess blank lines, strip trailing spaces per line."""
        import re
        text = re.sub(r"\n{3,}", "\n\n", text)
        lines = [line.rstrip() for line in text.split("\n")]
        return "\n".join(lines)

    # ── Lazy loaders ─────────────────────────────────────────────────────

    def _get_llmlingua_compressor(self) -> "LLMLinguaCompressor | None":  # pragma: no cover
        if self._llmlingua_compressor is None:
            try:
                from superlocalmemory.optimize.compress.prose_llmlingua import LLMLinguaCompressor
                self._llmlingua_compressor = LLMLinguaCompressor()
            except ImportError:
                logger.warning("LLMLinguaCompressor not available — prose compression disabled")
                return None
        return self._llmlingua_compressor

    def _get_ccr_store(self) -> "CCRStore":
        if self._ccr_store is None:
            from superlocalmemory.optimize.compress.ccr import CCRStore
            self._ccr_store = CCRStore()
        return self._ccr_store

    def _get_aligner(self) -> "CacheAligner":
        if self._aligner is None:
            from superlocalmemory.optimize.compress.align import CacheAligner
            self._aligner = CacheAligner()
        return self._aligner

    def _get_config(self):
        if self._config_store is None:
            self._config_store = ConfigStore()
        return self._config_store.get()

    # ── CCR helpers ───────────────────────────────────────────────────────

    def _ccr_store_original(self, original_bytes: bytes, model: str, tenant_id: str) -> str:
        """B-03: Store original BEFORE compression. Returns ccr_id or '' on failure."""
        try:
            store = self._get_ccr_store()
            return store.store(original=original_bytes, model=model, tenant_id=tenant_id)
        except Exception as exc:
            logger.warning("CCR store failed (non-fatal): %s", exc)
            return ""

    def _ccr_update_compressed(self, ccr_id: str, compressed_bytes: bytes) -> None:
        """B-03: Update CCR row with compressed bytes after compression completes."""
        try:
            store = self._get_ccr_store()
            store.update_compressed(ccr_id, compressed_bytes)
        except Exception as exc:
            logger.debug("CCR update_compressed failed (non-fatal): %s", exc)

    # ── Public convenience method (M-06) ──────────────────────────────────

    def compress_text(self, text: str, strategy: str = "auto") -> "CompressTextResult":
        """Convenience method for test harness. NEVER raises."""
        try:
            cfg = self._get_config()
            aggressive = cfg.compress_mode == "aggressive"
            compressed, tb, ta, strat = self._compress_text(
                text, aggressive, request_id="eval", model="", tenant_id="default"
            )
            return CompressTextResult(
                compressed_text=compressed, strategy=strat,
                tokens_before=tb, tokens_after=ta,
                lossy=strat == "llmlingua2_prose",
            )
        except Exception as exc:
            logger.debug("compress_text failed (non-fatal): %s", exc)
            t = _token_estimate(text)
            return CompressTextResult(
                compressed_text=text, strategy="none",
                tokens_before=t, tokens_after=t,
                lossy=False,
            )


@dataclass
class CompressTextResult:
    """Result of a compress_text() call.

    UX-02 note: lossy=True only when strategy="llmlingua2_prose" (Layer 2).
    In the default install (LLMLingua optional dep not installed), lossy is
    always False — install `llmlingua>=0.2.0` and set compress_prose=True +
    compress_mode="aggressive" to activate lossy compression.
    """
    compressed_text: str
    strategy: str  # "normalize" | "llmlingua2_prose" | "none"
    tokens_before: int
    tokens_after: int
    lossy: bool = False  # K-10: True only for llmlingua2_prose (Layer 2)


# ── Module-level helpers ──────────────────────────────────────────────────────

def _token_estimate(text: str) -> int:
    return len(text.split()) if text else 0


def _msg_has_tool_result(msg: dict) -> bool:
    """B-09: Detect historical tool_result blocks in messages."""
    content = msg.get("content", "")
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_result":
                return True
            if "tool_use_id" in block:
                return True
    return False


def _detect_language(text: str) -> str | None:
    """Detect programming language. Returns None if ambiguous.

    B-11: Line-start anchored regex. Threshold = 3 signal lines.
    """
    import re

    if not text or len(text) < 50:
        return None

    lines = text.split("\n", 30)

    if lines and lines[0].startswith("#!"):
        shebang = lines[0].lower()
        if "python" in shebang:
            return "python"
        if "node" in shebang or "javascript" in shebang:
            return "javascript"

    if lines and lines[0].startswith("```"):
        lang_hint = lines[0][3:].strip().lower()
        _KNOWN = {"python", "javascript", "js", "typescript", "ts", "go", "rust", "java", "cpp", "c++", "c"}
        if lang_hint in _KNOWN:
            if lang_hint in ("cpp", "c++"):
                return "cpp"
            if lang_hint in ("js", "typescript", "ts"):
                return "javascript"
            return lang_hint

    _LANG_PATTERNS: dict[str, list[str]] = {
        "python": [
            r"^\s*(async\s+)?def\s+\w+\s*\(",
            r"^\s*class\s+\w+[\s:(]",
            r"^\s*import\s+\w+",
            r"^\s*from\s+\w+\s+import\s+",
        ],
        "javascript": [
            r"^\s*(async\s+)?function\s+\w+\s*\(",
            r"^\s*class\s+\w+(\s+extends)?",
            r"^\s*const\s+\w+\s*=",
            r"^\s*import\s+.+from\s+['\"]",
            r"^\s*export\s+(default\s+|const\s+|class\s+)",
        ],
        "go": [
            r"^\s*func\s+(\(\w[\w\s\*]*\)\s+)?\w+\s*\(",
            r"^\s*package\s+\w+",
            r"^\s*import\s+[\(\"\`]",
            r"^\s*type\s+\w+\s+(struct|interface)\s*\{",
        ],
        "rust": [
            r"^\s*(pub\s+)?(async\s+)?fn\s+\w+",
            r"^\s*use\s+\w+",
            r"^\s*(pub\s+)?struct\s+\w+",
            r"^\s*impl(\s+\w+)?\s+",
        ],
        "java": [
            r"^\s*(public|private|protected)\s+(static\s+)?\w+\s+\w+\s*\(",
            r"^\s*import\s+[\w\.]+;",
            r"^\s*(public|private|protected)?\s*class\s+\w+",
        ],
    }

    def _count_signals(patterns: list[str]) -> int:
        count = 0
        for line in lines:
            for pat in patterns:
                if re.search(pat, line):
                    count += 1
                    break
        return count

    scores: dict[str, int] = {
        lang: _count_signals(patterns)
        for lang, patterns in _LANG_PATTERNS.items()
    }
    best_lang, best_score = max(scores.items(), key=lambda kv: kv[1])
    if best_score >= 3:
        return best_lang

    return None


def _tool_result_text(block: dict) -> str:
    content = block.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
        return "\n".join(parts)
    return ""


def _set_tool_result_text(block: dict, new_text: str) -> dict:
    content = block.get("content", "")
    if isinstance(content, str):
        return {**block, "content": new_text}
    if isinstance(content, list):
        new_content = []
        replaced = False
        for b in content:
            if not replaced and isinstance(b, dict) and b.get("type") == "text":
                new_content.append({**b, "text": new_text})
                replaced = True
            else:
                new_content.append(b)
        return {**block, "content": new_content}
    return block
