# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under the MIT License - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Summarizer — Mode A heuristic + Mode B Ollama + Mode C OpenRouter.

Generates cluster summaries and search synthesis. All LLM failures
fall back to heuristic silently — never crashes the caller.

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

from __future__ import annotations

import logging
import os
import re

logger = logging.getLogger(__name__)


class Summarizer:
    """Generate summaries using heuristic or LLM based on mode."""

    def __init__(self, config) -> None:
        self._config = config
        self._mode = config.mode.value if hasattr(config.mode, 'value') else str(config.mode)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def summarize_cluster(self, members: list[dict]) -> str:
        """Generate a human-readable cluster summary.

        Args:
            members: List of dicts with 'content' key.

        Returns:
            Summary string (2-3 sentences).
        """
        texts = [m.get("content", "") for m in members if m.get("content")]
        if not texts:
            return "Empty cluster."
        if self._mode in ("b", "c") and self._has_llm():
            try:
                prompt = self._cluster_prompt(texts[:10])
                return self._call_llm(prompt, max_tokens=150)
            except Exception as exc:
                logger.warning("LLM cluster summary failed, using heuristic: %s", exc)
        return self._heuristic_summary(texts[:5])

    def synthesize_answer(self, query: str, facts: list[dict]) -> str:
        """Generate a synthesized answer from query + retrieved facts.

        Returns empty string in Mode A (no LLM available).
        """
        if self._mode == "a" or not self._has_llm():
            return ""
        texts = [f.get("content", "") for f in facts if f.get("content")]
        if not texts:
            return ""
        try:
            prompt = self._synthesis_prompt(query, texts[:8])
            return self._call_llm(prompt, max_tokens=250)
        except Exception as exc:
            logger.warning("LLM synthesis failed: %s", exc)
            return ""

    # ------------------------------------------------------------------
    # Heuristic (Mode A — always available)
    # ------------------------------------------------------------------

    def _heuristic_summary(self, texts: list[str]) -> str:
        """First sentence from top-3 texts, joined."""
        sentences = []
        for text in texts[:3]:
            first = self._first_sentence(text)
            if first and first not in sentences:
                sentences.append(first)
        return " ".join(sentences)[:300] if sentences else "No summary available."

    @staticmethod
    def _first_sentence(text: str) -> str:
        """Extract first sentence (up to period, question mark, or 100 chars)."""
        text = text.strip()
        match = re.match(r'^(.+?[.!?])\s', text)
        if match:
            return match.group(1).strip()
        return text[:100].strip()

    # ------------------------------------------------------------------
    # LLM calls (Mode B/C)
    # ------------------------------------------------------------------

    def _has_llm(self) -> bool:
        """Check if LLM is available."""
        if self._mode == "b":
            return True  # Ollama assumed running
        if self._mode == "c":
            return bool(
                os.environ.get("OPENROUTER_API_KEY")
                or getattr(self._config.llm, 'api_key', None)
            )
        return False

    def _call_llm(self, prompt: str, max_tokens: int = 200) -> str:
        """Route to Ollama (B) or OpenRouter (C)."""
        if self._mode == "b":
            return self._call_ollama(prompt, max_tokens)
        return self._call_openrouter(prompt, max_tokens)

    def _call_ollama(self, prompt: str, max_tokens: int = 200) -> str:
        """Call local Ollama for summary generation."""
        import httpx
        model = getattr(self._config.llm, 'model', None) or "llama3.1:8b"
        with httpx.Client(timeout=httpx.Timeout(20.0)) as client:
            resp = client.post("http://localhost:11434/api/generate", json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": max_tokens, "temperature": 0.3},
            })
            resp.raise_for_status()
            return resp.json().get("response", "").strip()

    def _call_openrouter(self, prompt: str, max_tokens: int = 200) -> str:
        """Call OpenRouter API for summary generation."""
        import httpx
        api_key = (
            os.environ.get("OPENROUTER_API_KEY")
            or getattr(self._config.llm, 'api_key', None)
        )
        if not api_key:
            raise RuntimeError("No OpenRouter API key")
        model = (
            getattr(self._config.llm, 'model', None)
            or "meta-llama/llama-3.1-8b-instruct:free"
        )
        with httpx.Client(timeout=httpx.Timeout(20.0)) as client:
            resp = client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": 0.3,
                },
            )
            resp.raise_for_status()
            choices = resp.json().get("choices", [])
            if choices:
                return choices[0].get("message", {}).get("content", "").strip()
            return ""

    # ------------------------------------------------------------------
    # Prompt templates
    # ------------------------------------------------------------------

    @staticmethod
    def _cluster_prompt(texts: list[str]) -> str:
        numbered = "\n".join(f"{i+1}. {t[:200]}" for i, t in enumerate(texts))
        return (
            "Summarize the following related memories in 2-3 concise sentences. "
            "Focus on the common theme and key facts.\n\n"
            f"Memories:\n{numbered}\n\n"
            "Summary:"
        )

    @staticmethod
    def _synthesis_prompt(query: str, texts: list[str]) -> str:
        numbered = "\n".join(f"- {t[:200]}" for t in texts)
        return (
            f"Based on these stored memories, answer the question concisely.\n\n"
            f"Question: {query}\n\n"
            f"Relevant memories:\n{numbered}\n\n"
            "Answer (2-3 sentences):"
        )
