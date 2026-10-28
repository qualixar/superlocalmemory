# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""What an OpenAI-compatible embedding endpoint needs before it is saved (#302).

The model name is sent verbatim to the endpoint and the width has to match the
vectors it returns. Neither can be guessed: the local bundle's model id is not
one the endpoint serves, and a guessed width either fails at the first request
or rebuilds a store that held another one. So every door that saves an
embedding config asks for both and says which is missing.

Pure function, no I/O.

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

from __future__ import annotations

from typing import Any

#: The ``code`` a refused save carries, for a client that wants to branch on it.
INCOMPLETE_ENDPOINT_CODE = "embedding_endpoint_incomplete"


def incomplete_endpoint_error(embedding: Any) -> str | None:
    """An actionable message when an endpoint config lacks a model or a width, else ``None``.

    Only an OpenAI-compatible config (provider ``openai`` with an endpoint) is
    checked; every other provider has its own defaults.
    """
    if not (getattr(embedding, "provider", "") == "openai" and getattr(embedding, "api_endpoint", "")):
        return None
    missing = []
    if not (getattr(embedding, "model_name", "") or "").strip():
        missing.append("model name")
    try:
        width = int(getattr(embedding, "dimension", 0) or 0)
    except (TypeError, ValueError):
        width = 0
    if width <= 0:
        missing.append("dimension")
    if not missing:
        return None
    return (
        f"A custom embedding endpoint needs its {' and '.join(missing)}. Fill in "
        f"the model the endpoint serves (for example Qwen3-Embedding) and the "
        f"vector size it returns (use Test Embedding to read it). SuperLocalMemory "
        f"will not guess either one."
    )


__all__ = ["INCOMPLETE_ENDPOINT_CODE", "incomplete_endpoint_error"]
