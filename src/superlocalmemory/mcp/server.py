# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under the MIT License - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""SuperLocalMemory V3 — MCP Server.

Clean MCP server calling V3 MemoryEngine. Supports all MCP-compatible IDEs.

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

from __future__ import annotations

# CRITICAL: Set BEFORE any torch/transformers import to prevent Metal/MPS
# GPU memory reservation on Apple Silicon.
import os as _os
_os.environ.setdefault('PYTORCH_MPS_HIGH_WATERMARK_RATIO', '0.0')
_os.environ.setdefault('PYTORCH_MPS_MEM_LIMIT', '0')
_os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
_os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
_os.environ.setdefault('TORCH_DEVICE', 'cpu')

import logging
import sys

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)

server = FastMCP("SuperLocalMemory V3")

# Lazy engine singleton -------------------------------------------------------

_engine = None


def get_engine():
    """Return (or create) the singleton MemoryEngine."""
    global _engine
    if _engine is None:
        from superlocalmemory.core.config import SLMConfig
        from superlocalmemory.core.engine import MemoryEngine

        config = SLMConfig.load()
        _engine = MemoryEngine(config)
        _engine.initialize()
    return _engine


def reset_engine():
    """Reset engine singleton (for testing or mode switch)."""
    global _engine
    _engine = None


# Register all tools and resources --------------------------------------------

from superlocalmemory.mcp.tools_core import register_core_tools
from superlocalmemory.mcp.tools_v28 import register_v28_tools
from superlocalmemory.mcp.tools_v3 import register_v3_tools
from superlocalmemory.mcp.resources import register_resources

register_core_tools(server, get_engine)
register_v28_tools(server, get_engine)
register_v3_tools(server, get_engine)
register_resources(server, get_engine)


if __name__ == "__main__":
    server.run(transport="stdio")
