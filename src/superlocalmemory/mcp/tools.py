# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Backward compatibility — tools are now split across sub-modules.

tools_core: 13 core tools (remember, recall, search, etc.)
tools_v28: 6 V2.8-ported tools (report_outcome, lifecycle, audit, etc.)
tools_v3: 5 V3-only tools (set_mode, health, consistency, recall_trace)

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

from superlocalmemory.mcp.tools_core import register_core_tools
from superlocalmemory.mcp.tools_v3 import register_v3_tools
from superlocalmemory.mcp.tools_v28 import register_v28_tools

__all__ = ["register_core_tools", "register_v28_tools", "register_v3_tools"]
