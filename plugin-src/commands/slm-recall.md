---
description: Recall relevant facts and decisions from SuperLocalMemory by query.
argument-hint: <what to recall>
allowed-tools: recall, search, Bash
---

Recall from SuperLocalMemory using the query: $ARGUMENTS

1. Call `recall(query="$ARGUMENTS", limit=10)` via MCP.
2. If no confident match or count==0, also call `search("$ARGUMENTS", 10)`.
3. Present results concisely — fact, tags, importance, date. Never invent or fabricate a memory.
4. If MCP is unavailable, fall back to CLI:
   - `slm recall "$ARGUMENTS" --limit 10`
   - then `slm search "$ARGUMENTS"`

SuperLocalMemory v3.6.15 · Qualixar · AGPL-3.0-or-later
