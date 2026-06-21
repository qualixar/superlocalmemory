---
description: Save an atomic fact or decision to SuperLocalMemory.
argument-hint: <the fact> [#tag1,tag2]
allowed-tools: recall, remember, Bash
---

Save to SuperLocalMemory: $ARGUMENTS

1. First call `recall("$ARGUMENTS", 5)` — dedupe check. If a near-identical memory exists, tell the user and stop; delegate to slm-memory-advisor to update_memory if a correction is needed.
2. Extract tags from any trailing `#tag1,tag2` pattern in the arguments.
3. Determine importance: use 8 for blockers/security/architecture decisions; use 7 for conventions and constraints; use 5 for general facts.
4. Call `remember(content="$ARGUMENTS", tags=<extracted tags>, importance=<n>)`.
5. Confirm only on success:true. If success is not true, report the error — never claim "saved."
6. MCP unavailable → CLI fallback: `slm remember "$ARGUMENTS" --tags <tags>` (note: `--importance` is MCP-only, not a CLI flag).

SuperLocalMemory v3.6.17 · Qualixar · AGPL-3.0-or-later
