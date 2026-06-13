# slm-optimize — Context Optimizer Skill

Automatically compresses large tool outputs and caches repeated reads, reducing
context window usage without a proxy and without losing the full 1M window.

## Prerequisites

- SuperLocalMemory v3.6.11+
- `slm mcp` daemon running
- `slm_compress` visible in `tools/list` (verify with your IDE's MCP tool inspector)

## Install (Claude Code)

```bash
cp skills/slm-optimize/SKILL.md ~/.claude/skills/slm-optimize/SKILL.md
```

## Activate

**Option A — On demand:**
Invoke via the `Skill` tool: `Skill("slm-optimize")`

**Option B — Auto-activate:**
Add to your project or global `CLAUDE.md`:
```markdown
## Context Management
Use the `slm-optimize` skill to compress large outputs and cache repeated reads.
Invoke at session start if context > 50k tokens.
```

## Verify

After activation, run `slm_optimize_stats()` from Claude Code. If it returns `ok:True`,
all 5 optimize tools are reachable and the skill is working.

## What it does

| Feature | How |
|---|---|
| Compresses large tool outputs | `slm_compress` → compressed text + optional `ccr_id` |
| Caches repeated file reads | `slm_cache_set` / `slm_cache_get` keyed by file path |
| Caches repeated bash/search | Same KV tools, keyed by command |
| Recovers exact originals | `slm_retrieve(ccr_id)` when byte-identical content needed |
| Session stats | `slm_optimize_stats()` |

## What it does NOT do

- **Full-turn caching**: impossible without a proxy (`ANTHROPIC_BASE_URL`).
  Use Surface A (proxy mode) for that.
- **Guarantee savings**: results depend on content type and daemon compress config.

## For Cursor, Antigravity, Codex

Same SKILL.md works — all IDEs that support MCP tool calls can use the 5 tools.
The skill itself is pure Markdown with no IDE-specific code.
