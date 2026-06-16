---
name: slm-optimize
description: >
  Context optimizer for Claude Code + SLM. Compresses large outputs via
  slm_compress (reversible CCR), caches repeated reads via slm_cache_set/get,
  recovers originals via slm_retrieve. Works on any Claude plan, no proxy,
  full context window preserved.
  Requires SuperLocalMemory v3.6.11+ with optimize MCP tools enabled.
version: "3.6.14"
license: AGPL-3.0-or-later
triggers:
  - "slm-optimize"
  - "compress context"
  - "optimize context"
  - "reduce tokens"
---

# slm-optimize: Context Optimizer for Claude Code

Reduces context window usage through compression and caching.
No proxy required. Full 1M window preserved.

**What this skill does NOT do:**
- Cache the primary Claude conversation turn (impossible without a proxy).
- Guarantee any specific savings percentage — results depend on content type and daemon config.

---

## When This Skill Is Active

Apply these rules automatically throughout the session once this skill is loaded.

---

## Decision Rules

### RULE 1 — Session start: compress CLAUDE.md if large

At the start of any session where this skill is loaded:
1. Read `CLAUDE.md` (or project CLAUDE.md).
2. If `len(content) > 3000` characters: call `slm_compress(content=<text>, mode="normalize", reversible=False)`.
3. Replace the in-context CLAUDE.md block with the compressed version.
4. **Do not modify the actual file** — compress only the in-context representation.

### RULE 2 — Large tool output: compress before processing

After any tool call that returns > 2000 characters:
1. Call `slm_compress(content=<output>, mode="auto", reversible=True)`.
2. Work from `compressed` in the response.
3. If `lossy=True`, store the returned `ccr_id` — call `slm_retrieve` only if you need the exact original for verification, diffing, or code parsing.

**Never compress** (see RULE 6 for the complete exclusion list).

### RULE 3 — Repeated file reads: KV cache

Before calling `Read(file_path)` for a file you have already read this session:
1. `slm_cache_get(key=f"file:{file_path}")`.
2. If `hit=True` → use `value` directly. Skip the `Read` call.
3. If `hit=False` → call `Read`, then `slm_cache_set(key=f"file:{file_path}", value=<content>, ttl_seconds=1800)`.

If you edit a file via Edit/Write, immediately update or invalidate the cache entry.

### RULE 4 — Repeated bash/search results: KV cache

For any Bash or WebSearch call whose result you expect to reuse:
1. `slm_cache_set(key=f"bash:{command[:64]}", value=<output>, ttl_seconds=600)`.
2. On the next identical call: `slm_cache_get` first.

Use `ttl_seconds=3600` for web search results (more expensive to re-run).

### RULE 5 — Retrieve original via slm_retrieve

If you compressed content with `lossy=True` and need the exact original:
1. Call `slm_retrieve(ccr_id=<ccr_id from prior compress response>)`.
2. Check `size_bytes` before retrieving if context is nearly full.
3. Do not retrieve unless you actually need byte-identical content.

### RULE 6 — What NOT to compress or cache

- **Code being sent to Edit/Write**: never compress — the editor needs exact bytes.
- **JSON or structured data you will parse**: compression can alter whitespace and break parsers.
- **Secrets, API keys, credentials, tokens**: never store via `slm_cache_set`.
- **CCR IDs**: never cache via `slm_cache_set` — treat as ephemeral session values.
- **Very short outputs (< 500 chars)**: compression overhead exceeds the gain.

### RULE 7 — Stats and awareness

If the user asks "how much context have you saved?" or "what is the optimization status?":
1. Call `slm_optimize_stats()`.
2. Report `cache_kv_hits`, `compress_runs`, and `tokens_saved_compress` clearly.
3. Note that proxy stats are daemon-persisted; KV stats are this session only.

### RULE 8 — Fail-open behavior

If any `slm_compress`, `slm_retrieve`, `slm_cache_set`, `slm_cache_get`, or
`slm_optimize_stats` call returns `ok:False`:
- Continue with the original (uncompressed/uncached) content.
- Do not retry. Do not surface the error unless the user asks about optimization.
- Optimization is always advisory — it must never block any primary task.

---

## Examples

### Compress a large file read

```
# Instead of working with 8000-char file output directly:
result = slm_compress(content=<file_content>, mode="auto", reversible=True)
# Work from result["compressed"]; keep result["ccr_id"] if lossy=True
```

### Cache a repeated grep result

```
cached = slm_cache_get(key="bash:grep -rn MyClass src/")
if not cached["hit"]:
    output = Bash("grep -rn MyClass src/")
    slm_cache_set(key="bash:grep -rn MyClass src/", value=output, ttl_seconds=600)
```

### Recover original when needed

```
original = slm_retrieve(ccr_id=<ccr_id>)
# original["content"] is byte-identical to what was compressed
```

---

## Stats and Troubleshooting

Run `slm_optimize_stats()` to see current session counters.

If tools return `ok:False`, the daemon may be down or the optimize module may not be
configured. Continue with normal tool calls — this skill degrades gracefully.

Verify tools are available: check that `slm_compress` appears in your MCP `tools/list`.
Requires SuperLocalMemory v3.6.11+ with `slm mcp` running.
