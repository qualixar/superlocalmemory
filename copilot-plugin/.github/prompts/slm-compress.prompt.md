---
name: slm-compress
description: Compress large text, tool output, or transcripts to reduce context-window usage while keeping the full 1M window intact — call slm_compress(content, mode, reversible, ttl_seconds) to shrink content; if the result is lossy a ccr_id is returned so you can call slm_retrieve(ccr_id) later to recover the exact original; always fail-open (ok:false → continue with the original).
version: "3.8.2"
agent: agent
tools:
  - slm_compress
  - slm_retrieve
  - Bash
---

# slm-compress — Reversible Context Compression (Surface B)

## Purpose

When a tool output, transcript, or accumulated context grows large enough to crowd out working space, `slm_compress` reduces it in-place. The compressed form is used for the remainder of the session; the exact original is recoverable on demand via `slm_retrieve`. This works without a proxy and without touching `ANTHROPIC_BASE_URL`, so the full 1M context window is never sacrificed.

## Primary MCP Tool: slm_compress

```
slm_compress(
    content:      str,          # required — text to compress (max 1 MB)
    mode:         str = "auto",  # "normalize" | "auto" | "aggressive"
    reversible:   bool = True,  # store original in CCR for later retrieval
    ttl_seconds:  int = 86400,  # CCR lifetime in seconds (default 24 h)
) -> dict
```

### Return dict (all keys always present)

| Key | Type | Meaning |
|-----|------|---------|
| `ok` | bool | `True` on success; `False` on internal error or empty input |
| `compressed` | str | Compressed text (or original on failure) |
| `strategy` | str | Which strategy was applied (e.g. `"normalize"`, `"none"`) |
| `tokens_before` | int | Word-count estimate of the input |
| `tokens_after` | int | Word-count estimate of the output |
| `ratio` | float | `tokens_after / tokens_before` (lower = more compact) |
| `lossy` | bool | Whether information was removed |
| `ccr_id` | str \| None | UUID4 session token; present only when `lossy=True` and `reversible=True` |
| `note` | str \| None | Human-readable note (e.g. warnings, recovery hint) |

### Mode semantics (verified from source)

- **`"normalize"`** — lossless whitespace collapse; no daemon dependency; `lossy: false`, `ccr_id: null`.
- **`"auto"`** — delegates to `CompressRouter`; may be lossy depending on daemon config; default.
- **`"aggressive"`** — requests aggressive compression from the daemon; daemon must have `compress_mode=aggressive` set in config; note field will warn if daemon config does not match.

## Recovery Tool: slm_retrieve

When `slm_compress` returns `lossy: true`, the original is stored under the `ccr_id`. Use `slm_retrieve` to get it back:

```
slm_retrieve(ccr_id: str) -> dict
```

| Key | Type | Meaning |
|-----|------|---------|
| `ok` | bool | `True` when content was found |
| `content` | str \| None | Original text, decoded from UTF-8 (or Latin-1 fallback) |
| `size_bytes` | int | Byte length of the stored original |
| `error` | str \| None | Error message on failure; `None` on success |

`ccr_id` must be a valid UUID4. Non-UUID4 strings return `ok: false` immediately.

### CCR security rule

`ccr_id` values are **unguessable session tokens**. Treat them like short-lived credentials:
- Never log them.
- Never share them across agents.
- Never pass them as tool arguments to any tool other than `slm_retrieve`.
- Never compress a `ccr_id` string itself.
- They expire after `ttl_seconds` (default 24 h); `slm_retrieve` returns `ok: false` after expiry.

## Decision: When to Compress

**Compress when:**
- A single tool output or transcript exceeds approximately 2 000 characters.
- You are accumulating repeated context (e.g. full file reads across multiple steps).
- Context is nearing the point where recall quality or response quality degrades.

**Do NOT compress:**
- Code you are about to read, edit, or diff — you need every character.
- JSON you will parse programmatically — compression may alter structure.
- Secrets, credentials, or `ccr_id` strings.
- Anything under ~500 characters — overhead exceeds benefit.
- The compressed form of content already compressed this session.

## Fail-Open Guarantee

`slm_compress` never raises an exception. On any internal error it returns:

```json
{ "ok": false, "compressed": "<original input>", "ratio": 1.0, ... }
```

**When `ok` is `false`, continue with the original content.** Never block a task waiting for compression to succeed.

## Worked Example

```python
# Step 1: compress a large tool output
result = await slm_compress(
    content=long_log_text,
    mode="auto",
    reversible=True,
    ttl_seconds=3600,
)

if result["ok"]:
    working_text = result["compressed"]
    ccr_id = result["ccr_id"]   # None if lossless
else:
    working_text = long_log_text  # fail-open
    ccr_id = None

# ... work with working_text ...

# Step 2: restore original when needed (e.g. before final summary)
if ccr_id:
    restore = await slm_retrieve(ccr_id=ccr_id)
    if restore["ok"]:
        original_text = restore["content"]
```

## Secondary CLI (fallback when MCP is unavailable)

The `slm compress` subcommand exists but has known pre-existing parse-test failures. Prefer the MCP tools above. If you must use CLI:

```bash
slm compress status [--json]
slm compress mode safe|aggressive [--json]
slm compress code on|off [--json]
slm compress prose on|off [--json]
slm compress ccr on|off [--json]
slm compress align on|off [--json]
```

These subcommands control daemon-level compression settings — they do not compress content inline. For inline compression, use `slm_compress` via MCP.

## Size Cap

Content over 1 MB (1 000 000 bytes UTF-8) is processed but `reversible` is forced to `False` and `ccr_id` will be `None`. The `note` field will state `"content over 1MB: ccr skipped"`.

---

## Related skills

- `slm-cache` — for repeated reads; use cache-aside before compressing a frequently re-read result
- `slm-status` — check `tokens_saved_compress` and `compress_runs` from `slm_optimize_stats`

---

SuperLocalMemory v3.8.2 · Qualixar · AGPL-3.0-or-later
