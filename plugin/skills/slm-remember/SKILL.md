---
name: slm-remember
description: Capture durable facts, decisions, constraints, and gotchas into SuperLocalMemory. Use when the user says "remember that", "save this decision", "note this constraint", or when a session produces a conclusion worth persisting across sessions. Always recall first to avoid duplicates.
when_to_use: |
  - "Remember that we use JWT with 1h expiry"
  - "Save this architectural decision"
  - "Store the constraint that X must not Y"
  - "Note this as a gotcha / blocker / convention"
  - After making a non-obvious decision during a coding session
  - After resolving a bug whose root cause should be persisted
allowed-tools: remember, recall, update_memory, Bash
---

# slm-remember — Capture Durable Facts

Store atomic, durable facts into SuperLocalMemory for retrieval in future
sessions. One fact per call. Recall before you remember.

---

## What to store (and what not to)

**Store:**
- Architectural decisions ("Decided to use Postgres not MySQL — reason: JSONB support")
- Project conventions ("All API routes follow /api/v1/resource/{id} pattern")
- Hard constraints ("Never expose raw SQL errors to the HTTP response")
- Resolved gotchas ("Ollama needs keep_alive=-1 or it unloads the model between calls")
- Security rules ("Rate limit all public endpoints at 100 req/min")

**Do not store:**
- Transient context that is only relevant within this conversation
- Large blobs of code or full file contents (those belong in the project, not memory)
- Facts the project README already captures

---

## Recall-before-remember (mandatory discipline)

Before calling `remember`, always call `recall` first with the core terms of
what you are about to store. If a near-duplicate exists:

- Use `update_memory(fact_id, content)` to refine the existing fact instead
  of creating a new one.
- Only call `remember` when no sufficiently similar fact is found.

Duplicates degrade retrieval quality for every future session.

---

## MCP-first workflow

### 1. Check for duplicates first

```
recall(query="JWT token expiry auth", limit=5, session_id="<sid>")
```

If a near-duplicate is returned:

```
update_memory(
  fact_id="f8a2bc91",
  content="JWT tokens use 1h expiry for API access tokens; refresh tokens 30d (updated 2026-06-16)",
)
```

`update_memory` returns `{"success": true, "fact_id": "f8a2bc91", "content": "..."}`.

### 2. Store a new fact

```
remember(
  content="Decided to use JWT with 1h expiry for API auth; refresh tokens persist 30 days",
  tags="auth,security,decision",
  project="superlocalmemory",
  importance=8,
  session_id="<sid>",
)
```

Real response shape:
```json
{
  "success": true,
  "fact_ids": ["c9d4e112"],
  "count": 1,
  "pending": false,
  "message": "Stored (recallable now; enriching async)."
}
```

When `pending: true`, the daemon was offline at save time; the fact enters a
pending queue and becomes recallable once the daemon is back. Do not re-save.

**Never claim "saved" unless `success: true` is in the response.**

### 3. Parameter reference

```
remember(
  content: str,       # required — the atomic fact to store
  tags: str = "",     # comma-separated tags, e.g. "auth,security,gotcha"
  project: str = "",  # project scope, e.g. "superlocalmemory"
  importance: int = 5,# 1–10; see scale below
  session_id: str = "",# from session_init; attributes the write to this session
  scope: str = None,   # v3.6.15 multi-scope: "personal" (default) | "shared" | "global"
  shared_with: str = "",# comma-separated profile_ids for scope="shared"
)
```

> **Multi-scope (v3.6.15, opt-in):** leave `scope` unset for `personal` (private to
> this profile — the default, identical to 3.6.14). `"global"` is visible to every
> profile on the machine; `"shared"` is visible to the profiles in `shared_with`.
> See [docs/shared-memory.md](../../../docs/shared-memory.md).

**importance scale:**
- 1–3: Low — passing notes, ideas, soft preferences
- 4–6: Normal — patterns, conventions, standard decisions (default: 5)
- 7–8: High — architectural decisions, integration contracts, known gotchas
- 9–10: Critical — security rules, blockers, irreversible decisions

Use 7–10 only for facts that would cause real damage if forgotten.

### 4. One fact per call

Store one atomic fact per `remember` call. Do not concatenate multiple unrelated
points into a single content string — they will be hard to update individually
and harder to retrieve cleanly. If you have three separate decisions, make three
calls.

### 5. Always set tags and project

Untagged, unscoped facts are harder to retrieve and harder to manage. Minimum:
set `tags` to one or two relevant terms and `project` to the repo/product name.

---

## Deleting stale facts via CLI

For deletion, the CLI is the authoritative surface. The MCP `forget` tool in
v3.6.14 runs an Ebbinghaus decay cycle — it does NOT delete by query. For
targeted deletion, use the CLI:

```bash
# Preview what would be deleted (always do this first)
slm forget "<query>" --dry-run [--json]

# Execute deletion after confirming the preview
slm forget "<query>" --yes [--json]

# Delete a specific fact by exact ID (use when you have the fact_id)
slm delete <fact_id> --yes [--json]
```

Flags verified in source (main.py):
- `slm forget`: positional `query`, `--dry-run`, `--yes` / `-y`, `--json`
- `slm delete`: positional `fact_id`, `--yes` / `-y`, `--json`

Always run `--dry-run` first and review the preview before passing `--yes`.

---

## CLI fallback (when MCP is unavailable)

```bash
# Store a fact
slm remember "<content>" [--tags a,b,c] [--json]

# Store a shared/global fact (v3.6.15, opt-in)
slm remember "<content>" --scope global
slm remember "<content>" --scope shared --shared-with alice,bob

# Flags verified in source (main.py): --tags, --json, --sync, --scope, --shared-with
# --sync: wait for full enrichment before returning (default is async)
# --scope: personal (default) | shared | global ; --shared-with: profile ids for shared
```

**Flags that do NOT exist** on `slm remember`:
`--importance`, `--project`, `--format` — these are MCP-only params or fabricated.

---

## Update vs forget discipline

| Scenario | Action |
|----------|--------|
| Fact is still true but needs refinement | `update_memory(fact_id, new_content)` |
| Fact is superseded or wrong | `slm forget "<query>" --dry-run` then `--yes` |
| Duplicate found that matches recall result | `update_memory` on the existing one |
| Fact has a known ID and is clearly obsolete | `slm delete <fact_id> --yes` |

---

*SuperLocalMemory v3.6.18 · Qualixar · AGPL-3.0-or-later*
