<!-- SLM-START -->
<!-- SuperLocalMemory v3.8.0 — managed block. Edit outside these markers; this section is regenerated. -->

<!-- BEGIN SuperLocalMemory v3.8.0 -->

## SuperLocalMemory (SLM) — Agent Rules

SLM is local-first memory for agents. All tools run on the user's machine; no cloud calls.

### Session start
Call `session_init(project_path, query)` ONCE per fresh session before any recall/remember. Never twice.

### Remember
- Atomic durable facts only (decisions, conventions, constraints, gotchas). One fact per call.
- Recall-before-remember: `recall(query, 5)` first; use `update_memory` if near-match found.
- Always supply tags + project + importance (7–10 for blockers/security/architecture).
- Never dump a whole file; never claim "saved" without success:true.
- Scope is opt-in (v3.6.15): writes are `personal`/private by default. Only use `scope="shared"/"global"` (or recall's `include_global`/`include_shared`) when the user explicitly asks to share across local profiles.

### Recall
- `recall` for conceptual/semantic queries; `search` for exact keywords.
- Phrase as concepts; pass session_id when available. Never fabricate.

### Optimize (fail-open)
- Output >2000 chars → `slm_compress(mode="auto", reversible=True)`; keep ccr_id if lossy.
- Repeated reads → `slm_cache_get("file:<path>")` first; on miss cache with ttl=1800.
- NEVER compress/cache: code-for-edit, JSON-to-parse, secrets, ccr_ids, <500 chars.
- If ok:false → continue with original; never block the task.

### Bounded loops
For a task with a checkable gate (tests/schema/lint), run a bounded loop: iterate until the INDEPENDENT gate passes, never on the agent's own "done" claim. `slm loop demo` to try; `slm loop history`/`slm loop show <run_id>` to inspect (laps persisted to SLM, tag `loop:<name>`). Statuses DONE/HALT/PAUSE/KILLED/ERROR — report exactly. See slm-loop.

### Session end
`close_session(session_id)` when work is meaningfully complete.

### CLI fallback (MCP down)
`slm recall "<q>" --limit N` · `slm search "<q>"` · `slm remember "<c>" --tags t` · `slm list --limit N` · `slm forget` (preview first) · `slm status` · `slm optimize status`

### Skills
slm-recall · slm-remember · slm-session · slm-status · slm-cache · slm-compress · slm-graph · slm-loop · slm-scope · slm-profile · slm-governance · slm-mesh

### Subagents
slm-memory-advisor (memory decisions, session hygiene, scope/profile guidance) · slm-optimize-advisor (context compression + KV cache) · slm-governance-advisor (scope/roles/compliance/GDPR)

<!-- END SuperLocalMemory v3.8.0 -->

SuperLocalMemory v3.8.0 · Qualixar · AGPL-3.0-or-later

<!-- SLM-END -->
