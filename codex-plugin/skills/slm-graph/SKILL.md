---
name: slm-graph
description: >
  Index and query a codebase as a structural graph — build the code graph, trace blast radius of a
  change, find callers/callees/inheritors, semantic code search by meaning, assemble PR review
  context, and detect what changed since last index. Use when the user asks how code connects, what
  breaks if X changes, what calls a function, what a class inherits from, how to navigate an
  unfamiliar codebase, or to understand risk before editing.
when_to_use: |
  - what calls X
  - what breaks if I change Y
  - what does Z inherit from
  - find code that handles authentication
  - impact analysis before editing
  - code navigation in unfamiliar repos
  - blast radius before a PR
  - pre-commit change detection
allowed-tools: build_code_graph, query_graph, get_blast_radius, semantic_search_code, get_review_context, detect_changes, Bash
---

# slm-graph — Code Intelligence Skill

Index any repo as a code knowledge graph and answer structural questions about it: callers, callees, impact radius, semantic search, and review context. Requires the `code` MCP profile (set `SLM_MCP_PROFILE=code` in your plugin `.mcp.json`).

**Prerequisite rule:** every tool except `build_code_graph` self-guards — if the graph is not built it returns `{"success": false, "error": "Code graph not built. Run build_code_graph first."}`. Always index first.

---

## Tool Reference

### 1. `build_code_graph` — index a repository

```
build_code_graph(
    repo_path: str,
    languages: str = "",
    exclude_patterns: str = "",
) -> {success, files_parsed, nodes, edges, flows, communities, duration_ms}
```

Parses all supported source files, extracts functions/classes/imports, builds the call graph, detects execution flows, and identifies code communities. Replaces any previous index for the same repo.

- `repo_path` — absolute path to the repository root. Must exist.
- `languages` — comma-separated language filter, e.g. `"python,typescript"`. Empty string = index all supported languages.
- `exclude_patterns` — comma-separated glob patterns to exclude, e.g. `"**/node_modules/**,**/.venv/**"`. Empty = no exclusions.

When to (re)build:
- Before using any other graph tool for the first time on a repo.
- After significant changes to the codebase (pull, merge, large refactor).
- When `detect_changes` or `query_graph` returns stale/unexpected results.
- Rebuild is safe and idempotent — it replaces the previous index atomically per file.

```
# Index the full repo
build_code_graph(repo_path="/abs/path/to/myrepo")

# Index only Python, skip tests and generated code
build_code_graph(
    repo_path="/abs/path/to/myrepo",
    languages="python",
    exclude_patterns="**/tests/**,**/generated/**"
)
```

---

### 2. `query_graph` — traverse relationships

```
query_graph(
    pattern: str,
    target: str = "",
    limit: int = 20,
) -> {success, pattern, target, results: [{qualified_name, kind, file_path, name}]}
```

Query the graph for structural relationships. `pattern` is required and must be one of the eight valid values below. `target` is a qualified name, partial name, or node ID — matched with exact-then-LIKE fallback.

Valid patterns:

| pattern | returns |
|---|---|
| `callers_of` | functions/methods that call `target` |
| `callees_of` | functions/methods that `target` calls |
| `imports_of` | modules/symbols that `target` imports |
| `imported_by` | who imports `target` |
| `tests_for` | test nodes associated with `target` |
| `inherits_from` | base classes of `target` |
| `inherited_by` | subclasses of `target` |
| `contains` | symbols defined inside `target` (e.g. methods in a class) |

```
# Who calls the auth handler?
query_graph(pattern="callers_of", target="authenticate_user")

# What does the payment processor import?
query_graph(pattern="imports_of", target="PaymentProcessor", limit=30)

# What classes inherit from BaseModel?
query_graph(pattern="inherited_by", target="BaseModel")
```

---

### 3. `get_blast_radius` — impact analysis

```
get_blast_radius(
    changed_files: str,
    max_depth: int = 2,
    max_nodes: int = 500,
) -> {success, changed_nodes, impacted_nodes, impacted_files, edges, depth_reached, truncated}
```

Computes the full impact radius for one or more changed files using bidirectional BFS (callers and callees). Returns every node and file reachable within `max_depth` hops. Use this before editing to understand risk surface.

- `changed_files` — comma-separated file paths relative to the repo root, e.g. `"src/auth/handler.py,src/auth/models.py"`.
- `max_depth` — BFS depth. Default 2. Increase to 3–4 for deep call chains; lower to 1 for a quick first-degree check.
- `max_nodes` — caps the result set. If `truncated=true` in the response, the real blast radius is larger.

```
# What breaks if I change the auth handler?
get_blast_radius(changed_files="src/auth/handler.py")

# Deeper analysis across two files
get_blast_radius(
    changed_files="src/payments/gateway.py,src/payments/models.py",
    max_depth=3,
    max_nodes=200
)
```

If `truncated` is `true`, narrow the scope with `max_nodes` or reduce `max_depth` to get a reliable result.

---

### 4. `semantic_search_code` — find code by meaning

```
semantic_search_code(
    query: str,
    kind: str = "",
    limit: int = 20,
) -> {success, results: [{qualified_name, kind, file_path, score, line_start, name}]}
```

Hybrid FTS5 + vector search over all indexed code entities. Use when you know what the code *does* but not what it's *called*.

- `query` — natural language description, e.g. `"retry logic for HTTP requests"` or `"parse JWT token from header"`.
- `kind` — optional filter: `"Function"`, `"Class"`, `"File"`, or `"Test"`. Empty = all kinds. Case-insensitive match in the engine.
- `limit` — max results. Default 20.

Results include a `score` field (higher = more relevant).

```
# Find where authentication is handled
semantic_search_code(query="authenticate user from request token")

# Find only test functions that cover database writes
semantic_search_code(query="database write transaction rollback", kind="Test")

# Find the rate limiter class
semantic_search_code(query="rate limiting middleware", kind="Class", limit=5)
```

---

### 5. `get_review_context` — assemble PR review context

```
get_review_context(
    changed_files: str,
    include_source: bool = True,
) -> {success, summary, review_items, test_gaps, risk_score}
```

Produces a token-optimized review package for a set of changed files: a plain-language summary, a ranked list of review items with per-node risk scores, and a list of changed symbols that have no associated test coverage.

- `changed_files` — comma-separated file paths relative to the repo root.
- `include_source` — whether to include source code snippets in the context (default `True`). Set `False` to reduce token usage when you only need the risk analysis.

`risk_score` is a float 0–1 on the overall changeset. `review_items[].risk_score` is per-node.

```
# Get review context for a PR touching two files
get_review_context(changed_files="src/auth/handler.py,src/auth/utils.py")

# Risk summary only, no source snippets
get_review_context(
    changed_files="src/payments/gateway.py",
    include_source=False
)
```

---

### 6. `detect_changes` — what changed since last index

```
detect_changes(
    base: str = "HEAD~1",
) -> {success, summary, risk_score, changed_functions, test_gaps, review_priorities}
```

Runs `git diff` against `base`, maps the changed hunks to graph nodes, and returns a risk-scored list of changed functions, test gaps, and review priorities. Requires the repo to be a git repository.

- `base` — git ref to diff against. Default `"HEAD~1"` (one commit back). Any valid git ref works: `"main"`, `"v3.6.13"`, a commit SHA, etc.

```
# What changed in the last commit?
detect_changes()

# What changed since the release branch?
detect_changes(base="release/v3.6.13")

# What changed relative to main?
detect_changes(base="main")
```

Returns `error` if the repo root is not a git repository or if git is not available.

---

## Realistic Workflow

### Explore an unfamiliar codebase

```
# 1. Index it
build_code_graph(repo_path="/abs/path/to/repo")

# 2. Find the entry point by meaning
semantic_search_code(query="request router entry point", kind="Function")

# 3. Trace what it calls
query_graph(pattern="callees_of", target="handle_request")

# 4. See who else calls the same core function
query_graph(pattern="callers_of", target="authenticate_user")
```

### Before editing a function

```
# 1. Know what you are touching
semantic_search_code(query="retry HTTP requests with backoff")

# 2. Understand blast radius before making the change
get_blast_radius(changed_files="src/http/client.py")

# 3. Check test gaps
get_review_context(changed_files="src/http/client.py")
```

### Pre-commit / PR review

```
# 1. What changed in this branch vs main?
detect_changes(base="main")

# 2. Full impact analysis for the changed files
get_blast_radius(changed_files="src/auth/handler.py,src/auth/models.py")

# 3. Assemble review context
get_review_context(changed_files="src/auth/handler.py,src/auth/models.py")
```

---

## Error Handling

All tools return `{"success": false, "error": "<message>"}` on failure — they never raise.

| Error message | Cause | Fix |
|---|---|---|
| `Code graph not built. Run build_code_graph first.` | No index exists | Call `build_code_graph(repo_path=...)` first |
| `Repository path does not exist: <path>` | Bad `repo_path` in build | Pass an absolute path that exists |
| `Git not available or not a git repository: ...` | `detect_changes` needs git | Only works in git repos with git installed |
| `Invalid pattern '...'` | Wrong `pattern` in `query_graph` | Use one of the 8 valid pattern strings |
| `No node found matching '<target>'` | Target not in index | Rebuild or check the qualified name via `semantic_search_code` |

If `build_code_graph` returns `files_parsed: 0`, no supported source files were found — check `repo_path` and `exclude_patterns`.

---

## Profile Requirement

This skill uses graph tools that are only active under the `code` MCP profile
(or `full` / `power`). Your plugin `.mcp.json` must include:

```json
"env": {
  "SLM_MCP_PROFILE": "code"
}
```

Without this, the six graph tools are not registered and will appear as unknown
tools. Run `slm status` to confirm the active profile.

**Switching profiles at runtime (v3.8.0+):** Use `switch_profile("code")` via
MCP to activate the code profile in a session that started with a different
profile. See `slm-profile` for the full profile switching workflow.

---

## Related skills

- `slm-profile` — workspace isolation and profile switching (required for code tools)
- `slm-recall` — retrieve architectural decisions before graph queries
- `slm-status` — confirm the active profile and graph index health

---

SuperLocalMemory v3.8.0 · Qualixar · AGPL-3.0-or-later
