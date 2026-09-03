# SLM Error Reference

Every error SLM surfaces to users carries a code from this catalog.
Use `slm doctor` to diagnose most issues automatically.

## Queue Error Codes

| Code | Message | Recovery | Exit | HTTP |
|------|---------|----------|------|------|
| `RATE_LIMITED` | Too many requests — back off and retry | Wait the `retry_after_ms` value in the error envelope, then retry | 1 | 429 |
| `QUEUE_FULL` | Recall queue is at capacity | Reduce concurrent callers or increase `SLM_QUEUE_MAX_PENDING` | 1 | 503 |
| `TIMEOUT` | Recall did not complete in time | Check `slm doctor` for daemon health; increase `SLM_RECALL_TIMEOUT_S` if legitimate | 1 | 504 |
| `CANCELLED` | Request was cancelled by the caller | No action needed — caller withdrew the request | 0 | 499 |
| `DEAD_LETTER` | Request failed after max retries | Run `slm doctor`; check daemon logs at `~/.superlocalmemory/logs/daemon.log` | 1 | 504 |
| `DAEMON_DOWN` | SLM daemon is not reachable | Run `slm restart` or `slm doctor` | 1 | 502 |
| `INTERNAL` | Unexpected internal error | Report at github.com/qualixar/superlocalmemory/issues with `slm doctor` output | 2 | 500 |

## Exception Types

| Exception | Module | When raised |
|-----------|--------|-------------|
| `PoolError` | `mcp._pool_adapter` | Worker pool returned an error envelope (`{"ok": false}`) — worker crashed or timed out |
| `CapabilityError` | `core.engine_capabilities` | A LIGHT-mode MCP engine was asked for a FULL-mode operation (recall/store). Route through the daemon instead |
| `SafeFsError` | `core.safe_fs` | File-system safety check failed — symlink detected, wrong owner, or cloud-synced directory |
| `QueueTimeoutError` | `core.recall_queue` | `poll_result` exceeded its deadline waiting for the worker to complete |
| `DeadLetterError` | `core.recall_queue` | Request exhausted `max_receives` retries and was moved to the dead-letter queue |
| `QueueCancelledError` | `core.recall_queue` | All subscribers withdrew before the worker completed |

## CLI Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Operational error (see error code above) |
| 2 | Internal / unexpected error |
| 130 | Interrupted by Ctrl-C (SIGINT) |

## Structured Error Envelope

Daemon HTTP errors use this JSON envelope. MCP tools expose structured failures,
but their historical tool contracts are not one uniform envelope: callers must
handle both the newer `ok: false` form and legacy `success: false` / `error`
forms documented by the individual tool.

```json
{
  "ok": false,
  "error_code": "RATE_LIMITED",
  "error": "too many requests — back off and retry",
  "request_id": "r-abc123",
  "retry_after_ms": 1200
}
```

Fields: `ok` (always `false`), `error_code` (from the table above),
`error` (human-readable), `request_id` (if applicable),
`retry_after_ms` (only for `RATE_LIMITED`).

## Storage Notes

### sqlite-vec (`fact_embeddings`) row deletes vs full-table deletes

Row-targeted deletes work: `DELETE FROM fact_embeddings WHERE rowid = ?`
is the supported primitive (used by the vector store's per-fact delete and
by GDPR erasure, which deletes vectors fact by fact). A bare full-table
`DELETE FROM fact_embeddings` with no `WHERE` clause is rejected by
sqlite-vec with `database disk image is malformed` — that error is
misleading: the database is NOT corrupt, and `PRAGMA integrity_check` /
`quick_check` correctly keep reporting `ok`. Do not nuke the database over
it.

Supported ways to clear vector data:

- Per-fact removal through the product APIs (GDPR erase paths), which
  delete the vec0 row, the metadata row, and the row-map row together.
- Full rebuild through the embedding migration path, which drops and
  recreates the virtual table (`DROP TABLE` + `CREATE VIRTUAL TABLE ...
  USING vec0(...)`) rather than deleting from it.

Never hand-edit the database with raw SQL against the `fact_embeddings*`
virtual tables; their shadow-table layout is a sqlite-vec internal and is
not a stable API.
