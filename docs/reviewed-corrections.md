# Reviewed corrections

SuperLocalMemory treats a correction as a reviewed lifecycle rather
than an in-place rewrite. This preserves the exact historical fact while
preventing an unreviewed replacement from entering current recall.

1. `slm update`, MCP `update_memory`, or the authenticated HTTP update route
   creates an immutable successor and a profile-scoped `proposed` case.
2. Current recall excludes the successor while its case is `proposed`,
   `rejected`, or `rolled_back`. The predecessor remains current during that
   period.
3. An authenticated reviewer applies the case using its expected version. Only
   then does SLM transaction-expire the predecessor and admit the successor.
4. A reviewer can reject a proposal without changing recall, or roll back an
   applied correction to restore the predecessor's exact temporal tuple.

The lifecycle is profile- and scope-checked inside the canonical SQLite writer.
It records identifiers and bounded metadata, not fact text. A direct forget of
any fact linked to correction history returns a conflict instead of pretending
the writer is unavailable; a dedicated erasure workflow owns privacy deletion
across the fact and its ledger.

## Cases SLM proposes by itself

In Mode B and Mode C the save path also files a `proposed` case when it only
guesses that a new memory updates an older one (reasons
`temporal_contradiction`, `consolidation_update`, `consolidation_supersede`).
These guesses never withhold a memory from recall and never block a delete, and
a delete, replace or edit of yours closes them automatically (see
`slm corrections overtaken`). When they pile up unreviewed, `slm corrections
dismiss` clears them in bulk, with a dry run, a backup before `--apply` and a
restore for each case. There is no bulk reject, because a rejected case counts
as decided and then protects its memories from deletion for good. Details:
[`slm corrections dismiss`](cli-reference.md#slm-corrections-dismiss).

## Time-aware recall

Current recall performs hard correction admission before fusion and again after
bridge or scene expansion. Cached session context and pinned facts use the
same admission policy. If the lifecycle record cannot be read, current recall
abstains rather than reintroduce known-stale information. Historical `as_of`
queries retain the predecessor when the system had not yet learned the
correction at that time.

This also means a successor linked to a `proposed`, `rejected`, or
`rolled_back` case is excluded even when it is pinned. A pin is a retrieval
preference, not an authority that can bypass correction review.

Event time and system time stay separate. Applying a correction records when
SLM learned it; `valid_until` changes only when the reviewer supplies an
independently validated real-world boundary.

## Adaptive ranking

Adaptive ranking is an explicit operator choice. With no
`SLM_RANKING` setting, the optional adaptive ranker is off; normal semantic,
BM25, temporal, associative, and graph retrieval remain available. To enable
adaptive ranking, set `SLM_RANKING=v1`, `v2`, or
`v2-ensemble` deliberately in the runtime environment. Correction cases,
BrainTruth observations, and external receipts never become ranking inputs.

## What the Living Brain means

The Living Brain is an observation surface. It shows memory activity, normal
feedback signals, claimed versus independently verified receipt evidence,
external Bounded Loops observations, and correction-case counts. These views
do not automatically change recall, ranking, rewards, model routing, or a
correction's review decision.
