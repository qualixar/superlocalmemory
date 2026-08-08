# Migration from V2
> SuperLocalMemory V4 Documentation
> https://superlocalmemory.com | Part of Qualixar

Upgrade from SuperLocalMemory V2 to V3. Verify backup before migrating; see
rollback caveats below. No `slm migrate --dry-run` exists.

> **V4.0.0 additive migrations:** V4.0.0 includes `M038_learning_feedback_channel`
> (eager, applied at startup on `learning.db`) and `M039_scene_fact_members`
> (deferred, applied once engine-owned tables exist on `memory.db`). Manual
> `slm db migrate` is not normally required for V4.0.0 itself. `slm db migrate` is
> **forward-only** (`status` / `--dry-run` / apply; no `slm db migrate
> --rollback` and no `slm migrate --rollback` for V4 DBs — see
> `src/superlocalmemory/cli/db_migrate.py` and
> `src/superlocalmemory/storage/migration_runner.py`). It refuses to run
> against a DB written by a newer build and holds back migrations whose
> dependency did not complete. Schema downgrade is unsupported — to revert a V4
> upgrade, restore a **verified pre-upgrade complete backup of the whole data
> root** ( `slm serve stop` first so WAL/SHM checkpoint, then copy all present
> `*.db` plus `-wal`/`-shm` sidecars and `lance/` if present). The `M039`
> projection is `scene_fact_members` with **profile-scoped composite**
> membership (see `src/superlocalmemory/storage/migrations/M039_scene_fact_members.py`).

---

## What Changed in V3

| Area | V2 | V3 |
|------|----|----|
| **Retrieval** | Single-channel semantic search | Five candidate producers (Semantic + BM25 + Temporal + Spreading-Activation + Hopfield) -> RRF fusion + entity-graph post-fusion enhancement |
| **Modes** | One mode (cloud required for smart features) | Three modes: A (zero-cloud), B (local LLM), C (cloud LLM) |
| **Math layer** | None | Fisher-Rao similarity, Sheaf consistency, Langevin lifecycle |
| **Ingestion** | Basic text storage | 11-step pipeline: entities, facts, emotions, beliefs, graph, and more |
| **Data directory** | `~/.claude-memory/` | `~/.superlocalmemory/` (`~/.claude-memory/` symlink preserves old path) |
| **Consistency** | Manual | Automatic contradiction detection |
| **Recall quality** | Good | Significantly better on complex queries (multi-hop, temporal) |

**What stays the same:** All CLI commands, MCP tools, IDE integrations, profiles, trust scores, and learned patterns carry forward.

## Before You Migrate

1. **Update to the latest version:**

```bash
npm update -g superlocalmemory
```

2. **Check your current version:**

```bash
slm --version
# Should show 3.x.x or 4.x.x
```

3. **Verify a complete pre-upgrade backup exists** (do not rely on a live
   `memory.db` copy):

```bash
slm serve stop
# Copy the complete data root to an encrypted/private destination and verify
# owner-only modes (0600/0700). See docs/cloud-backup.md and
# docs/SECURITY-encryption-at-rest.md. Destination follows process umask —
# do not assume 0600 inheritance.
ls -l ~/.superlocalmemory/backups/
slm serve start
```

> No `slm migrate --dry-run` exists for the V2→V3 migrator. For V4 additive
> DB migrations the inspect command is `slm db migrate --dry-run` (and
> `slm db migrate --status`), forward-only.

## Run the Migration

```bash
slm migrate
```

The migration:

1. Creates a backup of your V2 database (verify it is complete and
   owner-only before proceeding)
2. Copies data from `~/.claude-memory/` to `~/.superlocalmemory/`
3. Creates a symlink (`~/.claude-memory/ -> ~/.superlocalmemory/`) so old IDE configs still work
4. Extends the database schema with V3 tables (15 new tables)
5. Re-indexes existing memories for multi-producer retrieval
6. Sets Mode A as default (zero breaking changes)
7. Verifies integrity

> Duration: under 30 s for most databases; 10 000+ memories may take 1–2 min.
> The migration spans file copies, SQLite commits, and a rename/symlink — not a
> single global transaction. Do **not** treat it as globally
> transactional/zero-loss without a verified pre-upgrade backup.

**V4 migrations after that:** `M038` (adds `learning_feedback.channel` for
`pattern_miner`) and the deferred `M039` normalized scene/fact projection are
applied automatically; see header note for DDL details.

## What Gets Preserved

Everything:

- All stored memories (content, timestamps, metadata)
- All profiles and their isolation boundaries
- Trust scores and provenance data
- Learned patterns and behavioral data
- Compliance settings and retention policies
- Audit trail (hash-chain intact)
- IDE configurations (via symlink)

## What Gets Added

The migration adds V3 capabilities to your existing data:

- BM25 token index for keyword search
- Entity graph nodes and edges
- Temporal event entries
- Fisher-Rao similarity metadata
- Sheaf consistency sections
- Langevin lifecycle state

These are computed from your existing memories during migration.

## After Migration

### Verify

```bash
slm status --json
# or slm status for the text summary
slm db migrate --status   # shows M038/M039 applied state: see docs/cli-reference.md
```

Confirm:
- Mode shows `A` (default after migration)
- Memory count matches your V2 count (`slm status --json | jq '.data.fact_count'`)
- `slm db migrate --status` shows expected migrations as applied/verified

### Try a recall

```bash
slm recall "something you stored in V2"
```

Results should match or exceed V2 quality. V3's multi-producer retrieval finds memories that V2's single-channel search might have missed.

### Explore V3 features

```bash
slm trace "your query"       # See channel-by-channel breakdown
slm health                   # Check math layer status
slm mode b                   # Try local LLM mode (if Ollama installed)
# Use slm db migrate --status / --dry-run to inspect additive DB migrations
# (forward-only; no rollback). See `slm ops status` / `slm ops list` for
# stuck operations after upgrades.
```

## Rollback

Rollback of the V2→V3 `slm migrate` is **only** possible while a valid
pre-migration backup still exists and is **not** automatic or retained for
30 days. There is no automatic 30-day retention or timed deletion — verify the
backup file before migrating. Re-creating the backup during migration does not
guarantee a coherent cross-store set on the legacy per-file path
(`docs/cloud-backup.md`).

Downgrade of a V4 DB (M038/M039) is **unsupported**: there is no
`slm db migrate --rollback` (and no `slm migrate --rollback` for V4 DBs).
To revert a V4 upgrade, restore a verified **pre-upgrade complete backup of
the whole data root** (stop the daemon first — `slm serve stop` — and include
WAL/SHM sidecars plus `lance/` if present). Copying a live `memory.db` alone
while the daemon runs is unsafe and does not guarantee a coherent restore set.

## IDE Configuration Updates

### Automatic (recommended)

The migration preserves your IDE configs via symlink. No IDE reconfiguration needed.

### Manual (optional)

If you want to update your IDE configs to use the new path directly:

```bash
slm connect
```

This updates all detected IDE configs to point to `~/.superlocalmemory/` instead of relying on the symlink.

## FAQ

**Q: Will my IDE break during migration?**
No. The symlink ensures old paths still work. Your IDE will not notice the change.

**Q: Do I need to reconfigure my API keys?**
No. API keys are migrated to the new config location automatically (plaintext
`0600` in `config.json` — prefer env if you want to avoid disk persistence).

**Q: Can I run V2 and V3 side by side?**
No. The migration converts your database in place (with backup). No side-by-side.

**Q: What if migration fails halfway?**
The migration spans multiple file copies/SQLite commits and a symlink; it is **not**
a globally atomic/transactional switch. Keep the verified pre-upgrade complete
backup (offline whole-root copy with daemon stopped) and restore that if needed.
Do not rely on an unverified live `memory.db` copy.

**Q: I have multiple profiles. Are they all migrated?**
Yes. All profiles are migrated together. Profile isolation is preserved.

**Q: How big will my database get after migration?**
The V3 schema adds approximately 20-40% to database size due to the entity graph, BM25 index, and math layer metadata. A 50MB V2 database becomes roughly 60-70MB.

---

*SuperLocalMemory V4 — Copyright 2026 Varun Pratap Bhardwaj. AGPL-3.0-or-later. Part of Qualixar.*
