# Migration from V2 — V4.0.0

Upgrading from SuperLocalMemory V2 (2.8.6 or earlier) to V4 is a one-command **data migration** that spans file copies, SQLite commits, and a rename/symlink — not a single global transaction. Additive schema work in V4 uses a different command (`slm db migrate`, forward only — see [CLI Reference](CLI-Reference)); do not conflate them.

> **V4.0.0 note:** V4.0.0 includes M038 (eager) and M039 (deferred) migrations that are **automatically applied at startup**; no manual `slm db migrate` is normally required for V4.0.0 itself (see [CHANGELOG](https://github.com/qualixar/superlocalmemory/blob/main/CHANGELOG.md#400---2026-08-01--verifiable-memory-transactions) and `src/superlocalmemory/storage/migration_runner.py`). Schema downgrade is unsupported — to revert a V4 upgrade, restore a verified pre-upgrade backup of the complete data root (stop the daemon first; see backup guidance below). The steps below apply only when upgrading from V2.

## What's New Since V2

- **Three operating modes** (A/B/C) — choose your privacy/accuracy trade-off
- **Multi-producer retrieval** — **five** candidate producers (semantic, BM25, temporal, spreading-activation, Hopfield) fused via RRF, with an entity-graph post-fusion score enhancement (not a 6th candidate)
- **Mathematical foundations** — information-geometric similarity, consistency checking, self-organizing lifecycle (historical V3 research, labeled as such — see [[V3 Architecture]] and [[V3 Mathematical Foundations]])
- **Governed writes (V4)** — admission + policy, per-store obligations, hash-sealed manifest, reconciler (see [Home](Home) reliability contract)
- **SLM-Mesh, multi-scope profiles, cache/compress, Entity Explorer, skill evolution, Modes A/B/C, GDPR/RBAC** (see [[Capabilities and Operations]])
- **Bounded loops & framework adapters (V3.8.0→V4)** — see [[Bounded Loops]] and [[Framework Adapters]]
- **Scale Engine** — parity-gated CozoDB/LanceDB projections (see `slm db scale`)
- **Scene and bridge discovery** — connects related memories across conversations
- **Cross-encoder reranking (Mode C)** — precision ordering of results
- **Enhanced entity resolution** — smarter deduplication and linking

## Before You Migrate

1. **Back up your database** (recommended but not required — migration creates a backup at `~/.superlocalmemory/memory-v2-backup.db` and `~/.claude-memory-v2-original` when applicable). Do **not** copy a live SQLite file while the daemon is running. Stop the daemon first and back up the complete data root or documented store set, handling WAL/SHM sidecars:

```bash
slm serve stop   # stop the daemon so the DB and WAL/SHM are quiescent
# then back up the complete data root (or the documented store set):
cp -a ~/.superlocalmemory ~/.superlocalmemory.pre-migrate-backup
# alternative: tar with WAL/SHM sidecars
tar -czf ~/slm-pre-migrate.tgz -C ~ .superlocalmemory
# verify the backup before proceeding:
ls -lh ~/.superlocalmemory.pre-migrate-backup/memory.db*  # includes memory.db, memory.db-wal, memory.db-shm where present
```

2. **Update to the latest V4:**

```bash
npm install -g superlocalmemory@latest
# or
python -m pip install --upgrade superlocalmemory
slm restart && slm doctor
```

3. **Check migration readiness** (V2→V3 migrator only — `slm db migrate --status` is the V4 schema command instead):

```bash
slm migrate            # prints status if no V2 found, or proceeds with confirmation
```

> There is no `slm migrate --status` flag in the installed parser (`src/superlocalmemory/cli/main.py`); `slm migrate` without args already reports whether a V2 installation exists. Use `slm db migrate --status` to inspect V4 additive migrations.

## Migration Steps (V2→V3)

Run the migration command:

```bash
slm migrate
```

The migration will (not a single atomic transaction — failures are reported mid-migration, not silently rolled back):
1. Create a backup of your V2 database at `~/.superlocalmemory/memory-v2-backup.db` (and `~/.claude-memory-v2-original` for the directory) — verify it exists before relying on rollback
2. Add V3 tables (entity graph, scenes, temporal events, math state)
3. Add V3 columns to existing tables
4. Re-index memories for 5-producer retrieval
5. Build the entity graph from existing memories
6. Move the database to `~/.superlocalmemory/` (with symlink/junction from `~/.claude-memory/` — platform-dependent)
7. Update IDE configurations where possible

This takes 1-5 minutes depending on database size. It creates a backup but does not guarantee zero data loss as a global transactional property — operators must verify (`slm status`, `slm health`, `slm status --json | jq '.data.fact_count'`, and recall checks) before decommissioning the prior backup.

## What Gets Preserved

Everything from V2 carries over:

- All stored memories (content, tags, timestamps, importance)
- All profiles and profile isolation
- Trust scores per agent
- Learning system state (LightGBM models, patterns)
- Provenance chains
- Compliance settings (retention policies, audit logs)
- Knowledge graph data

## What Changes

| Item | V2 | V4 (via V3) |
|------|----|----|
| Database location | `~/.claude-memory/` | `~/.superlocalmemory/` (symlink for compat) |
| Default mode | Single mode | Mode A (zero cloud) |
| Retrieval | Semantic + FTS5 | **Five producers** (semantic + BM25 + temporal + spreading-activation + Hopfield) → RRF + entity-graph enhancement |
| Lifecycle | Manual | Self-organizing (Langevin dynamics) |
| Consistency | None | Automatic contradiction detection (sheaf) |
| Writes (V4) | — | Governed, manifest-sealed, reconciled |

## After Migration

Verify the migration succeeded:

```bash
slm status
slm health
slm ops status --json
```

Check your memory count matches what you had before (`slm status --json | jq '.data.fact_count'`).

## V4 Additive Schema Maintenance (not V2 migration)

```bash
slm db migrate --status    # Inspect pending forward/deferred migrations
slm db migrate --dry-run   # Preview (no writes)
slm db migrate             # Apply pending additive migrations (forward only)
```

V4's `slm db migrate` is **forward apply only** (`status` / `dry-run` / apply); there is no `slm db migrate --rollback` (`src/superlocalmemory/cli/db_migrate.py` and `src/superlocalmemory/storage/migration_runner.py`). It refuses to run against a DB written by a newer build and holds back migrations whose dependencies did not complete. M038 is applied eagerly at startup; M039 is deferred until engine-owned tables exist — no manual command is normally required. Schema downgrade is unsupported; restore a verified pre-upgrade backup of the complete data root (stop the daemon first and include WAL/SHM) instead.

## Rollback (V2→V3 only)

If anything goes wrong with the V2→V3 migration, rollback is only possible while the created backup still exists:

```bash
# verify the backup the migrator created still exists before rolling back:
ls -lh ~/.superlocalmemory/memory-v2-backup.db
ls -ld ~/.claude-memory-v2-original  # directory backup when applicable
slm migrate --rollback
```

This restores your V2 database from the backup (`~/.superlocalmemory/memory-v2-backup.db` or `~/.claude-memory-v2-original`) and reverts IDE configurations where possible. Code has no automatic 30-day deletion — rollback is only possible while you still have the backup; verify before use.

## FAQ

**Will my IDE connections break?**
No. The migration updates IDE configs automatically. The `~/.claude-memory/` path is symlinked to `~/.superlocalmemory/`, so old paths still work.

**Do I need to re-store my memories?**
No. All existing memories are preserved and re-indexed for V4's 5-producer retrieval.

**Can I go back to V2?**
Only while the migrator-created backup still exists (`~/.superlocalmemory/memory-v2-backup.db` / `~/.claude-memory-v2-original`): `slm migrate --rollback`. Verify the backup exists before use — code has no automatic deletion or guaranteed time window.

**Does migration require an internet connection?**
No. The migration is entirely local.

**How long does it take?**
1-5 minutes for typical databases (under 50,000 memories). Larger databases may take longer.

---
*Part of [Qualixar](https://qualixar.com) | Created by [Varun Pratap Bhardwaj](https://varunpratap.com)*
