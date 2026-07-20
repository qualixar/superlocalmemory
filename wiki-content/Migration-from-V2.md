# Migration from V2

Upgrading from SuperLocalMemory V2 to V3 is a one-command process. No data is lost.

## What's New in V3

- **Three operating modes** (A/B/C) — choose your privacy/accuracy trade-off
- **Multi-producer retrieval** — five candidate producers (semantic, BM25, temporal, spreading-activation, Hopfield) fused via RRF, with an entity-graph post-fusion score enhancement
- **Mathematical foundations** — information-geometric similarity, consistency checking, self-organizing lifecycle
- **Scene and bridge discovery** — connects related memories across conversations
- **Cross-encoder reranking** (Mode C) — precision ordering of results
- **Enhanced entity resolution** — smarter deduplication and linking

## Before You Migrate

1. **Back up your database** (recommended but not required — migration creates an automatic backup):

```bash
cp ~/.claude-memory/memory.db ~/.claude-memory/memory.db.backup
```

2. **Update to the latest version:**

```bash
npm install -g superlocalmemory@latest
```

3. **Check migration readiness:**

```bash
slm migrate --status
```

## Migration Steps

Run the migration command:

```bash
slm migrate
```

The migration will:
1. Create a backup of your V2 database
2. Add V3 tables (entity graph, scenes, temporal events, math state)
3. Add V3 columns to existing tables
4. Re-index memories for multi-producer retrieval
5. Build the entity graph from existing memories
6. Move the database to `~/.superlocalmemory/` (with symlink from `~/.claude-memory/`)
7. Update IDE configurations

This takes 1-5 minutes depending on database size.

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

| Item | V2 | V3 |
|------|----|----|
| Database location | `~/.claude-memory/` | `~/.superlocalmemory/` (symlink for compat) |
| Default mode | Single mode | Mode A (zero cloud) |
| Retrieval | Semantic + FTS5 | Five producers (semantic + BM25 + temporal + spreading-activation + Hopfield) -> RRF + entity-graph enhancement |
| Lifecycle | Manual | Self-organizing (Langevin dynamics) |
| Consistency | None | Automatic contradiction detection |

## After Migration

Verify the migration succeeded:

```bash
slm status
slm health
```

Check your memory count matches what you had before.

## Rollback

If anything goes wrong, rollback within 30 days:

```bash
slm migrate --rollback
```

This restores your V2 database from the automatic backup and reverts IDE configurations.

## FAQ

**Will my IDE connections break?**
No. The migration updates IDE configs automatically. The `~/.claude-memory/` path is symlinked to `~/.superlocalmemory/`, so old paths still work.

**Do I need to re-store my memories?**
No. All existing memories are preserved and re-indexed for V3's multi-producer retrieval.

**Can I go back to V2?**
Yes, use `slm migrate --rollback` within 30 days. After 30 days, the backup is automatically removed.

**Does migration require an internet connection?**
No. The migration is entirely local.

**How long does it take?**
1-5 minutes for typical databases (under 50,000 memories). Larger databases may take longer.

---
*Part of [Qualixar](https://qualixar.com) | Created by [Varun Pratap Bhardwaj](https://varunpratap.com)*
