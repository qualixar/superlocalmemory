# Migration from V2
> SuperLocalMemory V3 Documentation
> https://superlocalmemory.com | Part of Qualixar

Upgrade from SuperLocalMemory V2 to V3. Zero data loss, one command, rollback available.

---

## What Changed in V3

| Area | V2 | V3 |
|------|----|----|
| **Retrieval** | Single-channel semantic search | 4-channel: Semantic + BM25 + Entity Graph + Temporal |
| **Modes** | One mode (cloud required for smart features) | Three modes: A (zero-cloud), B (local LLM), C (cloud LLM) |
| **Math layer** | None | Fisher-Rao similarity, Sheaf consistency, Langevin lifecycle |
| **Ingestion** | Basic text storage | 11-step pipeline: entities, facts, emotions, beliefs, graph, and more |
| **Data directory** | `~/.superlocalmemory/` | `~/.superlocalmemory/` (symlink preserves old path) |
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
# Should show 3.x.x
```

3. **(Optional) Preview changes:**

```bash
slm migrate --dry-run
```

This shows exactly what will change without modifying anything.

## Run the Migration

```bash
slm migrate
```

The migration:

1. Creates a full backup of your V2 database
2. Moves data from `~/.superlocalmemory/` to `~/.superlocalmemory/`
3. Creates a symlink (`~/.superlocalmemory/ -> ~/.superlocalmemory/`) so old IDE configs still work
4. Extends the database schema with V3 tables (15 new tables)
5. Re-indexes existing memories for 4-channel retrieval
6. Sets Mode A as default (zero breaking changes)
7. Verifies integrity

**Duration:** Under 30 seconds for most databases. Large databases (10,000+ memories) may take 1-2 minutes.

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
slm status
```

Confirm:
- Mode shows `A` (default after migration)
- Memory count matches your V2 count
- Health shows all green
- Profile is your previous active profile

### Try a recall

```bash
slm recall "something you stored in V2"
```

Results should match or exceed V2 quality. V3's 4-channel retrieval finds memories that V2's single-channel search might have missed.

### Explore V3 features

```bash
slm trace "your query"       # See channel-by-channel breakdown
slm health                   # Check math layer status
slm consistency              # Run contradiction detection
slm mode b                   # Try local LLM mode (if Ollama installed)
```

## Rollback

If anything goes wrong, roll back within 30 days:

```bash
slm migrate --rollback
```

This restores your V2 database from the backup created during migration. The symlink is removed and the original `~/.superlocalmemory/` directory is restored.

**After 30 days:** The backup is automatically cleaned up. If you need to roll back after 30 days, restore from your own backups.

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
No. API keys are migrated to the new config location automatically.

**Q: Can I run V2 and V3 side by side?**
No. The migration converts your database in place (with backup). Use `--rollback` if you want to return to V2.

**Q: What if migration fails halfway?**
The migration is transactional. If any step fails, everything is rolled back automatically. Your V2 data remains untouched.

**Q: I have multiple profiles. Are they all migrated?**
Yes. All profiles are migrated together. Profile isolation is preserved.

**Q: How big will my database get after migration?**
The V3 schema adds approximately 20-40% to database size due to the entity graph, BM25 index, and math layer metadata. A 50MB V2 database becomes roughly 60-70MB.

---

*SuperLocalMemory V3 — Copyright 2026 Varun Pratap Bhardwaj. AGPL-3.0-or-later. Part of Qualixar.*
