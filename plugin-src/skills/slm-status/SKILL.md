---
name: slm-status
description: Check SuperLocalMemory system status, health, and statistics. Use when the user wants to know memory count, graph stats, patterns learned, database health, or system diagnostics. Shows comprehensive system health dashboard.
version: "3.4.23"
license: AGPL-3.0-or-later
compatibility: "Requires SuperLocalMemory V2 installed at ~/.claude-memory/"
attribution:
  creator: Varun Pratap Bhardwaj
  role: Solution Architect & Original Creator
  project: SuperLocalMemory V2
---

# SuperLocalMemory: Status

Check system status, health metrics, and statistics for your local memory system.

## Usage

```bash
slm status [--verbose] [--check-integrity]
```

## Example Output

### Basic Status
```bash
$ slm status
```

**Output:**
```
╔══════════════════════════════════════════════════════╗
║  SuperLocalMemory V2 - System Status                ║
╚══════════════════════════════════════════════════════╝

📊 Memory Statistics
   Total Memories:        1,247
   This Month:            143
   This Week:             28
   Today:                 5

📈 Knowledge Graph
   Nodes (Entities):      892
   Edges (Relationships): 3,456
   Clusters:              47
   Avg Cluster Size:      19 memories

🎯 Pattern Learning
   Coding Patterns:       34
   Framework Preferences: React (72%), Vue (18%), Angular (10%)
   Testing Style:         TDD (65%), BDD (35%)
   Performance Priority:  High (78%)

💾 Database Health
   Size:                  4.2 MB
   Integrity:             ✅ OK
   Last Backup:           2026-02-07 09:15
   Backup Count:          12

🔧 Current Profile
   Name:                  default
   Created:               2026-01-15
   Last Used:             2026-02-07 14:23

⚙️  System Info
   Install Path:          ~/.claude-memory
   Database:              memory.db
   Python Version:        3.11.7
   SQLite Version:        3.43.2

✅ Status: HEALTHY
```

### Verbose Mode
```bash
$ slm status --verbose
```

**Additional information:**
- Recent memory IDs
- Top entities in graph
- Pattern confidence scores
- Database table sizes
- Index statistics

### Integrity Check
```bash
$ slm status --check-integrity
```

**Runs full database integrity check:**
```
Running integrity check...

Database Structure:       ✅ OK
FTS5 Index:              ✅ OK
Graph Consistency:        ✅ OK
Orphaned Nodes:          0 found
Duplicate Memories:      0 found
Corrupted Entries:       0 found

✅ All checks passed
```

## What This Shows

### 1. Memory Statistics
- **Total:** All memories ever saved
- **This Month:** Memories added in current month
- **This Week:** Last 7 days
- **Today:** Memories added today

**Useful for:**
- Understanding usage patterns
- Tracking growth
- Identifying active periods

### 2. Knowledge Graph
- **Nodes:** Unique entities extracted (people, technologies, concepts)
- **Edges:** Relationships between entities
- **Clusters:** Auto-discovered topic groups
- **Avg Cluster Size:** Memories per cluster

**Health indicators:**
- High edges/nodes ratio = well-connected knowledge
- Many clusters = diverse topics
- Large clusters = focused work

### 3. Pattern Learning
- **Coding Patterns:** Identified preferences and decisions
- **Framework Preferences:** Usage distribution
- **Testing Style:** TDD vs BDD preference
- **Performance Priority:** How important performance is to you

**Based on:**
- Keywords in memories ("prefer", "use", "avoid")
- Frequency of mentions
- Importance levels
- Recency (recent patterns weighted higher)

### 4. Database Health
- **Size:** Database file size
- **Integrity:** PRAGMA integrity_check result
- **Last Backup:** Most recent backup timestamp
- **Backup Count:** Total backups available

**Warning signs:**
- ❌ Integrity: NOT OK → Database corrupted
- ⚠️  Size > 100MB → Consider archiving old memories
- ⚠️  No backups → Enable backup system

### 5. Current Profile
- **Name:** Active profile (default, work, personal, etc.)
- **Created:** When profile was created
- **Last Used:** Last access timestamp

**Profiles allow:**
- Profile isolation
- Context switching
- Separate memory spaces

### 6. System Info
- **Install Path:** Where SuperLocalMemory is installed
- **Database:** Database filename
- **Python Version:** Python interpreter version
- **SQLite Version:** SQLite engine version

## Options

| Option | Description | Use Case |
|--------|-------------|----------|
| `--verbose` | Show detailed stats | Debugging, analysis |
| `--check-integrity` | Run full DB check | Troubleshooting |
| `--format json` | JSON output | Scripting |
| `--format text` | Human-readable (default) | Terminal use |

## Use Cases

### 1. Health Check Before Important Work
```bash
slm status --check-integrity
# Ensure DB is healthy before big import
```

### 2. Understanding Memory Usage
```bash
slm status
# "Do I have enough memories for pattern learning?"
# (Need 20+ for basic patterns, 50+ for advanced)
```

### 3. Performance Monitoring
```bash
slm status --verbose
# Check graph stats, optimize if needed
```

### 4. Backup Verification
```bash
slm status | grep "Last Backup"
# Ensure recent backup exists
```

### 5. Profile Switching Context
```bash
# Before switching
slm status
# Note: "Current Profile: work"

slm switch-profile personal

slm status
# Note: "Current Profile: personal"
```

## Advanced Usage

### Scripting & Automation

**Daily health check (cron job):**
```bash
#!/bin/bash
# Daily at 9 AM

status=$(slm status --check-integrity)
if echo "$status" | grep -q "NOT OK"; then
  echo "SuperLocalMemory: Integrity check FAILED" | mail -s "Alert" you@example.com
fi
```

**Monitoring script:**
```bash
#!/bin/bash
# Monitor memory growth

count=$(slm status | grep "Total Memories:" | awk '{print $3}' | tr -d ',')
echo "$(date),${count}" >> memory-growth.csv
```

**JSON output for dashboards:**
```bash
slm status --format json > status.json
# Parse with jq, send to monitoring system
```

### Performance Indicators

**Good indicators:**
- Graph nodes > 100 → Rich knowledge base
- Edges/nodes ratio > 2 → Well-connected
- Patterns learned > 10 → AI understands your style
- Integrity: OK → Database healthy

**Warning signs:**
- Database size > 50MB but <100 memories → Possible issue
- Backup count: 0 → No disaster recovery
- Last used: >30 days ago → Stale data

## Troubleshooting

### "Status command hangs"
**Cause:** Database locked by another process

**Solution:**
```bash
# Check for locks
lsof ~/.claude-memory/memory.db

# Kill hanging processes
killall python3

# Try again
slm status
```

### "Integrity check fails"
**Cause:** Database corruption

**Solution:**
```bash
# Restore from backup
cp ~/.claude-memory/backups/memory.db.backup.* ~/.claude-memory/memory.db

# Verify
slm status --check-integrity
```

### "Pattern stats missing"
**Cause:** Need more memories (minimum 20)

**Solution:**
```bash
# Check memory count
slm status | grep "Total Memories"

# Add more memories
slm remember "Prefer React hooks over classes"
# ... add 20+ memories ...

# Rebuild patterns
slm build-graph
```

## Output Interpretation

### Status: HEALTHY
✅ All systems operational
- Database intact
- Graph built
- Patterns learned
- Backups available

### Status: WARNING
⚠️  Minor issues detected
- Old backups
- Large database
- Few patterns learned

**Action:** Review verbose output

### Status: ERROR
❌ Critical issues
- Database corrupted
- Integrity check failed
- No accessible data

**Action:** Restore from backup immediately

## Performance Benchmarks

| Command | Typical Time | Notes |
|---------|-------------|-------|
| `slm status` | ~200ms | Fast, lightweight |
| `slm status --verbose` | ~500ms | More data fetching |
| `slm status --check-integrity` | ~2s | Full DB scan |

**For large databases (10,000+ memories):**
- Basic status: ~500ms
- Verbose: ~1.5s
- Integrity check: ~10s

## Notes

- **Non-destructive:** Status check never modifies data
- **Real-time:** Shows current state (not cached)
- **Cross-tool:** Same status from all AI tools
- **Privacy:** All checks local, no external calls

## Related Commands

- `slm list` - List recent memories
- `slm build-graph` - Rebuild knowledge graph
- `slm switch-profile` - Switch memory profile
- `slm recall` - Search memories

---

**Created by:** [Varun Pratap Bhardwaj](https://github.com/varun369) (Solution Architect)
**Project:** SuperLocalMemory V2
**License:** MIT (see [LICENSE](../../LICENSE))
**Repository:** https://github.com/varun369/SuperLocalMemoryV2

*Open source doesn't mean removing credit. Attribution must be preserved per MIT License terms.*
