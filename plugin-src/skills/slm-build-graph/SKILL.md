---
name: slm-build-graph
description: Build or rebuild the knowledge graph from existing memories using TF-IDF entity extraction and Leiden clustering. Use when search results seem poor, after bulk imports, or to optimize performance. Automatically discovers relationships between memories and creates topic clusters.
version: "3.4.23"
license: AGPL-3.0-or-later
compatibility: "Requires SuperLocalMemory V2 installed at ~/.claude-memory/, optional dependencies: python-igraph, leidenalg"
attribution:
  creator: Varun Pratap Bhardwaj
  role: Solution Architect & Original Creator
  project: SuperLocalMemory V2
---

# SuperLocalMemory: Build Knowledge Graph

Build or rebuild the knowledge graph from existing memories to improve search quality and discover hidden relationships.

## Usage

```bash
slm build-graph [--force] [--clustering]
```

## What It Does

### 1. Entity Extraction (TF-IDF)
- Scans all memories
- Identifies important terms (entities)
- Creates nodes in knowledge graph
- Examples: "FastAPI", "JWT", "PostgreSQL", "React hooks"

### 2. Relationship Discovery
- Finds memories sharing entities
- Calculates similarity scores
- Creates edges between related nodes
- Discovers indirect connections

### 3. Topic Clustering (Optional)
- Groups related memories into clusters
- Uses Leiden algorithm (community detection)
- Creates semantic topic groups
- Examples: "Authentication cluster", "Database cluster"

## Examples

### Example 1: Basic Graph Build
```bash
$ slm build-graph
```

**Output:**
```
🔄 Building Knowledge Graph...

Phase 1: Entity Extraction
  Scanning 1,247 memories...
  Extracted 892 unique entities
  Created 892 graph nodes
  ✓ Complete (3.2s)

Phase 2: Relationship Discovery
  Computing similarity scores...
  Created 3,456 edges (relationships)
  Avg edges per node: 3.9
  ✓ Complete (5.1s)

Phase 3: Optimization
  Indexing graph structure...
  Pruning weak edges (score < 0.3)...
  Final edge count: 2,134
  ✓ Complete (1.2s)

✅ Knowledge graph built successfully!

Graph Statistics:
  Nodes: 892
  Edges: 2,134
  Density: 0.27%
  Largest Component: 856 nodes (96%)

Next: Use `slm recall` to see improved search results
```

### Example 2: Force Rebuild
```bash
$ slm build-graph --force
```

**Rebuilds from scratch** (deletes existing graph first)

**Use when:**
- Graph seems corrupted
- Major bulk import completed
- Want fresh start

### Example 3: With Clustering
```bash
$ slm build-graph --clustering
```

**Requires optional dependencies:**
```bash
pip3 install python-igraph leidenalg
```

**Additional output:**
```
Phase 4: Topic Clustering (Leiden)
  Detecting communities...
  Found 47 clusters
  Largest cluster: 89 memories
  Smallest cluster: 3 memories
  Modularity score: 0.82 (excellent)
  ✓ Complete (2.3s)

Discovered Clusters:
  Cluster 1 (89 memories): "Authentication & Security"
    Top entities: JWT, OAuth, tokens, auth, security

  Cluster 2 (76 memories): "Database & PostgreSQL"
    Top entities: PostgreSQL, database, SQL, queries, indexes

  Cluster 3 (54 memories): "React & Frontend"
    Top entities: React, hooks, components, state, props

  ...
```

## Arguments

| Argument | Description | When to Use |
|----------|-------------|-------------|
| `--force` | Delete existing graph and rebuild | Corruption, fresh start |
| `--clustering` | Run topic clustering | Want to discover topic groups |
| `--verbose` | Show detailed progress | Debugging, understanding process |
| `--dry-run` | Preview without saving | Testing, analysis |

## When to Run

### Always Run After:
1. **Bulk imports** - Added 50+ memories at once
2. **Database restore** - Restored from backup
3. **Major project milestone** - Sprint complete, project phase done

### Run Periodically:
4. **Monthly** - Keep graph optimized
5. **After 500 new memories** - Maintain quality
6. **When search feels slow** - Rebuild indexes

### Run on Issues:
7. **Poor search results** - Graph may be stale
8. **Missing relationships** - Rebuild connections
9. **Corrupted graph errors** - Force rebuild

## What Gets Built

### Graph Nodes
**Entities extracted from memories:**
- Technologies: "FastAPI", "PostgreSQL", "React"
- Concepts: "authentication", "performance", "testing"
- Patterns: "TDD", "async", "REST API"
- Decisions: "prefer X over Y"

**Node properties:**
- Entity text
- Frequency (how many memories mention it)
- Importance score
- First seen / last seen

### Graph Edges
**Relationships between entities:**
- **Similarity edge:** Memories share similar content
- **Co-occurrence edge:** Entities appear together
- **Sequential edge:** Memories created close in time

**Edge properties:**
- Similarity score (0.0 - 1.0)
- Shared entities list
- Edge type

### Clusters (if --clustering)
**Topic groups discovered:**
- Cluster ID
- Cluster name (auto-generated from top entities)
- Member memories (which memories belong)
- Top entities in cluster
- Modularity score (how well-defined)

## Performance

| Memory Count | Build Time | Notes |
|--------------|------------|-------|
| 100 | ~1s | Instant |
| 1,000 | ~10s | Fast |
| 10,000 | ~2min | Acceptable |
| 50,000+ | ~15min | Plan accordingly |

**With clustering (add ~50%):**
- 1,000 memories: ~15s
- 10,000 memories: ~3min

**Factors affecting speed:**
- Memory content length
- Vocabulary size (unique words)
- Hardware (CPU, RAM)

## Advanced Usage

### Incremental Updates

```bash
# Add new memories
slm remember "New content..." --tags new

# Incremental graph update (fast)
slm build-graph  # Only processes new memories

# Force full rebuild (slower, thorough)
slm build-graph --force
```

### Monitoring Quality

```bash
# Check graph stats before
slm status | grep "Knowledge Graph"

# Build graph
slm build-graph --verbose

# Check stats after
slm status | grep "Knowledge Graph"
```

### Scripting & Automation

**Weekly rebuild (cron job):**
```bash
#!/bin/bash
# Every Sunday at 3 AM

echo "$(date): Starting graph rebuild"
slm build-graph --clustering >> /var/log/slm-build.log 2>&1
echo "$(date): Graph rebuild complete"
```

**Post-import hook:**
```bash
#!/bin/bash
# After bulk import

memories_added=$1

if [ "$memories_added" -gt 50 ]; then
  echo "Large import detected, rebuilding graph..."
  slm build-graph
fi
```

### Clustering Analysis

```bash
# Build with clustering
slm build-graph --clustering

# Check discovered clusters
slm status --verbose | grep -A 20 "Topic Clusters"

# Search within specific cluster
slm recall "FastAPI" --cluster "Backend & APIs"
```

## Troubleshooting

### "Build failed: Memory error"

**Cause:** Not enough RAM for large graph

**Solution:**
```bash
# Build in chunks (process fewer memories at once)
slm build-graph --chunk-size 1000

# Or increase system memory
# Or archive old memories
```

### "Clustering requires python-igraph"

**Cause:** Optional dependencies not installed

**Solution:**
```bash
pip3 install python-igraph leidenalg

# Verify
python3 -c "import igraph; import leidenalg"

# Try again
slm build-graph --clustering
```

### "Graph build slow"

**Causes:**
- Large database
- Slow disk I/O
- Complex memory content

**Solutions:**
```bash
# Show progress
slm build-graph --verbose

# Skip clustering (faster)
slm build-graph  # No --clustering flag

# Check disk space
df -h ~/.claude-memory/
```

### "Edges seem wrong"

**Cause:** Stale graph or poor similarity threshold

**Solution:**
```bash
# Force complete rebuild
slm build-graph --force

# Adjust similarity threshold (advanced)
slm build-graph --min-similarity 0.4  # Default: 0.3
```

## Graph Metrics Explained

### Node Count
**Total unique entities found**
- Good: > 100 for 1,000 memories
- Poor: < 10 for 1,000 memories

**Why matters:** More nodes = richer semantic understanding

### Edge Count
**Total relationships discovered**
- Good: Edges/Nodes ratio > 2
- Poor: Ratio < 1 (disconnected graph)

**Why matters:** More edges = better search via relationships

### Density
**How connected the graph is**
- Formula: (Edges / Possible Edges) × 100
- Typical: 0.1% - 1%
- Too low (<0.05%): Memories very disconnected
- Too high (>5%): May indicate poor entity extraction

### Largest Component
**Size of biggest connected subgraph**
- Good: >80% of nodes
- Poor: <50% (fragmented knowledge)

**Why matters:** Smaller component = isolated knowledge islands

### Modularity (Clustering)
**How well-defined clusters are**
- Excellent: >0.7
- Good: 0.5 - 0.7
- Poor: <0.3

**Why matters:** Higher = clearer topic separation

## Impact on Other Commands

### slm recall (Search)
**Before graph build:**
- Relies mainly on keyword matching
- May miss related memories

**After graph build:**
- Discovers indirect relationships
- Finds conceptually similar memories
- Better ranked results

**Example:**
```
Query: "authentication"

Before:
- Direct matches only (JWT, auth, login)

After:
- Direct matches (JWT, auth, login)
- + Related concepts (security, tokens, OAuth)
- + Connected memories (API design, user management)
```

### slm status
Shows updated graph statistics

### slm switch-profile
Each profile has separate graph

## Notes

- **Non-destructive:** Original memories never modified
- **Idempotent:** Can run multiple times safely
- **Automatic:** Search uses graph automatically after build
- **Privacy:** All processing local

## Related Commands

- `slm recall` - Search uses the graph
- `slm status` - Check graph stats
- `slm remember` - Add memories (triggers incremental update)

---

**Created by:** [Varun Pratap Bhardwaj](https://github.com/varun369) (Solution Architect)
**Project:** SuperLocalMemory V2
**License:** MIT (see [LICENSE](../../LICENSE))
**Repository:** https://github.com/varun369/SuperLocalMemoryV2

*Open source doesn't mean removing credit. Attribution must be preserved per MIT License terms.*
