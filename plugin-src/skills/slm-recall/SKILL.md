---
name: slm-recall
description: Search SuperLocalMemory for relevant facts, decisions, and past context. Use when the user asks to recall, search, find, or retrieve stored information. Invokes 5-channel retrieval with LightGBM reranking via MCP.
version: "3.4.23"
license: AGPL-3.0-or-later
compatibility: "SuperLocalMemory v3.4.22 — MCP (preferred) or CLI fallback"
attribution:
  creator: Varun Pratap Bhardwaj
  role: Solution Architect & Creator
  project: SuperLocalMemory v3
---

# SuperLocalMemory: Recall

Search and retrieve memories using semantic similarity, knowledge graph relationships, and full-text search.

## Usage

```bash
slm recall "<query>" [--limit N] [--min-score 0.0-1.0] [--tags tag1,tag2] [--project name]
```

## Examples

### Example 1: Basic Search
```bash
slm recall "FastAPI"
```

**Output:**
```
🔍 Search Results (3 found)

[ID: 42] Score: 0.85
We use FastAPI for REST APIs
Tags: python, backend, api
Project: myapp
Created: 2026-02-05 14:23

[ID: 38] Score: 0.72
FastAPI is faster than Flask for high-throughput APIs
Tags: performance, python
Project: default
Created: 2026-02-01 09:15

[ID: 29] Score: 0.68
Async endpoints in FastAPI improve concurrency
Tags: async, fastapi, python
Project: myapp
Created: 2026-01-28 11:42
```

### Example 2: Limited Results
```bash
slm recall "authentication" --limit 3
```

**Returns:** Top 3 most relevant results

### Example 3: Minimum Relevance Score
```bash
slm recall "React hooks" --min-score 0.7
```

**Only returns results with relevance score ≥ 0.7**

### Example 4: Filter by Tags
```bash
slm recall "database" --tags postgresql,performance
```

**Only searches memories tagged with specified tags**

### Example 5: Filter by Project
```bash
slm recall "API design" --project myapp
```

**Only searches memories in specified project**

## Arguments

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `<query>` | string | Yes | - | Search query |
| `--limit` | integer | No | 10 | Max results to return |
| `--min-score` | float | No | 0.3 | Minimum relevance (0.0-1-tags` | string | No | None | Filter by tags (comma-separated) |
| `--project` | string | No | None | Filter by project |

## Search Methods Used

SuperLocalMemory uses **3 search methods simultaneously** and merges results:

### 1. Semantic Search (TF-IDF)
- Converts query to vector
- Finds similar content vectors
- Best for: Conceptual matches

**Example:**
```
Query: "authentication patterns"
Matches: "JWT tokens", "OAuth flow", "session management"
```

### 2. Knowledge Graph Traversal
- Finds related memories via graph edges
- Discovers connected concepts
- Best for: Related information

**Example:**
```
Query: "FastAPI"
Graph finds: "REST API" → "JWT auth" → "token refresh"
```

### 3. Full-Text Search (FTS5)
- Exact keyword matching
- Fast for known terms
- Best for: Specific phrases

**Example:**
```
Query: "PostgreSQL 15"
Finds: Exact mentions of "PostgreSQL 15"
```

## Relevance Scores

**Score range:** 0.0 - 1.0

| Score | Meaning |
|-------|---------|
| **0.9 - 1.0** | Excellent match (almost exact) |
| **0.7 - 0.9** | Strong match (very relevant) |
| **0.5 - 0.7** | Good match (related) |
| **0.3 - 0.5** | Weak match (loosely related) |
| **< 0.3** | Poor match (filtered out by default) |

**Factors affecting score:**
- Keyword overlap
- Semantic similarity
- Graph distance
- Recency (newer = slight boost)
- Importance level
- Your usage patterns (results improve automatically over time)

## Advanced Usage

### Natural Language (in AI chat)

Most AI assistants automatically invoke this skill when you ask:
- "What did we decide about..."
- "Recall information about..."
- "Search for..."
- "What do we know about..."

**Example in Cursor/Claude:**
```
You: "What did we decide about authentication?"
AI: [Automatically invokes slm-recall skill]
Found 3 memories about JWT tokens and OAuth...
```

### Combined with Other Skills

**1. Recall then remember:**
```bash
# Find existing memories
slm recall "API design"

# Add new related memory
slm remember "New API versioning strategy: use /v2/ prefix" --tags api,versioning
```

**2. Recall then build graph:**
```bash
# Find memories
slm recall "performance"

# Rebuild graph to discover new connections
slm build-graph
```

### Scripting & Automation

**Find and export:**
```bash
# Search and save to file
slm recall "security" --min-score 0.7 > security-notes.txt

# Count memories matching query
slm recall "python" --limit 999 | grep "^\\[ID:" | wc -l
```

**Regular reminders:**
```bash
# Daily standup helper (cron job)
#!/bin/bash
echo "Yesterday's decisions:"
slm recall "decided" --limit 5

echo -e "\nCurrent blockers:"
slm recall "blocked" --tags critical --limit 3
```

## Troubleshooting

### "No memories found"

**Causes:**
1. No memories matching query
2. Min-score too high
3. Wrong project filter

**Solutions:**
```bash
# Lower minimum score
slm recall "query" --min-score 0.1

# Remove filters
slm recall "query"  # No project/tag filters

# Check what memories exist
slm list --limit 20
```

### "Search too slow"

**Causes:**
- Large database (10,000+ memories)
- Complex query
- Knowledge graph not optimized

**Solutions:**
```bash
# Rebuild indexes
slm build-graph

# Use filters to narrow search
slm recall "query" --project myapp --tags specific-tag

# Increase min-score (fewer results = faster)
slm recall "query" --min-score 0.7
```

### "Results not relevant"

**Causes:**
- Query too vague
- Need to add more context

**Solutions:**
```bash
# Be more specific
❌ slm recall "it"
✅ slm recall "authentication system"

# Use multiple keywords
✅ slm recall "FastAPI JWT authentication"

# Use tags to filter
✅ slm recall "performance" --tags database
```

## Output Formats

### Standard Format (Default)
```
🔍 Search Results (3 found)

[ID: 42] Score: 0.85
Content preview...
Tags: tag1, tag2
Project: myapp
Created: 2026-02-05
```

### Programmatic Use
```bash
# JSON output (for scripts)
slm recall "query" --format json
# {"results": [{"id": 42, "content": "...", "score": 0.85}, ...]}

# CSV output
slm recall "query" --format csv
# id,content,score,tags,project,created_at
# 42,"Content...",0.85,"tag1,tag2",myapp,2026-02-05
```

## Performance Benchmarks

| Database Size | Search Time | Notes |
|--------------|-------------|-------|
| 100 memories | ~100ms | Instant |
| 1,000 memories | ~500ms | Fast |
| 10,000 memories | ~1.5s | Acceptable |
| 50,000 memories | ~5s | Consider filtering |

**Optimization tips:**
- Use `--min-score` to filter early
- Use `--tags` or `--project` to narrow search
- Rebuild graph periodically: `slm build-graph`

## Notes

- **Multi-method:** Combines semantic, graph, and keyword search
- **Ranked results:** Best matches first
- **Cross-tool:** Same results in Cursor, ChatGPT, Claude, etc.
- **Privacy:** All search happens locally
- **Real-time:** Database updates reflected immediately

## Related Commands

- `slm remember "<content>"` - Save a new memory
- `slm list` - List recent memories (no search)
- `slm status` - Check memory count and graph stats
- `slm build-graph` - Optimize search performance

---

**Created by:** [Varun Pratap Bhardwaj](https://github.com/varun369) (Solution Architect)
**Project:** SuperLocalMemory V2
**License:** MIT (see [LICENSE](../../LICENSE))
**Repository:** https://github.com/varun369/SuperLocalMemoryV2

*Open source doesn't mean removing credit. Attribution must be preserved per MIT License terms.*

## V3 Update (v3.4.22)

**Preferred**: Use the `recall` MCP tool directly — it uses 5-channel retrieval
(semantic, BM25, entity-graph, temporal, Hopfield) with LightGBM reranking
and feeds the learning loop.

```
recall(query="your query", limit=10, session_id="<current_session>")
```

**CLI fallback** (also valid, no MCP required):
```bash
slm recall "<query>" [--limit N]
```

Always pass `session_id` when using MCP so engagement signals feed the ranker.
