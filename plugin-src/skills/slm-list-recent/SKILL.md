---
name: slm-list-recent
description: List most recent memories in chronological order. Use when the user wants to see what was recently saved, review recent conversations, check what they worked on today, or browse memory history. Shows memories sorted by creation time (newest first).
version: "3.4.23"
license: AGPL-3.0-or-later
compatibility: "Requires SuperLocalMemory V2 installed at ~/.claude-memory/"
attribution:
  creator: Varun Pratap Bhardwaj
  role: Solution Architect & Original Creator
  project: SuperLocalMemory V2
---

# SuperLocalMemory: List Recent

List most recent memories in chronological order (newest first).

## Usage

```bash
slm list [--limit N] [--project name] [--tags tag1,tag2] [--today|--week|--month]
```

## Examples

### Example 1: Last 10 Memories (Default)
```bash
$ slm list
```

**Output:**
```
📝 Recent Memories (10 most recent)

[ID: 1247] 5 minutes ago
Fixed JWT token refresh bug - tokens were expiring too fast
Tags: bug-fix, jwt, auth
Project: myapp
Importance: 8

[ID: 1246] 2 hours ago
React hooks best practices: useCallback for memoization
Tags: react, performance, hooks
Project: frontend-app
Importance: 6

[ID: 1245] 4 hours ago
Database migration strategy: use Alembic for versioning
Tags: database, postgresql, migration
Project: myapp
Importance: 7

[ID: 1244] Yesterday 18:42
Decided to use FastAPI over Flask for new microservice
Tags: python, backend, api, decision
Project: myapp
Importance: 9

[ID: 1243] Yesterday 15:30
Code review feedback: add more error handling to API endpoints
Tags: code-review, api, error-handling
Project: myapp
Importance: 6

...
```

### Example 2: Last 5 Memories
```bash
$ slm list --limit 5
```

### Example 3: Today's Memories
```bash
$ slm list --today
```

**Shows only memories created today**

### Example 4: This Week
```bash
$ slm list --week
```

**Shows memories from last 7 days**

### Example 5: Filter by Project
```bash
$ slm list --project myapp --limit 20
```

**Shows 20 most recent memories from "myapp" project**

### Example 6: Filter by Tags
```bash
$ slm list --tags security,auth --limit 15
```

**Shows 15 most recent memories tagged with security AND auth**

## Arguments

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `--limit` | integer | No | 10 | Number of memories to show |
| `--project` | string | No | All | Filter by project |
| `--tags` | string | No | All | Filter by tags (comma-separated) |
| `--today` | flag | No | - | Show only today's memories |
| `--week` | flag | No | - | Show last 7 days |
| `--month` | flag | No | - | Show last 30 days |

## Sorting & Display

### Chronological Order
Memories are always shown **newest first** (reverse chronological).

**Rationale:** Recent context is usually most relevant.

### Timestamps
- **"5 minutes ago"** - Within last hour
- **"2 hours ago"** - Within last 24 hours
- **"Yesterday 18:42"** - Yesterday with time
- **"Feb 05 14:23"** - Older than yesterday

### Content Preview
- First 200 characters shown
- Ellipsis (...) if truncated
- Use `slm recall --id <ID>` for full content

## Use Cases

### 1. Daily Standup Prep
```bash
# What did I work on yesterday?
slm list --yesterday
```

### 2. Resume Context
```bash
# What was I working on before lunch?
slm list --today --limit 5
```

### 3. Weekly Review
```bash
# What decisions did I make this week?
slm list --week --tags decision
```

### 4. Project Check-In
```bash
# Recent memories for current project
slm list --project myapp --limit 20
```

### 5. Security Audit
```bash
# All security-related memories
slm list --tags security --limit 100
```

## Advanced Usage

### Pagination
```bash
# First page
slm list --limit 10

# Next page (note IDs, then use recall)
slm recall --before-id 1237 --limit 10
```

### Export to File
```bash
# Save recent work to file
slm list --week > this-week.txt

# JSON export (for processing)
slm list --format json --limit 100 > memories.json
```

### Pipe to Other Commands
```bash
# Count memories per project
slm list --limit 1000 | grep "Project:" | sort | uniq -c

# Find common tags
slm list --limit 500 | grep "Tags:" | tr ',' '\n' | sort | uniq -c | sort -rn
```

### Combined with Other Skills
```bash
# 1. List recent memories
slm list --today

# 2. Notice interesting pattern

# 3. Search for related memories
slm recall "FastAPI performance"

# 4. Add new related memory
slm remember "FastAPI async endpoints improve throughput by 3x" --tags performance,fastapi
```

## Output Formats

### Standard Format (Default)
```
[ID: 42] Timestamp
Content preview...
Tags: tag1, tag2
Project: name
Importance: 7
```

### Compact Format
```bash
slm list --format compact
```
```
42 | 5m ago | Content preview... | myapp
43 | 2h ago | Another memory... | default
```

### JSON Format
```bash
slm list --format json
```
```json
{
  "memories": [
    {
      "id": 42,
      "content": "Full content here",
      "tags": ["tag1", "tag2"],
      "project": "myapp",
      "importance": 7,
      "created_at": "2026-02-07T14:23:00Z"
    }
  ],
  "count": 10,
  "total": 1247
}
```

### CSV Format
```bash
slm list --format csv
```
```csv
id,content,tags,project,importance,created_at
42,"Content here","tag1,tag2",myapp,7,2026-02-07T14:23:00Z
43,"Another memory","tag3",default,5,2026-02-07T12:15:00Z
```

## Performance

| Memory Count | List Time | Notes |
|--------------|-----------|-------|
| 10 | ~50ms | Instant |
| 100 | ~200ms | Fast |
| 1,000 | ~500ms | Acceptable |
| 10,000+ | ~1s | Use filters |

**Optimization tips:**
- Use `--limit` to reduce results
- Use `--project` or `--tags` filters
- Use time filters (`--today`, `--week`)

## Troubleshooting

### "No memories found"
**Cause:** Empty database or filters too restrictive

**Solution:**
```bash
# Check total memory count
slm status | grep "Total Memories"

# Remove filters
slm list  # No filters

# Try different project
slm list --project default
```

### "List takes too long"
**Cause:** Large database, no filters

**Solution:**
```bash
# Use smaller limit
slm list --limit 5

# Add filters
slm list --today --project myapp

# Rebuild indexes
slm build-graph
```

### "Timestamps wrong"
**Cause:** System timezone changed

**Solution:**
```bash
# Check system timezone
date

# Timestamps are stored in UTC, displayed in local time
# No action needed usually
```

## Comparison with Search

| Feature | `slm list` | `slm recall` |
|---------|------------|--------------|
| **Sorting** | Chronological | Relevance |
| **Use case** | Browse recent | Find specific |
| **Speed** | Fast | Slower |
| **Filters** | Basic | Advanced |
| **Scoring** | No | Yes (relevance) |

**Rule of thumb:**
- Use `list` when you want to see **what you worked on recently**
- Use `recall` when you want to **find specific information**

## Notes

- **Read-only:** Never modifies data
- **Real-time:** Shows latest state
- **Cross-tool:** Same list from Cursor, ChatGPT, Claude, etc.
- **Privacy:** All local, no external calls

## Related Commands

- `slm remember` - Save a new memory
- `slm recall` - Search memories by relevance
- `slm status` - Check memory count and stats
- `slm switch-profile` - View different profile's memories

---

**Created by:** [Varun Pratap Bhardwaj](https://github.com/varun369) (Solution Architect)
**Project:** SuperLocalMemory V2
**License:** MIT (see [LICENSE](../../LICENSE))
**Repository:** https://github.com/varun369/SuperLocalMemoryV2

*Open source doesn't mean removing credit. Attribution must be preserved per MIT License terms.*
