---
name: slm-remember
description: Save content to SuperLocalMemory with intelligent indexing and knowledge graph integration. Use when the user wants to remember information, save context, store coding decisions, or persist knowledge for future sessions. Automatically indexes, graphs, and learns patterns.
version: "3.4.23"
license: AGPL-3.0-or-later
compatibility: "Requires SuperLocalMemory V2 installed at ~/.claude-memory/"
attribution:
  creator: Varun Pratap Bhardwaj
  role: Solution Architect & Original Creator
  project: SuperLocalMemory V2
---

# SuperLocalMemory: Remember

Save content to your local memory system with automatic indexing, knowledge graph integration, and pattern learning.

## Usage

```bash
slm remember "<content>" [--tags tag1,tag2] [--project name] [--importance 1-10]
```

## Examples

### Example 1: Basic Memory
```bash
slm remember "We use FastAPI for REST APIs"
```

**What happens:**
- Content saved to SQLite database
- TF-IDF vectors generated for semantic search
- Entities extracted and added to knowledge graph
- Pattern learning analyzes for coding preferences
- Memory ID returned (e.g., 42)

### Example 2: With Tags
```bash
slm remember "JWT tokens expire after 24 hours" --tags security,auth,jwt
```

**Tags help with:**
- Organization
- Filtering
- Related memory discovery

### Example 3: With Project
```bash
slm remember "Database uses PostgreSQL 15 with UUID primary keys" --project myapp --tags database,postgresql
```

**Project isolation:**
- Separate memories per project
- Switch profiles with `slm switch-profile`
- No context bleeding

### Example 4: Important Memory
```bash
slm remember "CRITICAL: Production deploy requires approval from @lead" --importance 10 --tags deployment,production
```

**Importance (1-10):**
- 1-3: Low priority (notes, ideas)
- 4-6: Normal (coding patterns, decisions)
- 7-9: High priority (critical info, warnings)
- 10: Critical (blockers, security issues)

## Arguments

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `<content>` | string | Yes | - | The text to remember |
| `--tags` | string | No | None | Comma-separated tags |
| `--project` | string | No | "default" | Project name |
| `--importance` | integer | No | 5 | Priority level (1-10) |

## Output

```
Memory added with ID: 42

✅ Memory saved successfully

Next steps:
  • Use `slm recall <query>` to search this memory
  • Use `slm list` to see recent memories
```

## What Happens Behind the Scenes

1. **Content Storage:** Saved to SQLite (`~/.claude-memory/memory.db`)
2. **Semantic Indexing:** TF-IDF vectors generated for similarity search
3. **Knowledge Graph:** Entities extracted and nodes/edges created
4. **Pattern Learning:** Analyzes content for coding preferences (frameworks, style, testing)
5. **Full-Text Index:** FTS5 index updated for fast keyword search
6. **Timestamp:** Created timestamp recorded

## Advanced Usage

### Natural Language (in AI chat)

Most AI assistants will automatically invoke this skill when you say:
- "Remember that..."
- "Save this for later..."
- "I want to store..."
- "Keep track of..."

**Example in Cursor/Claude:**
```
You: "Remember that we decided to use React hooks over class components"
AI: [Automatically invokes slm-remember skill]
✓ Memory saved
```

### Bulk Import

Save multiple memories from a file:
```bash
# From text file (one memory per line)
while IFS= read -r line; do
  slm remember "$line" --project bulk-import
done < memories.txt

# From CSV (content,tags,project)
while IFS=',' read -r content tags project; do
  slm remember "$content" --tags "$tags" --project "$project"
done < memories.csv
```

### Integration with Git Hooks

**Pre-commit hook** (save commit messages):
```bash
#!/bin/bash
# .git/hooks/post-commit

commit_msg=$(git log -1 --pretty=%B)
commit_hash=$(git log -1 --pretty=%H)

slm remember "Commit: $commit_msg (${commit_hash:0:7})" \
  --tags git,commit \
  --project "$(basename $(git rev-parse --show-toplevel))"
```

## Error Handling

| Error | Cause | Solution |
|-------|-------|----------|
| "Database locked" | Another process accessing DB | Wait or `killall python3` |
| "Content cannot be empty" | Empty string passed | Provide content |
| "Invalid importance" | Value not 1-10 | Use number between 1-10 |
| "Database not found" | SuperLocalMemory not installed | Run `./install.sh` |

## Notes

- **100% local:** Nothing leaves your machine
- **Cross-tool sync:** All AI tools access same database (Cursor, ChatGPT, Claude, etc.)
- **Unlimited:** No memory limits, no quotas
- **Privacy:** Your data stays on your computer
- **Profiles:** Use `slm switch-profile` for profile isolation

## Related Commands

- `slm recall "<query>"` - Search memories semantically
- `slm list` - List recent memories
- `slm status` - Check system health
- `slm build-graph` - Rebuild knowledge graph
- `slm switch-profile <name>` - Switch memory profile

## Technical Details

**Database Schema:**
- Table: `memories`
- Fields: id, content, tags, project_name, importance, created_at, etc.
- Indexes: Full-text search (FTS5), TF-IDF vectors, timestamps

**Performance:**
- Add memory: ~50ms
- With knowledge graph: ~300ms
- Large content (10KB): ~1s

**Limits:**
- Max content size: 1MB
- Max tags: 50 per memory
- Max project name: 64 characters

---

**Created by:** [Varun Pratap Bhardwaj](https://github.com/varun369) (Solution Architect)
**Project:** SuperLocalMemory V2
**License:** MIT (see [LICENSE](../../LICENSE))
**Repository:** https://github.com/varun369/SuperLocalMemoryV2

*Open source doesn't mean removing credit. Attribution must be preserved per MIT License terms.*
