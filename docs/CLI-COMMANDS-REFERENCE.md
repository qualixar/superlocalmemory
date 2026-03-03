# SuperLocalMemory V2 - CLI Commands Reference

**Quick reference for all CLI commands**

**Version 2.4.1** - Universal integration across 16+ IDEs and CLI tools

SuperLocalMemory V2 offers three access methods:
1. **Universal CLI** - Simple `slm` commands (NEW in v2.1.0)
2. **Original CLI** - Full `superlocalmemoryv2-*` commands
3. **MCP/Skills** - IDE-specific integration (auto-configured)

All methods use the same local SQLite database.

---

## 🎯 Universal CLI Commands (NEW)

**Simple syntax for everyday use:**

### Memory Operations

```bash
# Save memory
slm remember "content" --tags tag1,tag2

# Search memories
slm recall "query"

# List recent memories
slm list

# System status
slm status

# Get project context
slm context
```

### Profile Management

```bash
# List profiles
slm profile list

# Create profile
slm profile create work --description "Work projects"

# Switch profile
slm profile switch work

# Delete profile
slm profile delete old-project

# Show current profile
slm profile current
```

### Knowledge Graph

```bash
# Build graph
slm graph build

# View statistics
slm graph stats

# Find related memories
slm graph related --id 5

# View cluster
slm graph cluster --id 1
```

### Pattern Learning

```bash
# Update patterns
slm patterns update

# List patterns
slm patterns list 0.5

# Get coding identity
slm patterns context 0.7

# Pattern statistics
slm patterns stats
```

### Aider Integration

```bash
# Launch Aider with auto-context injection
aider-smart "Add authentication to the API"

# Regular Aider works too
aider
```

---

## 📊 Original CLI Commands (Still Supported)

All original commands continue to work unchanged. Use these if you prefer explicit paths or are scripting.

### Check System Status
```bash
superlocalmemoryv2-status
# OR
~/.claude-memory/bin/superlocalmemoryv2-status
```
Shows: total memories, graph stats, clusters, patterns, database size

---

## 👤 Profile Management (Original Commands)

**Default profile is always available.** Use profiles to maintain separate memory contexts for different projects, clients, or AI personalities.

### List All Profiles
```bash
superlocalmemoryv2-profile list
# OR
~/.claude-memory/bin/superlocalmemoryv2-profile list
```
Shows all profiles with active marker (→)

### Show Current Active Profile
```bash
superlocalmemoryv2-profile current
# OR
~/.claude-memory/bin/superlocalmemoryv2-profile current
```
Shows current profile details and memory count

### Create New Profile
```bash
# Empty profile
superlocalmemoryv2-profile create work --description "Work projects"

# Copy from current profile
superlocalmemoryv2-profile create personal --from-current
```

### Switch Profile
```bash
superlocalmemoryv2-profile switch work
```
**⚠️ Restart your IDE after switching if using MCP integration.**

### Delete Profile
```bash
superlocalmemoryv2-profile delete old-project
```
- Cannot delete "default" profile
- Cannot delete currently active profile
- Creates backup before deletion
- Requires typing profile name to confirm

---

## 🔄 Reset Commands (Destructive - Use with Caution)

### Check Status First
```bash
slm status
# OR
superlocalmemoryv2-status
```
**Always check status before any reset operation!**

### Soft Reset
```bash
superlocalmemoryv2-reset soft
# OR
~/.claude-memory/bin/superlocalmemoryv2-reset soft
```
- Clears all memories from current profile
- Clears graph, patterns, tree structure
- Keeps V2 schema intact
- Creates automatic backup
- Prompt: "Proceed with soft reset? (yes/no)"

### Hard Reset
```bash
superlocalmemoryv2-reset hard --confirm
# OR
~/.claude-memory/bin/superlocalmemoryv2-reset hard --confirm
```
- ⚠️ **NUCLEAR OPTION** ⚠️
- Deletes entire database file
- Reinitializes fresh V2 schema
- Creates automatic backup
- Prompt: Type "DELETE EVERYTHING" to confirm

### Layer Reset (Surgical)
```bash
superlocalmemoryv2-reset layer --layers graph patterns
# OR
~/.claude-memory/bin/superlocalmemoryv2-reset layer --layers graph patterns
```
Available layers:
- `graph` - Clear graph nodes, edges, clusters
- `patterns` - Clear learned identity patterns
- `tree` - Clear hierarchical tree structure
- `archive` - Clear compressed memory archives

Prompt: "Proceed with layer reset? (yes/no)"

---

## 🔍 Graph Operations

### Build Knowledge Graph
```bash
~/.claude-memory/venv/bin/python ~/.claude-memory/graph_engine.py build
```
Builds graph from all memories using TF-IDF + Leiden clustering

### Show Graph Statistics
```bash
~/.claude-memory/venv/bin/python ~/.claude-memory/graph_engine.py stats
```
Shows: nodes, edges, clusters with names and member counts

### Find Related Memories
```bash
~/.claude-memory/venv/bin/python ~/.claude-memory/graph_engine.py related --memory-id 5
```
Shows memories connected in the knowledge graph

### View Cluster Members
```bash
~/.claude-memory/venv/bin/python ~/.claude-memory/graph_engine.py cluster --cluster-id 1
```
Lists all memories in a specific cluster

---

## 🧠 Pattern Learning

### Update Patterns
```bash
~/.claude-memory/venv/bin/python ~/.claude-memory/pattern_learner.py update
```
Analyzes all memories and learns new identity patterns

### List Learned Patterns
```bash
~/.claude-memory/venv/bin/python ~/.claude-memory/pattern_learner.py list 0.1
```
Shows patterns with confidence >= 0.1 (10%)

### Get Claude Context
```bash
~/.claude-memory/venv/bin/python ~/.claude-memory/pattern_learner.py context 0.7
```
Outputs patterns formatted for Claude with confidence >= 0.7 (70%)

### Show Pattern Statistics
```bash
~/.claude-memory/venv/bin/python ~/.claude-memory/pattern_learner.py stats
```
Shows pattern count, avg confidence, types, categories

---

## 📦 Traditional Memory Operations

### Add Memory
```bash
python ~/.claude-memory/memory_store_v2.py add "content" --tags tag1,tag2
```

### Search Memories
```bash
python ~/.claude-memory/memory_store_v2.py search "query"
```

### List Recent Memories
```bash
python ~/.claude-memory/memory_store_v2.py list 20
```

### Get Stats
```bash
python ~/.claude-memory/memory_store_v2.py stats
```

---

## ⚙️ Adding Commands to Shell

### Option 1: Bash Aliases
Add to `~/.bashrc` or `~/.zshrc`:

```bash
# SuperLocalMemory V2 aliases
alias memory-status='~/.claude-memory/bin/memory-status'
alias memory-reset='~/.claude-memory/bin/memory-reset'
alias memory-profile='~/.claude-memory/bin/memory-profile'
alias memory-graph='~/.claude-memory/venv/bin/python ~/.claude-memory/graph_engine.py'
alias memory-patterns='~/.claude-memory/venv/bin/python ~/.claude-memory/pattern_learner.py'
```

Then reload: `source ~/.bashrc` or `source ~/.zshrc`

### Option 2: Add to PATH
Add to `~/.bashrc` or `~/.zshrc`:

```bash
export PATH="$HOME/.claude-memory/bin:$PATH"
```

Then reload and use short commands:
```bash
memory-status
memory-reset status
memory-profile list
```

---

## 🔐 Safety Features

All destructive commands include:
- ✅ Confirmation prompts
- ✅ Automatic backups
- ✅ Clear warnings
- ✅ Cannot delete default profile
- ✅ Cannot delete active profile

### Backup Location
```bash
~/.claude-memory/backups/
```

Format: `pre-reset-YYYYMMDD-HHMMSS.db`

### Restore from Backup
```bash
cp ~/.claude-memory/backups/pre-reset-20260205-143000.db \
   ~/.claude-memory/memory.db
```

---

## 📝 Help Commands

### Memory Status
```bash
~/.claude-memory/bin/memory-status
# No --help flag, just shows status
```

### Memory Reset
```bash
~/.claude-memory/bin/memory-reset --help
```

### Memory Profile
```bash
~/.claude-memory/bin/memory-profile --help
```

---

## 🎯 Common Workflows

### Daily Usage
```bash
# Standalone: Use Python commands directly
python ~/.claude-memory/memory_store_v2.py add "content" --tags tag1

# Optional Claude CLI: Use /remember skill
# /remember "content"
```

### Weekly Maintenance
```bash
# Rebuild graph (as memories grow)
~/.claude-memory/venv/bin/python ~/.claude-memory/graph_engine.py build

# Update patterns
~/.claude-memory/venv/bin/python ~/.claude-memory/pattern_learner.py update
```

### Switching Contexts
```bash
# Check current profile
memory-profile current

# Switch to work profile
memory-profile switch work

# If using Claude CLI integration, restart it
# If using standalone, no restart needed

# Verify switch
memory-profile current
```

### Fresh Start
```bash
# Check what you have
memory-status

# Soft reset (keeps schema)
memory-reset soft
# Type: yes

# Or create new profile instead
memory-profile create fresh
memory-profile switch fresh
```

---

**All commands are safe by default with warnings and confirmations for destructive operations.**

For complete documentation, see:
- `~/.claude-memory/docs/README.md` - Main documentation
- `~/.claude-memory/RESET-GUIDE.md` - Reset procedures
- `~/.claude-memory/PROFILES-GUIDE.md` - Profile management
- `docs/ARCHITECTURE.md` - All features
