---
name: slm-switch-profile
description: Switch between memory profiles for context isolation and management. Use when the user wants to change profile context, separate work/personal memories, or manage multiple independent memory spaces. Each profile has its own database, graph, and patterns.
version: "3.4.23"
license: AGPL-3.0-or-later
compatibility: "Requires SuperLocalMemory V2 installed at ~/.claude-memory/"
attribution:
  creator: Varun Pratap Bhardwaj
  role: Solution Architect & Original Creator
  project: SuperLocalMemory V2
---

# SuperLocalMemory: Switch Profile

Switch between memory profiles to maintain separate contexts for different projects or use cases.

## Usage

```bash
slm switch-profile <name> [--create]
```

## What are Profiles?

**Profiles = Isolated Memory Spaces**

Each profile has its own:
- ✅ **Separate database** - Zero context bleeding
- ✅ **Independent knowledge graph** - Profile-specific relationships
- ✅ **Unique patterns** - Different coding preferences per profile
- ✅ **Isolated history** - No cross-contamination

**Think of profiles as workspaces:**
- `default` - General use
- `work` - Work projects
- `personal` - Personal projects
- `client-acme` - Specific client work
- `experiment` - Testing/learning

## Examples

### Example 1: Switch to Existing Profile
```bash
$ slm switch-profile work
```

**Output:**
```
🔄 Switching profile: default → work

Profile Information:
  Name: work
  Created: 2026-01-20
  Memories: 342
  Last Used: 2026-02-05 09:15

✅ Switched to profile "work"

Restart your AI tools (Cursor, ChatGPT, etc.) to use the new profile.
```

### Example 2: Create New Profile
```bash
$ slm switch-profile client-acme --create
```

**Output:**
```
🆕 Creating new profile: client-acme

Profile created successfully!
  Location: ~/.claude-memory/profiles/client-acme/
  Database: memory.db (empty)
  Status: Active

✅ Switched to profile "client-acme"

Next steps:
  • Start saving memories: slm remember "..."
  • AI tools will use this profile automatically
```

### Example 3: List All Profiles
```bash
$ slm list-profiles
```

**Output:**
```
📁 Available Profiles (4 total)

→ work (active)
  Created: 2026-01-20
  Memories: 342
  Size: 1.8 MB
  Last used: 2 minutes ago

  default
  Created: 2026-01-15
  Memories: 1,247
  Size: 4.2 MB
  Last used: 3 hours ago

  personal
  Created: 2026-01-22
  Memories: 89
  Size: 0.4 MB
  Last used: Yesterday

  client-acme
  Created: 2026-02-07
  Memories: 0
  Size: 0 KB
  Last used: Never
```

### Example 4: Return to Default
```bash
$ slm switch-profile default
```

**Always works** (default profile always exists)

## Arguments

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `<name>` | string | Yes | Profile name (lowercase, hyphens allowed) |
| `--create` | flag | No | Create profile if doesn't exist |
| `--force` | flag | No | Switch even if current profile has unsaved work |

## Profile Naming Rules

**Valid:**
- Lowercase letters: `a-z`
- Numbers: `0-9`
- Hyphens: `-`
- Length: 1-64 characters

**Examples:**
- ✅ `work`
- ✅ `client-acme`
- ✅ `project-2024`
- ✅ `personal`

**Invalid:**
- ❌ `Work` (uppercase)
- ❌ `client_acme` (underscore)
- ❌ `client acme` (space)
- ❌ `client.acme` (period)

## Use Cases

### 1. Work/Personal Separation
```bash
# Morning: Start work
slm switch-profile work
slm remember "Sprint planning: focus on auth system"

# Evening: Personal projects
slm switch-profile personal
slm remember "Learning Rust - ownership is tricky"
```

**Benefit:** No mixing of work and personal context

### 2. Multi-Client Consulting
```bash
# Client A project
slm switch-profile client-a
slm remember "Client A uses AWS, prefers TypeScript"

# Client B project
slm switch-profile client-b
slm remember "Client B uses Azure, prefers C#"
```

**Benefit:** Client-specific preferences and patterns

### 3. Experimentation
```bash
# Create throwaway profile for learning
slm switch-profile experiment --create
slm remember "Testing new framework..."

# Later: delete when done
slm delete-profile experiment
```

**Benefit:** Safe experimentation without polluting main memory

### 4. Project Isolation
```bash
# Different tech stacks
slm switch-profile backend-api  # Python/FastAPI
slm switch-profile frontend-app  # React/TypeScript
slm switch-profile ml-pipeline   # Python/PyTorch
```

**Benefit:** Pattern learning adapts to each stack

### 5. Team Shared Profiles (Advanced)
```bash
# Sync profile to team folder
slm export-profile work > team-shared/work-profile.db

# Team members import
slm import-profile team-shared/work-profile.db --name work
```

**Benefit:** Shared knowledge base for team

## How Switching Works

### Behind the Scenes

**1. Profile Structure:**
```
~/.claude-memory/
├── memory.db               ← Current active profile (symlink)
├── profiles/
│   ├── default/
│   │   └── memory.db       ← Default profile database
│   ├── work/
│   │   └── memory.db       ← Work profile database
│   └── personal/
│       └── memory.db       ← Personal profile database
└── current_profile         ← Text file with active profile name
```

**2. Switching Process:**
```bash
$ slm switch-profile work
```

- **Step 1:** Verify "work" profile exists
- **Step 2:** Update symlink: `memory.db` → `profiles/work/memory.db`
- **Step 3:** Write "work" to `current_profile` file
- **Step 4:** Notify user to restart AI tools

**3. AI Tools Read:**
- MCP server reads `~/.claude-memory/memory.db`
- Symlink automatically points to active profile
- Transparent to AI tools

## Important Notes

### Restart Required
**After switching, restart your AI tools:**
- **Cursor:** Cmd+Q, reopen
- **Claude Desktop:** Quit, reopen
- **ChatGPT:** Close, reopen
- **VS Code:** Reload window

**Why:** Tools cache database connection at startup

### Profile Isolation is Complete
**Each profile has:**
- ❌ **NO access** to other profiles' memories
- ❌ **NO shared** knowledge graph
- ❌ **NO cross-profile** pattern learning

**This is by design** for true isolation.

### Default Profile Special
- Always exists
- Cannot be deleted
- Safe fallback

## Advanced Usage

### Profile Management

**Create multiple profiles:**
```bash
slm switch-profile work --create
slm switch-profile personal --create
slm switch-profile client-a --create
```

**Backup profile:**
```bash
cp -r ~/.claude-memory/profiles/work ~/.claude-memory/backups/work-2026-02-07
```

**Delete profile:**
```bash
slm delete-profile experiment
# Warning: This permanently deletes all memories!
```

**Rename profile:**
```bash
slm rename-profile old-name new-name
```

### Scripting & Automation

**Auto-switch by project directory:**
```bash
# Add to .bashrc or .zshrc
function cd() {
  builtin cd "$@"

  # Check for .slm-profile file
  if [ -f ".slm-profile" ]; then
    profile=$(cat .slm-profile)
    slm switch-profile "$profile" --quiet
  fi
}
```

**Usage:**
```bash
# In project directory
echo "myapp" > .slm-profile

# CD into directory
cd ~/projects/myapp
# Automatically switches to "myapp" profile
```

**Daily profile switching:**
```bash
#!/bin/bash
# cron: 0 9 * * 1-5 (Mon-Fri at 9 AM)

slm switch-profile work
echo "Switched to work profile" | notify-send "SuperLocalMemory"
```

### Profile Statistics

**Compare profiles:**
```bash
slm list-profiles --stats
```

**Output:**
```
Profile          Memories  Size     Nodes  Edges  Patterns
default          1,247     4.2 MB   892    2,134  34
work             342       1.8 MB   256    789    12
personal         89        0.4 MB   67     145    4
```

**Export profile summary:**
```bash
slm profile-summary work > work-summary.txt
```

## Troubleshooting

### "Profile not found"

**Cause:** Typo or profile doesn't exist

**Solution:**
```bash
# List existing profiles
slm list-profiles

# Create if needed
slm switch-profile myprofile --create
```

### "Switch failed: Database locked"

**Cause:** AI tool still using old profile

**Solution:**
```bash
# Kill all MCP servers
killall python3

# Try again
slm switch-profile work

# Restart AI tools
```

### "Memories disappeared after switching"

**Cause:** Switched to empty/different profile (working as intended)

**Solution:**
```bash
# Check current profile
slm status | grep "Current Profile"

# Switch back to profile with memories
slm switch-profile default

# Or list all profiles to find the right one
slm list-profiles
```

### "AI tools still see old profile"

**Cause:** Didn't restart tools

**Solution:**
```bash
# Must completely restart (not just reload)
# macOS: Cmd+Q, then reopen
# Windows/Linux: Close all windows, reopen
```

## Performance

| Operation | Time | Notes |
|-----------|------|-------|
| Switch profile | ~100ms | Instant |
| Create profile | ~200ms | Fast |
| List profiles | ~50ms | Very fast |

**Switching is fast** because it's just updating a symlink.

## Notes

- **No data loss:** Switching never deletes memories
- **Profile persistence:** Memories stay in original profile
- **Tool transparency:** AI tools don't need special config per profile
- **Backup-friendly:** Each profile is separate directory

## Related Commands

- `slm list-profiles` - Show all profiles
- `slm delete-profile <name>` - Delete a profile
- `slm rename-profile <old> <new>` - Rename profile
- `slm status` - Shows current profile
- `slm export-profile <name>` - Export to file
- `slm import-profile <file>` - Import from file

---

**Created by:** [Varun Pratap Bhardwaj](https://github.com/varun369) (Solution Architect)
**Project:** SuperLocalMemory V2
**License:** MIT (see [LICENSE](../../LICENSE))
**Repository:** https://github.com/varun369/SuperLocalMemoryV2

*Open source doesn't mean removing credit. Attribution must be preserved per MIT License terms.*
