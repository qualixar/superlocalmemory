---
name: slm-show-patterns
description: Show what SuperLocalMemory has learned about your preferences, workflow patterns, and project context. Use when the user asks "what have you learned about me?" or wants to see their coding identity patterns. Shows tech preferences, workflow sequences, and engagement health.
version: "2.7.0"
license: AGPL-3.0-or-later
compatibility: "Requires SuperLocalMemory V2.7+ with learning features"
attribution:
  creator: Varun Pratap Bhardwaj
  role: Solution Architect & Original Creator
  project: SuperLocalMemory V2
---

# SuperLocalMemory: Show Patterns

Show what SuperLocalMemory has learned about your preferences, workflow, and coding identity.

## Usage

```bash
slm patterns list [threshold]
slm learning status
slm engagement
```

## Example Output

### Learned Patterns
```bash
$ slm patterns list 0.5
```

**Output:**
```
Learned Patterns (confidence >= 0.5)
=====================================

TECH PREFERENCES (cross-project):
  #1  preferred_framework: React        confidence: 0.92  (seen in 3 profiles)
  #2  preferred_language: Python         confidence: 0.88  (seen in 2 profiles)
  #3  preferred_backend: FastAPI         confidence: 0.85  (seen in 2 profiles)
  #4  testing_style: pytest              confidence: 0.78  (seen in 1 profile)
  #5  preferred_db: PostgreSQL           confidence: 0.71  (seen in 2 profiles)

WORKFLOW PATTERNS:
  #6  morning_sequence: recall -> code -> remember    frequency: 34
  #7  debug_sequence: recall -> recall -> remember    frequency: 18
  #8  review_sequence: list -> recall -> remember     frequency: 12

PROJECT CONTEXT:
  #9  active_project: SuperLocalMemoryV2    last_seen: 2 hours ago
  #10 active_project: client-dashboard      last_seen: 1 day ago

Total: 10 patterns (7 high confidence)
```

### Learning Status
```bash
$ slm learning status
```

**Output:**
```
SuperLocalMemory v2.7 -- Learning System Status
==================================================
LightGBM:  installed (4.5.0)
SciPy:     installed (1.14.1)
ML Ranking:  available
Full Learning: available

Feedback signals: 247
Unique queries:   89
Patterns learned: 34 (12 high confidence)
Workflow patterns: 8
Sources tracked:  4
Models trained:   2
Learning DB size: 128 KB
```

### Engagement Metrics
```bash
$ slm engagement
```

**Output:**
```
SuperLocalMemory -- Engagement Health
======================================
Status: HEALTHY

This Week:
  Memories saved:    12
  Recalls performed: 28
  Memories marked useful: 8
  Feedback ratio:    28.6%

Trends:
  Recall frequency:  increasing (up 15% from last week)
  Save frequency:    stable
  Useful feedback:   increasing (up 40% from last week)

Streaks:
  Current daily streak: 5 days
  Longest streak:       14 days
```

## What the Patterns Mean

### Tech Preferences
Cross-project patterns that transfer between profiles. These represent your coding identity -- which frameworks, languages, and tools you consistently choose.

**How they are learned:**
- Extracted from memory content (mentions of frameworks, tools)
- Weighted by recency and frequency
- Confidence increases when the same preference appears across multiple profiles

**How they help:**
- When you `recall`, results matching your preferred stack rank higher
- Your AI tools can reference these to tailor suggestions

### Workflow Patterns
Sequences of actions you repeat. The system learns your work rhythm.

**Examples:**
- `recall -> code -> remember` = "Research, build, document" workflow
- `recall -> recall -> remember` = "Deep investigation" workflow

**How they help:**
- System predicts what you will need next
- Can pre-load relevant context based on your current workflow stage

### Engagement Health
Overall system usage metrics (fully local, zero telemetry).

**Healthy indicators:**
- Regular daily usage (streaks)
- Balanced save/recall ratio
- Increasing useful feedback

**Warning signs:**
- No recalls for 7+ days = stale memories
- No saves for 7+ days = not capturing knowledge
- Zero feedback = system cannot learn your preferences

## Correcting Patterns

If the system learned something wrong, correct it:

```bash
# See all patterns
slm patterns list

# Correct pattern #3 from "FastAPI" to "Django"
slm patterns correct 3 Django
```

The correction increases confidence to 1.0 and records the change history.

## Options

| Command | Description | Use Case |
|---------|-------------|----------|
| `slm patterns list` | All patterns (no threshold) | See everything learned |
| `slm patterns list 0.7` | High confidence only | See reliable patterns |
| `slm patterns correct <id> <value>` | Fix a wrong pattern | Override incorrect learning |
| `slm learning status` | System health | Check deps and stats |
| `slm learning retrain` | Force model retrain | After bulk feedback |
| `slm learning reset` | Delete all learning data | Fresh start (memories preserved) |
| `slm engagement` | Usage metrics | Track your engagement health |

## Use Cases

### 1. "What Have You Learned About Me?"
```bash
slm patterns list
# Shows all preferences, workflows, and project context
```

### 2. Pre-Session Context Loading
```bash
slm patterns context
# Returns structured context for AI tools to consume
```

### 3. Onboarding a New AI Tool
```bash
slm learning status
# Verify learning is active, then use your existing memories
# New tool benefits from all previously learned patterns
```

### 4. Weekly Review
```bash
slm engagement
# Check if you are using your memory system effectively
```

## Requirements

- SuperLocalMemory V2.7+
- Optional: `lightgbm` and `scipy` for ML-powered ranking
- Works without optional deps (uses rule-based ranking as fallback)

## Notes

- **Privacy:** All learning is local. Zero telemetry, zero cloud calls.
- **Separate storage:** Learning data lives in `learning.db`, separate from `memory.db`.
- **Non-destructive:** `slm learning reset` only deletes learning data, never memories.
- **Graceful degradation:** If learning deps are missing, core features work normally.

## Related Commands

- `slm recall` - Search memories (results ranked by learned patterns)
- `slm useful <id>` - Mark memory as useful (feedback for learning)
- `slm status` - Overall system status
- `slm patterns update` - Re-learn patterns from existing memories

---

**Created by:** [Varun Pratap Bhardwaj](https://github.com/varun369) (Solution Architect)
**Project:** SuperLocalMemory V2
**License:** MIT (see [LICENSE](../../LICENSE))
**Repository:** https://github.com/varun369/SuperLocalMemoryV2

*Open source doesn't mean removing credit. Attribution must be preserved per MIT License terms.*
