# SuperLocalMemory V2 - Dashboard Screenshots

**Generated:** February 12, 2026
**Profile:** demo-visual (synthetic data)
**Resolution:** 1920x1080 @ 2x DPI
**Format:** PNG, lossless

## Purpose

These screenshots document the SuperLocalMemory V2 web dashboard interface using synthetic, non-sensitive demo data. They are intended for:

- Documentation (GitHub README, Wiki)
- Marketing materials (website, blog posts)
- User guides and tutorials
- Project presentations

## Demo Data

The screenshots use a dedicated `demo-visual` profile containing 30 synthetic memories about software development best practices. Topics include:

- **API Development:** FastAPI, JWT authentication, rate limiting, versioning
- **Database:** PostgreSQL, Alembic migrations, connection pooling
- **Frontend:** React functional components, Zustand state management, Vite bundling
- **DevOps:** Docker Compose, GitHub Actions CI/CD, AWS deployment
- **Security:** Environment variables, CORS, security headers, input validation
- **Monitoring:** CloudWatch, Datadog, logging strategies
- **Testing:** pytest, Jest, load testing with Locust
- **Architecture:** Celery background tasks, Redis caching, WebSocket real-time updates

**Knowledge Graph Statistics:**
- 30 memories
- 30 graph nodes
- 34 edges (relationships)
- 6 clusters (semantic groupings)

## Screenshot Inventory

### Light Mode Screenshots

| Filename | View | Description | Size |
|----------|------|-------------|------|
| `dashboard-overview.png` | Knowledge Graph | Main dashboard with stats cards and force-directed graph visualization | 519 KB |
| `dashboard-live-events.png` | Live Events | Real-time event stream showing memory operations as they happen | 669 KB |
| `dashboard-agents.png` | Agents | Connected agents/clients and their activity statistics | 592 KB |
| `dashboard-memories.png` | Memories | Searchable list of all memories with tags, importance, and timestamps | 803 KB |
| `dashboard-clusters.png` | Clusters | Knowledge graph clusters with summaries and member counts | 546 KB |
| `dashboard-patterns.png` | Patterns | Learned coding patterns and preferences (Layer 4 feature) | 458 KB |
| `dashboard-timeline.png` | Timeline | Chronological view of memory creation over time | 456 KB |
| `dashboard-graph.png` | Knowledge Graph | Interactive graph explorer showing semantic relationships | 519 KB |
| `dashboard-filtered.png` | Memories (filtered) | Memories list with search filter applied ("api") | 803 KB |

### Dark Mode Screenshots

| Filename | View | Description | Size |
|----------|------|-------------|------|
| `dashboard-overview-dark.png` | Knowledge Graph | Main dashboard in dark mode | 520 KB |
| `dashboard-live-events-dark.png` | Live Events | Event stream in dark mode | 669 KB |
| `dashboard-memories-dark.png` | Memories | Memory list in dark mode | 803 KB |

**Total:** 12 screenshots, ~7.1 MB

## Key Features Visible

### Statistics Cards (Top Row)
- **Total Memories:** 30
- **Clusters:** 6 semantic groupings
- **Graph Nodes:** 30 entities
- **Connections:** 34 relationships

### Navigation Tabs
1. **Knowledge Graph** - Interactive force-directed graph visualization
2. **Memories** - Searchable table with filters and sorting
3. **Clusters** - Community detection results with summaries
4. **Patterns** - Learned coding preferences (Layer 4)
5. **Timeline** - Chronological memory view
6. **Live Events** - Real-time operation stream (Layer 10)
7. **Agents** - Connected clients (MCP, CLI, API)
8. **Settings** - Configuration and preferences

### Notable UI Elements
- Profile switcher (top right): "demo-visual (30)"
- Knowledge Graph Explorer button (top right)
- Dark mode toggle (top right, moon icon)
- Refresh button on graph view
- Search/filter inputs on Memories tab
- Color-coded event types (Live Events)
- Importance indicators (star ratings 1-10)
- Tag pills with color coding
- Timestamps and metadata

## Usage Instructions

### For Documentation

**GitHub README:**
```markdown
![SuperLocalMemory Dashboard](assets/screenshots/dashboard/dashboard-overview.png)
```

**Website/Marketing:**
- Use `dashboard-overview.png` for landing page hero section
- Use `dashboard-memories.png` to show memory management features
- Use `dashboard-live-events.png` to demonstrate real-time capabilities

**User Guides:**
- Use `dashboard-filtered.png` to explain search functionality
- Use `dashboard-clusters.png` to explain knowledge graph clustering
- Use `dashboard-patterns.png` to explain pattern learning features

### For Presentations

Recommended sequence for product demos:
1. `dashboard-overview.png` - Start with the big picture (graph + stats)
2. `dashboard-memories.png` - Show the memory list and organization
3. `dashboard-live-events.png` - Demonstrate real-time updates
4. `dashboard-graph.png` - Deep dive into knowledge graph relationships

### Regenerating Screenshots

If you need to regenerate these screenshots with updated UI:

```bash
# 1. Create demo data (30 synthetic memories)
python3 create-demo-data.py

# 2. Start the dashboard on demo-visual profile
python3 ~/.claude-memory/ui_server.py
# (Dashboard runs on http://localhost:8765)

# 3. Capture screenshots
python3 capture-screenshots.py

# 4. Switch back to your main profile
python3 src/memory-profiles.py switch default
```

## Scripts

Two Python scripts are included in the repository root for regenerating this content:

### `create-demo-data.py`
- Creates a `demo-visual` profile
- Populates it with 30 synthetic software development memories
- Builds the knowledge graph (6 clusters)
- Safe to run multiple times (replaces existing demo data)

### `capture-screenshots.py`
- Uses Playwright to automate browser screenshots
- Captures all dashboard views in light mode
- Captures key views in dark mode
- Handles tab navigation and waiting for content to load
- Outputs to `assets/screenshots/dashboard/`

**Requirements:**
```bash
pip install playwright
python -m playwright install chromium
```

## Notes

- All data in these screenshots is synthetic and non-sensitive
- No real API keys, passwords, or proprietary information
- The `demo-visual` profile is separate from your working profile
- Switching profiles does not affect your main memory database
- Screenshots were captured at 2x DPI for retina display quality

## Attribution

SuperLocalMemory V2
Copyright © 2026 Varun Pratap Bhardwaj
Licensed under GNU Affero General Public License v3.0 or later

Creator: Varun Pratap Bhardwaj
Repository: https://github.com/qualixar/superlocalmemory
