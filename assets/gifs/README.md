# SuperLocalMemory V2 - Animated GIFs

Professional animated demonstrations of key features.

## Files

### 1. cli-demo.gif (0.65 MB)
**Demonstrates:** CLI commands in action
- `slm remember` → `slm recall` → `slm list` → loop
- **Specs:** 800x500px, 12 FPS
- **Use case:** README, CLI documentation

### 2. dashboard-search.gif (0.68 MB)
**Demonstrates:** Dashboard search/filter interaction
- Progressive reveal of search results
- **Specs:** 1200x700px, 15 FPS
- **Use case:** Web dashboard documentation

### 3. graph-interaction.gif (3.14 MB)
**Demonstrates:** Knowledge graph exploration
- Zoom in → explore cluster → zoom out → loop
- **Specs:** 1200x700px, 12 FPS
- **Use case:** Knowledge graph features, visual demos

### 4. event-stream.gif (1.30 MB)
**Demonstrates:** Real-time event stream (v2.5)
- Live events scrolling and updating
- **Specs:** 1100x640px, 64 colors, optimized
- **Use case:** v2.5 Event Bus feature demos

### 5. dashboard-tabs.gif (2.12 MB)
**Demonstrates:** Dashboard navigation
- Overview → Events → Memories → Graph → Timeline → loop
- **Specs:** 1400x900px, 12 FPS
- **Use case:** Feature overview, dashboard walkthrough

## Usage

All GIFs loop seamlessly and are optimized for web/documentation use.

**In Markdown:**
```markdown
![CLI Demo](assets/gifs/cli-demo.gif)
```

**In HTML:**
```html
<img src="assets/gifs/cli-demo.gif" alt="CLI Demo" width="800">
```

## Technical Details

- **Format:** Animated GIF with optimization
- **Compression:** Adaptive palette, color reduction, frame optimization
- **Source:** Generated from screenshots in `assets/screenshots/`
- **Generator:** `create_gifs.py`, `optimize_gifs.py`

## Regeneration

To regenerate all GIFs from the repository root:
```bash
python3 tools/create_gifs.py
python3 tools/optimize_gifs.py
```

For aggressive event stream optimization:
```bash
python3 tools/optimize_event_stream.py
```

---

**Copyright © 2026 Varun Pratap Bhardwaj**
Licensed under GNU Affero General Public License v3.0 or later
