# SuperLocalMemory V2 — Visual Assets

This directory contains all visual documentation, screenshots, GIFs, videos, and thumbnails for SuperLocalMemory V2.

## Directory Structure

```
assets/
├── screenshots/          # Static image documentation
│   ├── installation/     # Installation process screenshots
│   ├── cli/              # CLI usage and commands
│   ├── dashboard/        # Web dashboard UI
│   ├── v25/              # v2.5.0 feature screenshots
│   ├── ide/              # IDE integration examples (Cursor, Windsurf, VS Code)
│   ├── graph/            # Knowledge graph visualization
│   └── misc/             # Miscellaneous images
├── gifs/                 # Animated GIFs (demos, tutorials)
├── videos/               # Video files (.mp4, .webm, etc.)
└── thumbnails/           # Compressed thumbnails for web
```

## Guidelines

### Screenshots (`screenshots/`)

**Naming convention:** `{feature}-{number}.png`
- `installation-01-download.png`
- `cli-01-remember.png`
- `dashboard-01-home.png`
- `v25-01-event-bus.png`
- `ide-01-cursor-skills.png`
- `graph-01-clusters.png`

**Specifications:**
- Format: PNG (lossless)
- Dimensions: 1280×720 (16:9) or 1440×900 for detailed UI
- Quality: Native resolution (no upscaling)
- Color space: sRGB
- Include cursor/annotations for clarity when helpful

### GIFs (`gifs/`)

**Naming convention:** `{feature}-demo.gif`
- `installation-demo.gif`
- `cli-remember-demo.gif`
- `dashboard-search.gif`
- `pattern-learning-demo.gif`

**Specifications:**
- Format: GIF (animated)
- Duration: 5-15 seconds
- Frame rate: 10-15 FPS (balance file size and smoothness)
- Max file size: 10 MB
- Dimensions: 1280×720
- Loop: Infinite

### Videos (`videos/`)

**Naming convention:** `{feature}-{type}.mp4`
- `installation-walkthrough.mp4`
- `dashboard-tour.mp4`

**Specifications:**
- Format: MP4 (H.264 codec)
- Resolution: 1280×720 (720p) or 1920×1080 (1080p)
- Frame rate: 30 FPS
- Bitrate: 5000 kbps (720p) or 8000 kbps (1080p)
- Audio: AAC, 128 kbps, 48 kHz
- Duration: 1-5 minutes per video

### Thumbnails (`thumbnails/`)

**Naming convention:** `{original-filename}-thumb.png`
- `installation-01-thumb.png`
- `cli-01-thumb.png`
- `dashboard-tour-thumb.png`

**Specifications:**
- Format: PNG
- Dimensions: 320×180 (16:9 aspect ratio)
- File size: < 50 KB
- Generated from original screenshots/videos
- Used for landing pages, GitHub README, docs

## Usage

### In Markdown Files

```markdown
# Installation Guide

![Installation Step 1](../assets/screenshots/installation/installation-01-download.png)

## Dashboard Overview

![Dashboard Home](../assets/screenshots/dashboard/dashboard-01-home.png)
```

### In Website (Astro)

```astro
---
import dashboardImg from '../assets/screenshots/dashboard/dashboard-01-home.png';
---

<img src={dashboardImg} alt="Dashboard home view" />
```

### In Documentation

- Place links relative to the doc file
- Use descriptive alt text (required for accessibility)
- Include captions where helpful
- Organize by feature/version

## Version Organization

**v2.4.x and earlier:** `screenshots/`
**v2.5.0 features:** `screenshots/v25/`
**v2.6.0+ features:** Create new `screenshots/v26/` as needed

## Maintenance

- **Outdated images:** Move to `.backup/assets/` if no longer relevant
- **Superseded versions:** Update alt-text in docs, link to new screenshots
- **File size control:** Use tools like TinyPNG for PNG compression
- **Consistency:** Keep aspect ratios and styling consistent across screenshots

## Tools & Workflow

### Recommended Tools

1. **Screenshots:**
   - macOS: CMD+Shift+4 (region) or Screenshot.app
   - Windows: Snipping Tool or ShareX
   - Linux: GNOME Screenshot or Flameshot
   - Annotation: Annotate (macOS), Paint.NET (Windows), GIMP (Linux/Cross)

2. **GIFs:**
   - ScreenFlow (macOS) or Camtasia (cross-platform)
   - Convert to GIF: ffmpeg or ezgif.com
   - Optimize: gifsicle or ImageOptim

3. **Videos:**
   - Recording: ScreenFlow (macOS), Camtasia, OBS Studio (free)
   - Editing: Final Cut Pro, Adobe Premiere, DaVinci Resolve (free)
   - Encoding: ffmpeg

4. **Thumbnails:**
   - Batch resize: ImageMagick or sips (macOS)
   - Optimize: TinyPNG, ImageOptim, pngquant

### Batch Processing

```bash
# Resize all images to thumbnail size (320×180)
sips -Z 320 *.png --out thumbnails/

# Optimize PNG files
pngquant --quality=70-90 *.png

# Convert screenshot to GIF
ffmpeg -i recording.mov -r 10 -vf scale=1280:-1 output.gif
```

## Accessibility (WCAG 2.1)

- All images must have descriptive alt text
- GIFs must include a pause/play button (video tag)
- Use sufficient color contrast in annotations
- Avoid text in images (use captions instead)
- Provide transcripts for videos

## Legal & Licensing

- All screenshots are derived from SuperLocalMemory V2 (GNU Affero General Public License v3.0 or later)
- Original artwork or third-party images require explicit licensing
- Include copyright/attribution in the filename or metadata if applicable
- Do not commit proprietary screenshots (customer data, API keys, etc.)

## Common Scenarios

### Adding a new feature screenshot

1. Create the feature in your dev environment
2. Take a clean screenshot (clear background, no clutter)
3. Save as `screenshots/{category}/{feature}-{number}.png`
4. Create thumbnail in `thumbnails/`
5. Document usage in relevant docs
6. Commit with clear message: `docs: Add screenshot for feature-X`

### Creating a demo GIF

1. Record with ScreenFlow or similar (5-15 seconds)
2. Export as MP4
3. Convert to GIF with `ffmpeg -i video.mp4 -r 12 output.gif`
4. Optimize: `gifsicle --optimize=3 --colors 128 output.gif -o output-opt.gif`
5. Save to `gifs/feature-demo.gif`
6. Embed in documentation with `![Demo](../assets/gifs/feature-demo.gif)`

---

**Last Updated:** February 12, 2026
**Maintainer:** SuperLocalMemory V2 Project
