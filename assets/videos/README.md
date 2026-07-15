# SuperLocalMemory V2 - Video Assets

Silent video slideshows for documentation and marketing.

## Generated Videos

### 1. installation-walkthrough.mp4
- **Duration:** 1:10 (70 seconds)
- **Size:** 510 KB
- **Content:** Step-by-step installation guide
  - Title slide
  - Clone repository
  - Run installer
  - Verify installation
  - Create first memory
  - Success screen
- **Use cases:** Documentation, onboarding, GitHub README

### 2. quick-start.mp4
- **Duration:** 1:43 (103 seconds)
- **Size:** 783 KB
- **Content:** Quick start tutorial
  - Title slide
  - CLI: Remember command
  - CLI: Recall command
  - CLI: Build graph command
  - Dashboard overview
  - Graph visualization
  - Success screen
- **Use cases:** Documentation, tutorials, feature demos

### 3. dashboard-tour.mp4
- **Duration:** 1:39 (99 seconds)
- **Size:** 1.3 MB
- **Content:** Dashboard feature walkthrough
  - Title slide
  - Overview stats
  - Live events (v2.5)
  - Agents tab (v2.5)
  - Memories list
  - Knowledge graph
  - Clusters view
  - Patterns view
  - Dark mode
  - End screen
- **Use cases:** Marketing, feature showcase, v2.5 announcements

## Technical Specifications

- **Resolution:** 1920x1080 (Full HD)
- **Codec:** H.264 (High Profile)
- **Frame Rate:** 30 FPS
- **Pixel Format:** yuv420p (progressive)
- **Audio:** None (silent)
- **Transitions:** 0.5s fade in/out
- **Text Overlays:** White text with black shadow on semi-transparent background

## Generation

Videos are generated using `../screenshots/generate-videos.py`:

```bash
cd ../screenshots
python3 generate-videos.py
```

**Dependencies:**
- Python 3.8+
- Pillow (PIL)
- ffmpeg

**Process:**
1. Creates title slides with centered text
2. Adds text overlays to screenshot sequences
3. Generates videos with fade transitions using ffmpeg
4. Outputs to `assets/videos/`

## Usage

### Embedding in README
```markdown
![Installation Walkthrough](assets/videos/installation-walkthrough.mp4)
```

### GitHub Video Embedding
```markdown
https://github.com/qualixar/superlocalmemory/assets/videos/installation-walkthrough.mp4
```

### Documentation Links
```markdown
[Watch Installation Guide](https://github.com/qualixar/superlocalmemory/blob/main/assets/videos/installation-walkthrough.mp4)
```

## Regeneration

To regenerate all videos:

```bash
cd ../screenshots
python3 generate-videos.py
```

This will overwrite existing videos with updated content from the latest screenshots.

## License

Copyright (c) 2026 Varun Pratap Bhardwaj
Licensed under GNU Affero General Public License v3.0 or later
