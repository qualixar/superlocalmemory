#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 SuperLocalMemory (superlocalmemory.com)
"""SuperLocalMemory V2 - Thumbnail Generator

Generates optimized thumbnail versions of all screenshots.
- Size: 320×180px (16:9 ratio)
- Format: PNG (for wiki/docs) and WebP (for website)
- Quality: High enough to recognize content
- File size: < 50KB per thumbnail
"""
import os
import json
from pathlib import Path
from PIL import Image, ImageFilter, ImageOps
from datetime import datetime

# Configuration
SCREENSHOT_DIR = Path(__file__).parent.parent / "assets" / "screenshots"
THUMBNAIL_DIR = Path(__file__).parent.parent / "assets" / "thumbnails"
THUMBNAIL_SIZE = (320, 180)  # 16:9 ratio
QUALITY_PNG = 95
QUALITY_WEBP = 85
MAX_FILESIZE = 50 * 1024  # 50KB

# Category mapping based on filename patterns
CATEGORY_MAP = {
    "overview": "dashboard",
    "timeline": "timeline",
    "agents": "agents",
    "patterns": "patterns",
    "clusters": "clusters",
    "memories": "memories",
    "graph": "graph",
    "filtered": "search",
    "live-events": "events",
}

def get_category(filename):
    """Extract category from filename."""
    for pattern, category in CATEGORY_MAP.items():
        if pattern in filename.lower():
            return category
    return "general"

def get_title(filename):
    """Generate human-readable title from filename."""
    # Remove extensions and convert dashes/underscores to spaces
    name = Path(filename).stem
    # Remove 'dashboard-' prefix if present
    if name.startswith("dashboard-"):
        name = name[9:]
    # Remove '-dark' suffix
    name = name.replace("-dark", "")
    # Convert to title case
    return " ".join(word.capitalize() for word in name.split("-"))

def get_description(filename, category):
    """Generate description based on filename and category."""
    descriptions = {
        "overview": "Main dashboard with memory statistics and knowledge graph overview",
        "timeline": "Chronological timeline of all stored memories and events",
        "agents": "Agent connections and activity tracking",
        "patterns": "Learned coding patterns and user preferences",
        "clusters": "Knowledge graph clusters and relationships",
        "memories": "Detailed memory list with search and filtering",
        "graph": "Interactive knowledge graph visualization",
        "search": "Advanced memory search and filtering interface",
        "events": "Real-time live event stream from memory operations",
    }
    is_dark = "-dark" in filename.lower()
    base_desc = descriptions.get(category, "Dashboard interface")
    if is_dark:
        base_desc += " (dark mode)"
    return base_desc

def resize_and_crop(image, target_size):
    """
    Resize image to target size while maintaining aspect ratio.
    Crops if necessary to achieve exact dimensions.
    """
    img_ratio = image.width / image.height
    target_ratio = target_size[0] / target_size[1]

    if img_ratio > target_ratio:
        # Image is wider, crop width
        new_height = target_size[1]
        new_width = int(new_height * img_ratio)
        resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        left = (resized.width - target_size[0]) // 2
        return resized.crop((left, 0, left + target_size[0], target_size[1]))
    else:
        # Image is taller, crop height
        new_width = target_size[0]
        new_height = int(new_width / img_ratio)
        resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        top = (resized.height - target_size[1]) // 2
        return resized.crop((0, top, target_size[0], top + target_size[1]))

def apply_sharpening(image):
    """Apply subtle sharpening to enhance detail."""
    # Use UNSHARP_MASK equivalent with subtle settings
    return image.filter(ImageFilter.UnsharpMask(radius=1, percent=100, threshold=3))

def generate_thumbnail(source_path, dest_dir, metadata):
    """Generate PNG and WebP thumbnails for a single source image."""
    try:
        # Open image
        with Image.open(source_path) as img:
            # Convert RGBA to RGB if necessary (for PNG/WebP)
            if img.mode == "RGBA":
                background = Image.new("RGB", img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[3])
                img = background
            elif img.mode != "RGB":
                img = img.convert("RGB")

            # Resize and crop
            thumbnail = resize_and_crop(img, THUMBNAIL_SIZE)

            # Apply sharpening
            thumbnail = apply_sharpening(thumbnail)

            filename = source_path.stem

            # Save PNG version
            png_path = dest_dir / f"{filename}-thumb.png"
            thumbnail.save(png_path, "PNG", quality=QUALITY_PNG, optimize=True)
            png_size = png_path.stat().st_size

            # Save WebP version
            webp_path = dest_dir / f"{filename}-thumb.webp"
            thumbnail.save(webp_path, "WEBP", quality=QUALITY_WEBP, method=6)
            webp_size = webp_path.stat().st_size

            # Check file sizes
            if png_size > MAX_FILESIZE:
                print(f"⚠️  PNG {filename}: {png_size/1024:.1f}KB (exceeds limit)")
            if webp_size > MAX_FILESIZE:
                print(f"⚠️  WebP {filename}: {webp_size/1024:.1f}KB (exceeds limit)")

            print(f"✓ {filename}")
            print(f"  PNG: {png_size/1024:.1f}KB | WebP: {webp_size/1024:.1f}KB")

            # Store metadata
            category = get_category(filename)
            metadata[filename] = {
                "title": get_title(filename),
                "description": get_description(source_path.name, category),
                "category": category,
                "full_image": f"../screenshots/dashboard/{source_path.name}",
                "thumbnail_png": f"{filename}-thumb.png",
                "thumbnail_webp": f"{filename}-thumb.webp",
                "created": datetime.now().isoformat(),
                "original_size": f"{img.width}×{img.height}",
                "thumbnail_size": f"{THUMBNAIL_SIZE[0]}×{THUMBNAIL_SIZE[1]}",
                "png_size_kb": round(png_size / 1024, 2),
                "webp_size_kb": round(webp_size / 1024, 2),
            }
            return True
    except Exception as e:
        print(f"✗ {source_path.name}: {str(e)}")
        return False

def main():
    """Generate all thumbnails."""
    # Ensure thumbnail directory exists
    THUMBNAIL_DIR.mkdir(parents=True, exist_ok=True)

    # Verify screenshot directory exists
    if not SCREENSHOT_DIR.exists():
        print(f"Error: Screenshot directory not found: {SCREENSHOT_DIR}")
        return 1

    # Find all images in screenshot directory
    image_extensions = {".png", ".jpg", ".jpeg", ".webp"}
    sources = sorted([
        f for f in SCREENSHOT_DIR.glob("**/*")
        if f.is_file() and f.suffix.lower() in image_extensions and not f.name.startswith(".")
    ])

    if not sources:
        print(f"No images found in {SCREENSHOT_DIR}")
        return 1

    print(f"Found {len(sources)} images in {SCREENSHOT_DIR}")
    print(f"Generating thumbnails to {THUMBNAIL_DIR}\n")

    metadata = {}
    successful = 0
    failed = 0

    for source in sources:
        if generate_thumbnail(source, THUMBNAIL_DIR, metadata):
            successful += 1
        else:
            failed += 1

    # Save metadata index
    index_path = THUMBNAIL_DIR / "index.json"
    with open(index_path, "w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
    print(f"\n✓ Saved metadata index to {index_path}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Total processed: {len(sources)}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  PNG thumbnails: {len(list(THUMBNAIL_DIR.glob('*-thumb.png')))}")
    print(f"  WebP thumbnails: {len(list(THUMBNAIL_DIR.glob('*-thumb.webp')))}")
    print(f"  Total size: {sum(f.stat().st_size for f in THUMBNAIL_DIR.glob('*-thumb.*')) / 1024:.1f}KB")

    return 0 if failed == 0 else 1

if __name__ == "__main__":
    exit(main())
