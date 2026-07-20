#!/bin/bash
# ============================================================================
# SuperLocalMemory V3.7 - Wiki Sync Script
# Automatically syncs wiki-content/ to GitHub Wiki repository
# Copyright (c) 2026 Varun Pratap Bhardwaj
# ============================================================================

set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
WIKI_CONTENT_DIR="${REPO_DIR}/wiki-content"
WIKI_REPO_DIR="/tmp/superlocalmemory.wiki"
WIKI_REPO_URL="https://github.com/qualixar/superlocalmemory.wiki.git"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  SuperLocalMemory V3.7 - Wiki Sync Tool                      ║"
echo "║  by Varun Pratap Bhardwaj                                    ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Check if wiki-content exists
if [ ! -d "${WIKI_CONTENT_DIR}" ]; then
    echo "✗ Error: wiki-content/ directory not found"
    exit 1
fi

# Clone or update wiki repo
if [ -d "${WIKI_REPO_DIR}" ]; then
    echo "📥 Updating existing wiki repository..."
    cd "${WIKI_REPO_DIR}"
    DEFAULT_BRANCH=$(git symbolic-ref refs/remotes/origin/HEAD 2>/dev/null | sed 's@^refs/remotes/origin/@@' || echo "master")
    git pull origin "${DEFAULT_BRANCH}"
else
    echo "📥 Cloning wiki repository..."
    git clone "${WIKI_REPO_URL}" "${WIKI_REPO_DIR}"
    cd "${WIKI_REPO_DIR}"
fi

# Sync wiki files
echo ""
echo "📋 Syncing wiki files..."
rsync -av --delete \
    --exclude='.git' \
    --exclude='DEPLOYMENT-CHECKLIST.md' \
    --exclude='WIKI-UPDATE-SUMMARY.md' \
    --exclude='NEW-PAGES-SUMMARY.md' \
    --exclude='README.md' \
    "${WIKI_CONTENT_DIR}/" "${WIKI_REPO_DIR}/"

# Remove old naming
rm -f "${WIKI_REPO_DIR}/4-Layer-Architecture.md"

# Check for changes
if [ -z "$(git status --porcelain)" ]; then
    echo "✓ No changes to sync"
    exit 0
fi

# Show what changed
echo ""
echo "📝 Changes detected:"
git status --short

# Commit and push
echo ""
echo "💾 Committing changes..."
git add .
git commit -m "Sync: Update wiki from main repo ($(date '+%Y-%m-%d %H:%M'))

Automated sync from wiki-content/ directory

Created by: Varun Pratap Bhardwaj"

echo ""
echo "📤 Pushing to GitHub Wiki..."
git push origin "${DEFAULT_BRANCH:-master}"

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  ✅ Wiki synced successfully!                                 ║"
echo "║  View at: https://github.com/qualixar/superlocalmemory/wiki║"
echo "╚══════════════════════════════════════════════════════════════╝"
