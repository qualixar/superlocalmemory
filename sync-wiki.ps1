# ============================================================================
# SuperLocalMemory V2 - Wiki Sync Script (PowerShell)
# Automatically syncs wiki-content/ to GitHub Wiki repository
# Copyright (c) 2026 Varun Pratap Bhardwaj
# ============================================================================

param()

$ErrorActionPreference = "Stop"

# Configuration
$REPO_DIR = $PSScriptRoot
$WIKI_CONTENT_DIR = Join-Path $REPO_DIR "wiki-content"
$WIKI_REPO_DIR = Join-Path $env:TEMP "SuperLocalMemoryV2.wiki"
$WIKI_REPO_URL = "https://github.com/varun369/SuperLocalMemoryV2.wiki.git"

Write-Host "╔══════════════════════════════════════════════════════════════╗"
Write-Host "║  SuperLocalMemory V2 - Wiki Sync Tool                        ║"
Write-Host "║  by Varun Pratap Bhardwaj                                    ║"
Write-Host "╚══════════════════════════════════════════════════════════════╝"
Write-Host ""

# Check if wiki-content exists
if (-not (Test-Path $WIKI_CONTENT_DIR)) {
    Write-Host "✗ Error: wiki-content/ directory not found" -ForegroundColor Red
    exit 1
}

# Clone or update wiki repo
if (Test-Path $WIKI_REPO_DIR) {
    Write-Host "📥 Updating existing wiki repository..."
    Push-Location $WIKI_REPO_DIR
    try {
        $DEFAULT_BRANCH = git symbolic-ref refs/remotes/origin/HEAD 2>$null
        if ($DEFAULT_BRANCH) {
            $DEFAULT_BRANCH = $DEFAULT_BRANCH -replace '^refs/remotes/origin/', ''
        } else {
            $DEFAULT_BRANCH = "master"
        }
        git pull origin $DEFAULT_BRANCH
    } finally {
        Pop-Location
    }
} else {
    Write-Host "📥 Cloning wiki repository..."
    git clone $WIKI_REPO_URL $WIKI_REPO_DIR
    Set-Location $WIKI_REPO_DIR
    $DEFAULT_BRANCH = git symbolic-ref refs/remotes/origin/HEAD 2>$null
    if ($DEFAULT_BRANCH) {
        $DEFAULT_BRANCH = $DEFAULT_BRANCH -replace '^refs/remotes/origin/', ''
    } else {
        $DEFAULT_BRANCH = "master"
    }
}

# Sync wiki files (PowerShell equivalent of rsync)
Write-Host ""
Write-Host "📋 Syncing wiki files..."

Push-Location $WIKI_REPO_DIR
try {
    # Remove old files except .git and excluded files
    $excludeFiles = @("DEPLOYMENT-CHECKLIST.md", "WIKI-UPDATE-SUMMARY.md", "NEW-PAGES-SUMMARY.md", "README.md")
    Get-ChildItem -Path . -Recurse -File | Where-Object {
        $_.Name -notin $excludeFiles -and $_.FullName -notlike "*\.git\*"
    } | Remove-Item -Force

    # Copy new files from wiki-content
    Get-ChildItem -Path $WIKI_CONTENT_DIR -Recurse -File | Where-Object {
        $_.Name -notin $excludeFiles
    } | ForEach-Object {
        $relativePath = $_.FullName.Substring($WIKI_CONTENT_DIR.Length + 1)
        $targetPath = Join-Path $WIKI_REPO_DIR $relativePath
        $targetDir = Split-Path $targetPath -Parent
        if (-not (Test-Path $targetDir)) {
            New-Item -ItemType Directory -Path $targetDir -Force | Out-Null
        }
        Copy-Item $_.FullName -Destination $targetPath -Force
    }

    # Remove old naming
    $oldFile = Join-Path $WIKI_REPO_DIR "4-Layer-Architecture.md"
    if (Test-Path $oldFile) {
        Remove-Item $oldFile -Force
    }

    # Check for changes
    $changes = git status --porcelain
    if (-not $changes) {
        Write-Host "✓ No changes to sync" -ForegroundColor Green
        exit 0
    }

    # Show what changed
    Write-Host ""
    Write-Host "📝 Changes detected:"
    git status --short

    # Commit and push
    Write-Host ""
    Write-Host "💾 Committing changes..."
    git add .

    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm"
    $commitMessage = @"
Sync: Update wiki from main repo ($timestamp)

Automated sync from wiki-content/ directory

Created by: Varun Pratap Bhardwaj
"@

    git commit -m $commitMessage

    Write-Host ""
    Write-Host "📤 Pushing to GitHub Wiki..."
    git push origin $DEFAULT_BRANCH

    Write-Host ""
    Write-Host "╔══════════════════════════════════════════════════════════════╗"
    Write-Host "║  ✅ Wiki synced successfully!                                 ║"
    Write-Host "║  View at: https://github.com/varun369/SuperLocalMemoryV2/wiki║"
    Write-Host "╚══════════════════════════════════════════════════════════════╝"

} finally {
    Pop-Location
}
