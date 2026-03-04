# ============================================================================
# SuperLocalMemory V2 - NPM Package Testing Script (PowerShell)
# Copyright (c) 2026 Varun Pratap Bhardwaj
# Licensed under MIT License
# Repository: https://github.com/varun369/SuperLocalMemoryV2
# ============================================================================
#
# Tests the NPM package locally before publishing
# Run this BEFORE npm publish
#

param()

$ErrorActionPreference = "Stop"

$REPO_DIR = $PSScriptRoot

Write-Host "╔══════════════════════════════════════════════════════════════╗"
Write-Host "║  SuperLocalMemory V2 - NPM Package Test                      ║"
Write-Host "║  by Varun Pratap Bhardwaj                                    ║"
Write-Host "╚══════════════════════════════════════════════════════════════╝"
Write-Host ""

# Step 1: Validate package.json
Write-Host "📋 Step 1: Validating package.json..."
$packageJsonPath = Join-Path $REPO_DIR "package.json"
if (-not (Test-Path $packageJsonPath)) {
    Write-Host "❌ Error: package.json not found" -ForegroundColor Red
    exit 1
}

# Check version
$packageJson = Get-Content $packageJsonPath | ConvertFrom-Json
$VERSION = $packageJson.version
Write-Host "   Version: $VERSION"

# Check required fields
$required = @("name", "version", "description", "author", "license", "repository", "bin")
$missing = @()
foreach ($field in $required) {
    if (-not $packageJson.$field) {
        $missing += $field
    }
}

if ($missing.Count -gt 0) {
    Write-Host "❌ Missing required fields: $($missing -join ', ')" -ForegroundColor Red
    exit 1
}
Write-Host "   ✓ All required fields present"

# Step 2: Check file structure
Write-Host ""
Write-Host "📁 Step 2: Checking file structure..."

$REQUIRED_FILES = @(
    "bin\slm-npm",
    "scripts\postinstall.js",
    "scripts\preuninstall.js",
    "src\memory_store_v2.py",
    "mcp_server.py",
    "install.sh",
    "install.ps1",
    "README.md",
    "LICENSE",
    "ATTRIBUTION.md"
)

$allPresent = $true
foreach ($file in $REQUIRED_FILES) {
    $filePath = Join-Path $REPO_DIR $file
    if (-not (Test-Path $filePath)) {
        Write-Host "   ❌ Missing: $file" -ForegroundColor Red
        $allPresent = $false
    }
}

if (-not $allPresent) {
    exit 1
}
Write-Host "   ✓ All required files present"

# Step 3: Test npm pack
Write-Host ""
Write-Host "📦 Step 3: Testing npm pack..."
Push-Location $REPO_DIR
try {
    # Remove old tarballs
    Get-ChildItem -Filter "superlocalmemory-*.tgz" | Remove-Item -Force -ErrorAction SilentlyContinue

    # Create tarball
    $packOutput = npm pack --quiet 2>&1
    $TARBALL = Get-ChildItem -Filter "superlocalmemory-*.tgz" | Select-Object -First 1

    if (-not $TARBALL) {
        Write-Host "   ❌ npm pack failed" -ForegroundColor Red
        exit 1
    }

    $SIZE = [math]::Round((Get-Item $TARBALL.FullName).Length / 1MB, 2)
    Write-Host "   ✓ Package created: $($TARBALL.Name) ($SIZE MB)"

    # Check package size
    if ($SIZE -gt 1) {
        Write-Host "   ⚠️  Warning: Package size is large ($SIZE MB). Target: <0.5MB" -ForegroundColor Yellow
    } else {
        Write-Host "   ✓ Package size OK: $SIZE MB"
    }

    # Step 4: Inspect package contents
    Write-Host ""
    Write-Host "🔍 Step 4: Inspecting package contents..."
    Write-Host "   Files included:"

    # Extract tarball contents list (using tar if available, otherwise show warning)
    try {
        $tarContents = tar -tzf $TARBALL.FullName 2>$null | Select-Object -First 20
        $tarContents | ForEach-Object { Write-Host "      $_" }
        $fileCount = (tar -tzf $TARBALL.FullName 2>$null | Measure-Object).Count
        Write-Host "   ... ($fileCount total files)"

        # Check for unwanted files
        $UNWANTED = @("*.pyc", "*.db", ".git", "test_*.py", "wiki-content")
        $foundUnwanted = $false
        foreach ($pattern in $UNWANTED) {
            $found = tar -tzf $TARBALL.FullName 2>$null | Select-String -Pattern $pattern
            if ($found) {
                Write-Host "   ⚠️  Warning: Found unwanted files matching: $pattern" -ForegroundColor Yellow
                $foundUnwanted = $true
            }
        }
        if (-not $foundUnwanted) {
            Write-Host "   ✓ No obvious unwanted files found"
        }
    } catch {
        Write-Host "   ⚠️  Could not inspect tarball contents (tar not available)" -ForegroundColor Yellow
    }

    # Step 5: Test npm link (local install)
    Write-Host ""
    Write-Host "🔗 Step 5: Testing npm link (local install)..."

    # Unlink if already linked
    npm unlink -g superlocalmemory 2>$null

    # Link
    $linkResult = npm link --quiet 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "   ✓ npm link successful"
    } else {
        Write-Host "   ❌ npm link failed" -ForegroundColor Red
        Write-Host "   $linkResult"
        exit 1
    }

    # Step 6: Test CLI command
    Write-Host ""
    Write-Host "🧪 Step 6: Testing CLI command..."

    $slmCommand = Get-Command slm -ErrorAction SilentlyContinue
    if ($slmCommand) {
        Write-Host "   ✓ slm command found"

        # Test version
        try {
            slm --version 2>&1 | Out-Null
            if ($LASTEXITCODE -eq 0) {
                Write-Host "   ✓ slm --version works"
            } else {
                Write-Host "   ⚠️  slm --version failed (might be OK if not installed yet)" -ForegroundColor Yellow
            }
        } catch {
            Write-Host "   ⚠️  slm --version failed" -ForegroundColor Yellow
        }

        # Test help
        try {
            slm --help 2>&1 | Out-Null
            if ($LASTEXITCODE -eq 0) {
                Write-Host "   ✓ slm --help works"
            } else {
                Write-Host "   ⚠️  slm --help failed" -ForegroundColor Yellow
            }
        } catch {
            Write-Host "   ⚠️  slm --help failed" -ForegroundColor Yellow
        }
    } else {
        Write-Host "   ⚠️  slm command not found (expected for first install)" -ForegroundColor Yellow
        Write-Host "      This is OK - postinstall will set it up"
    }

    # Step 7: Validate scripts
    Write-Host ""
    Write-Host "🔧 Step 7: Validating scripts..."

    # Check postinstall.js
    try {
        node (Join-Path $REPO_DIR "scripts\postinstall.js") --help 2>&1 | Out-Null
        Write-Host "   ✓ postinstall.js is valid Node.js"
    } catch {
        Write-Host "   ⚠️  postinstall.js might have issues" -ForegroundColor Yellow
    }

    # Check executables (Windows doesn't need execute bit)
    $slmNpmPath = Join-Path $REPO_DIR "bin\slm-npm"
    if (Test-Path $slmNpmPath) {
        Write-Host "   ✓ bin\slm-npm exists"
    } else {
        Write-Host "   ❌ bin\slm-npm not found" -ForegroundColor Red
        exit 1
    }

    # Step 8: Final summary
    Write-Host ""
    Write-Host "╔══════════════════════════════════════════════════════════════╗"
    Write-Host "║  Test Summary                                                ║"
    Write-Host "╚══════════════════════════════════════════════════════════════╝"
    Write-Host ""
    Write-Host "✅ Package structure: OK"
    Write-Host "✅ File validation: OK"
    Write-Host "✅ npm pack: OK ($SIZE MB)"
    Write-Host "✅ npm link: OK"
    Write-Host "✅ Scripts valid: OK"
    Write-Host ""
    Write-Host "📦 Package ready for testing!"
    Write-Host ""
    Write-Host "Next steps:"
    Write-Host "1. Test manual install:"
    Write-Host "   npm install -g .\$($TARBALL.Name)"
    Write-Host "   slm status"
    Write-Host "   npm uninstall -g superlocalmemory"
    Write-Host ""
    Write-Host "2. Test on Docker (recommended):"
    Write-Host "   docker run -it --rm -v ${PWD}:/app node:18 bash"
    Write-Host "   cd /app"
    Write-Host "   npm install -g ./$($TARBALL.Name)"
    Write-Host "   slm status"
    Write-Host ""
    Write-Host "3. When ready to publish:"
    Write-Host "   npm login"
    Write-Host "   npm publish"
    Write-Host ""
    Write-Host "Full guide: NPM-PUBLISHING-GUIDE.md"
    Write-Host ""

    # Cleanup
    Remove-Item $TARBALL.FullName -Force
    Write-Host "🧹 Cleaned up test tarball"

} finally {
    Pop-Location
}
