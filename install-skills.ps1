# SuperLocalMemory V2 - Universal Skills Installer (PowerShell)
# Installs skills for Claude Code, Codex, Gemini CLI, Antigravity, and Windsurf

param(
    [switch]$Auto
)

$ErrorActionPreference = "Stop"

# Configuration
$REPO_DIR = $PSScriptRoot
$SKILLS_SOURCE = Join-Path $REPO_DIR "skills"

# Tool definitions
$TOOL_IDS = @("claude_code", "codex", "gemini_cli", "antigravity", "windsurf", "cursor", "vscode_copilot")
$TOOL_NAMES = @("Claude Code", "Codex", "Gemini CLI", "Antigravity", "Windsurf", "Cursor", "VS Code/Copilot")
$TOOL_DIRS = @(
    (Join-Path $env:USERPROFILE ".claude\skills"),
    (Join-Path $env:USERPROFILE ".codex\skills"),
    (Join-Path $env:USERPROFILE ".gemini\skills"),
    (Join-Path $env:USERPROFILE ".gemini\antigravity\skills"),
    (Join-Path $env:USERPROFILE ".windsurf\skills"),
    (Join-Path $env:USERPROFILE ".cursor\skills"),
    (Join-Path $env:USERPROFILE ".copilot\skills")
)

# Helper functions
function Get-ToolName {
    param([string]$ToolId)
    $index = $TOOL_IDS.IndexOf($ToolId)
    if ($index -ge 0) { return $TOOL_NAMES[$index] }
    return $ToolId
}

function Get-ToolDir {
    param([string]$ToolId)
    $index = $TOOL_IDS.IndexOf($ToolId)
    if ($index -ge 0) { return $TOOL_DIRS[$index] }
    return ""
}

# Skills to install
$SKILLS = @(
    "slm-remember",
    "slm-recall",
    "slm-status",
    "slm-list-recent",
    "slm-build-graph",
    "slm-switch-profile"
)

Write-Host "=========================================="
Write-Host "SuperLocalMemory V2 - Universal Skills Installer"
Write-Host "=========================================="
Write-Host ""

# Check if SuperLocalMemory V2 is installed
$memoryDir = Join-Path $env:USERPROFILE ".claude-memory"
if (-not (Test-Path $memoryDir)) {
    Write-Host "Warning: SuperLocalMemory V2 not found at $memoryDir" -ForegroundColor Yellow
    Write-Host "Skills require SuperLocalMemory V2 to be installed first."
    Write-Host ""
    if (-not $Auto) {
        $response = Read-Host "Do you want to continue anyway? (y/n)"
        if ($response -ne "y" -and $response -ne "Y") {
            Write-Host "Installation cancelled."
            Write-Host "Please install SuperLocalMemory V2 first: .\install.ps1"
            exit 1
        }
    } else {
        Write-Host "Auto mode: continuing anyway..."
    }
}

# Check if skills source directory exists
if (-not (Test-Path $SKILLS_SOURCE)) {
    Write-Host "Error: Skills directory not found: $SKILLS_SOURCE" -ForegroundColor Red
    Write-Host "Please run this script from the SuperLocalMemoryV2-repo directory."
    exit 1
}

# Verify all skill directories exist
Write-Host "Verifying skill files..."
$MISSING_SKILLS = 0
foreach ($skill in $SKILLS) {
    $skillDir = Join-Path $SKILLS_SOURCE $skill
    $skillFile = Join-Path $skillDir "SKILL.md"
    if (-not (Test-Path $skillDir)) {
        Write-Host "✗ Missing: $skill/" -ForegroundColor Red
        $MISSING_SKILLS++
    } elseif (-not (Test-Path $skillFile)) {
        Write-Host "✗ Missing: $skill/SKILL.md" -ForegroundColor Red
        $MISSING_SKILLS++
    } else {
        Write-Host "✓ Found: $skill/SKILL.md" -ForegroundColor Green
    }
}

if ($MISSING_SKILLS -gt 0) {
    Write-Host "Error: $MISSING_SKILLS skill(s) missing. Cannot proceed." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Found $($SKILLS.Count) skills to install"
Write-Host ""

# Detect available tools
Write-Host "=========================================="
Write-Host "Detecting AI Tools"
Write-Host "=========================================="
Write-Host ""

$DETECTED_TOOLS = @()

for ($i = 0; $i -lt $TOOL_IDS.Count; $i++) {
    $toolId = $TOOL_IDS[$i]
    $toolName = $TOOL_NAMES[$i]
    $toolDir = $TOOL_DIRS[$i]
    $parentDir = Split-Path $toolDir -Parent

    if (Test-Path $parentDir) {
        Write-Host "✓ $toolName detected: $toolDir" -ForegroundColor Green
        $DETECTED_TOOLS += $toolId
    } else {
        Write-Host "○ $toolName not found: $parentDir" -ForegroundColor Blue
    }
}

Write-Host ""

if ($DETECTED_TOOLS.Count -eq 0) {
    if ($Auto) {
        Write-Host "Warning: No supported AI tools detected. Skipping skills installation." -ForegroundColor Yellow
        Write-Host ""
        exit 0
    }
    Write-Host "Warning: No supported AI tools detected." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Supported tools:"
    for ($i = 0; $i -lt $TOOL_IDS.Count; $i++) {
        Write-Host "  - $($TOOL_NAMES[$i]): $($TOOL_DIRS[$i])"
    }
    Write-Host ""
    $response = Read-Host "Do you want to continue and create directories anyway? (y/n)"
    if ($response -ne "y" -and $response -ne "Y") {
        Write-Host "Installation cancelled."
        exit 1
    }
    # Add all tools to install list
    $DETECTED_TOOLS = $TOOL_IDS
} else {
    Write-Host "Will install skills for $($DETECTED_TOOLS.Count) tool(s)"
    Write-Host ""
}

# Ask user for installation method
if ($Auto) {
    $METHOD = "copy"  # Windows: copy is more reliable than symlink
    Write-Host "Auto mode: Using copy method..."
} else {
    Write-Host "Installation Methods:"
    Write-Host "  1. Copy - Stable, requires manual updates (recommended for Windows)"
    Write-Host "  2. Symlink - Changes in repo reflect immediately (requires admin on Windows)"
    Write-Host ""
    $choice = Read-Host "Choose installation method (1 or 2)"
    Write-Host ""

    if ($choice -eq "1") {
        $METHOD = "copy"
        Write-Host "Installing via copy..."
    } elseif ($choice -eq "2") {
        $METHOD = "symlink"
        Write-Host "Installing via symlink..."
    } else {
        Write-Host "Invalid choice. Please enter 1 or 2." -ForegroundColor Red
        exit 1
    }
}

Write-Host ""

# Install skills for each detected tool
$TOTAL_INSTALLED = 0
$TOTAL_FAILED = 0
$TOTAL_SKIPPED = 0

foreach ($toolId in $DETECTED_TOOLS) {
    $toolName = Get-ToolName $toolId
    Write-Host "=========================================="
    Write-Host "Installing for: $toolName"
    Write-Host "=========================================="

    $skillsDir = Get-ToolDir $toolId

    # Create skills directory if it doesn't exist
    if (-not (Test-Path $skillsDir)) {
        Write-Host "Creating directory: $skillsDir"
        try {
            New-Item -ItemType Directory -Path $skillsDir -Force | Out-Null
            Write-Host "✓ Directory created" -ForegroundColor Green
        } catch {
            Write-Host "✗ Failed to create directory" -ForegroundColor Red
            Write-Host ""
            $TOTAL_SKIPPED += $SKILLS.Count
            continue
        }
    } else {
        Write-Host "Directory exists: $skillsDir"
    }

    Write-Host ""

    # Install each skill
    foreach ($skill in $SKILLS) {
        $sourceFile = Join-Path (Join-Path $SKILLS_SOURCE $skill) "SKILL.md"
        $targetFile = Join-Path $skillsDir "$skill.md"

        # Remove existing file or symlink
        if (Test-Path $targetFile) {
            Remove-Item $targetFile -Force
        }

        try {
            if ($METHOD -eq "symlink") {
                # Create symlink (requires admin on Windows)
                New-Item -ItemType SymbolicLink -Path $targetFile -Target $sourceFile -Force | Out-Null
                Write-Host "✓ Symlinked: $skill.md" -ForegroundColor Green
                $TOTAL_INSTALLED++
            } else {
                # Copy file
                Copy-Item $sourceFile -Destination $targetFile -Force
                Write-Host "✓ Copied: $skill.md" -ForegroundColor Green
                $TOTAL_INSTALLED++
            }
        } catch {
            Write-Host "✗ Failed to install: $skill.md - $($_.Exception.Message)" -ForegroundColor Red
            $TOTAL_FAILED++
        }
    }

    Write-Host ""
}

# Installation Summary
Write-Host "=========================================="
Write-Host "Installation Summary"
Write-Host "=========================================="
Write-Host ""
Write-Host "Tools configured: $($DETECTED_TOOLS.Count)" -ForegroundColor Blue
Write-Host "Skills installed: $TOTAL_INSTALLED" -ForegroundColor Green
if ($TOTAL_FAILED -gt 0) {
    Write-Host "Failed: $TOTAL_FAILED" -ForegroundColor Red
}
if ($TOTAL_SKIPPED -gt 0) {
    Write-Host "Skipped: $TOTAL_SKIPPED" -ForegroundColor Yellow
}
Write-Host "Installation method: $METHOD"
Write-Host ""

# List installation locations
Write-Host "Skills installed to:"
foreach ($toolId in $DETECTED_TOOLS) {
    $toolName = Get-ToolName $toolId
    $toolDir = Get-ToolDir $toolId
    if ((Test-Path $toolDir) -and ((Get-ChildItem -Path $toolDir -Filter "*.md" -ErrorAction SilentlyContinue).Count -gt 0)) {
        $count = (Get-ChildItem -Path $toolDir -Filter "*.md").Count
        Write-Host "  ✓ $toolName`: $toolDir ($count skills)" -ForegroundColor Green
    }
}

Write-Host ""

# Verify installation
$EXPECTED_TOTAL = $DETECTED_TOOLS.Count * $SKILLS.Count
if ($TOTAL_INSTALLED -eq $EXPECTED_TOTAL) {
    Write-Host "✓ All skills installed successfully!" -ForegroundColor Green
} else {
    Write-Host "⚠ Some skills failed to install." -ForegroundColor Yellow
    Write-Host "Expected: $EXPECTED_TOTAL, Installed: $TOTAL_INSTALLED"
}

Write-Host ""

# List available skills
Write-Host "=========================================="
Write-Host "Available Skills"
Write-Host "=========================================="
Write-Host ""
foreach ($skill in $SKILLS) {
    Write-Host "  • $skill"
}
Write-Host ""

# Next Steps
Write-Host "=========================================="
Write-Host "Next Steps"
Write-Host "=========================================="
Write-Host ""
Write-Host "1. Restart your AI tool to load the new skills" -ForegroundColor Yellow
Write-Host ""
Write-Host "2. Verify skills are loaded:"
foreach ($toolId in $DETECTED_TOOLS) {
    $toolName = Get-ToolName $toolId
    Write-Host "   $toolName (check available commands/skills)" -ForegroundColor Green
}
Write-Host ""
Write-Host "3. Try your first skill:"
Write-Host "   /slm-status or /slm-remember `"test`"" -ForegroundColor Green
Write-Host ""
Write-Host "4. Read skill documentation:"
Write-Host "   Each skill has detailed docs in: $SKILLS_SOURCE\<skill-name>\SKILL.md"
Write-Host ""

if ($METHOD -eq "copy") {
    Write-Host "Note: Using copies. To update skills:"
    Write-Host "  cd $REPO_DIR"
    Write-Host "  git pull"
    Write-Host "  .\install-skills.ps1"
    Write-Host ""
}

Write-Host "=========================================="
Write-Host ""
Write-Host "✓ Universal skills installation complete!" -ForegroundColor Green
Write-Host ""
Write-Host "Skills installed for:"
foreach ($toolId in $DETECTED_TOOLS) {
    Write-Host "  • $(Get-ToolName $toolId)"
}
Write-Host ""
Write-Host "Remember: Skills are OPTIONAL convenience wrappers."
Write-Host "SuperLocalMemory V2 works standalone via terminal commands."
Write-Host ""
