# ============================================================================
# SuperLocalMemory - Installation Verification Script (PowerShell)
# Copyright (c) 2026 Varun Pratap Bhardwaj
# Licensed under MIT License
# ============================================================================

$ErrorActionPreference = "Continue"

$INSTALL_DIR = Join-Path $env:USERPROFILE ".claude-memory"

# Print banner
Write-Host ""
Write-Host "=================================================================="
Write-Host "  SuperLocalMemory - Installation Verification            "
Write-Host "  by Varun Pratap Bhardwaj                                       "
Write-Host "=================================================================="
Write-Host ""

# Track status
$CORE_OK = $true
$SEARCH_OK = $false
$UI_OK = $false
$ERRORS = @()

# ============================================================================
# CORE INSTALLATION CHECK
# ============================================================================

Write-Host "================================================================"
Write-Host "1. Core Installation"
Write-Host "================================================================"
Write-Host ""

# Check Python version
Write-Host -NoNewline "Python 3.8+               "
try {
    $pythonVersion = & python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>&1
    $pythonMajor = & python -c "import sys; print(sys.version_info.major)" 2>&1
    $pythonMinor = & python -c "import sys; print(sys.version_info.minor)" 2>&1
    
    if ([int]$pythonMajor -lt 3 -or ([int]$pythonMajor -eq 3 -and [int]$pythonMinor -lt 8)) {
        Write-Host "FAIL (found $pythonVersion)" -ForegroundColor Red
        $CORE_OK = $false
        $ERRORS += "Python 3.8+ required"
    } else {
        Write-Host "OK (v$pythonVersion)" -ForegroundColor Green
    }
} catch {
    Write-Host "FAIL (Python not found)" -ForegroundColor Red
    $CORE_OK = $false
    $ERRORS += "Python not installed"
}

# Check installation directory
Write-Host -NoNewline "Installation directory    "
if (Test-Path $INSTALL_DIR) {
    Write-Host "OK ($INSTALL_DIR)" -ForegroundColor Green
} else {
    Write-Host "FAIL (not found)" -ForegroundColor Red
    $CORE_OK = $false
    $ERRORS += "Installation directory missing"
}

# Check core files
Write-Host -NoNewline "Core scripts              "
$coreFiles = @("memory_store_v2.py", "graph_engine.py", "pattern_learner.py")
$missingFiles = @()
foreach ($file in $coreFiles) {
    $filePath = Join-Path $INSTALL_DIR $file
    if (-not (Test-Path $filePath)) {
        $missingFiles += $file
    }
}

if ($missingFiles.Count -eq 0) {
    Write-Host "OK" -ForegroundColor Green
} else {
    Write-Host "FAIL (missing: $($missingFiles -join ', '))" -ForegroundColor Red
    $CORE_OK = $false
    $ERRORS += "Core scripts missing"
}

# Check CLI wrappers
Write-Host -NoNewline "CLI wrappers              "
$binDir = Join-Path $INSTALL_DIR "bin"
if (Test-Path $binDir) {
    Write-Host "OK" -ForegroundColor Green
} else {
    Write-Host "FAIL" -ForegroundColor Red
    $CORE_OK = $false
    $ERRORS += "CLI wrappers missing"
}

# Check PATH configuration
Write-Host -NoNewline "PATH configuration        "
$userPath = [Environment]::GetEnvironmentVariable("PATH", "User")
if ($userPath -like "*$binDir*") {
    Write-Host "OK (commands globally available)" -ForegroundColor Green
} else {
    Write-Host "WARNING (not in PATH)" -ForegroundColor Yellow
    $ERRORS += "PATH not configured - add manually or re-run installer"
}

# Check database
Write-Host -NoNewline "Database                  "
$dbPath = Join-Path $INSTALL_DIR "memory.db"
if (Test-Path $dbPath) {
    $dbSize = [math]::Round((Get-Item $dbPath).Length / 1KB, 2)
    Write-Host "OK ($dbSize KB)" -ForegroundColor Green
} else {
    Write-Host "NOT CREATED (will be created on first use)" -ForegroundColor Yellow
}

# Check config
Write-Host -NoNewline "Configuration             "
$configPath = Join-Path $INSTALL_DIR "config.json"
if (Test-Path $configPath) {
    Write-Host "OK" -ForegroundColor Green
} else {
    Write-Host "WARNING (using defaults)" -ForegroundColor Yellow
}

Write-Host ""

# ============================================================================
# OPTIONAL FEATURES CHECK
# ============================================================================

Write-Host "================================================================"
Write-Host "2. Optional Features"
Write-Host "================================================================"
Write-Host ""

# Advanced Search
Write-Host -NoNewline "Advanced Search           "
try {
    & python -c "import sentence_transformers; import hnswlib" 2>$null
    Write-Host "ENABLED" -ForegroundColor Green
    $SEARCH_OK = $true
} catch {
    Write-Host "DISABLED" -ForegroundColor Yellow
    Write-Host "  Install: pip install -r requirements-search.txt"
}

# Web Dashboard
Write-Host -NoNewline "Web Dashboard             "
try {
    & python -c "import fastapi; import uvicorn" 2>$null
    Write-Host "ENABLED" -ForegroundColor Green
    $UI_OK = $true
    $apiServerPath = Join-Path $INSTALL_DIR "api_server.py"
    if (Test-Path $apiServerPath) {
        Write-Host "  Start: python $apiServerPath"
        Write-Host "  URL:   http://127.0.0.1:8000"
    }
} catch {
    Write-Host "DISABLED" -ForegroundColor Yellow
    Write-Host "  Install: pip install -r requirements-ui.txt"
}

Write-Host ""

# ============================================================================
# PERFORMANCE QUICK TEST
# ============================================================================

if ($CORE_OK) {
    Write-Host "================================================================"
    Write-Host "3. Performance Quick Test"
    Write-Host "================================================================"
    Write-Host ""

    # Test memory store initialization
    Write-Host -NoNewline "Memory store init         "
    try {
        & python -c "import sys; sys.path.insert(0, '$INSTALL_DIR'); from memory_store_v2 import MemoryStoreV2; store = MemoryStoreV2()" 2>$null
        Write-Host "OK" -ForegroundColor Green
    } catch {
        Write-Host "FAIL" -ForegroundColor Red
        $ERRORS += "Memory store initialization failed"
    }

    # Test database query
    Write-Host -NoNewline "Database query            "
    $startTime = Get-Date
    try {
        & python -c "import sys; sys.path.insert(0, '$INSTALL_DIR'); from memory_store_v2 import MemoryStoreV2; store = MemoryStoreV2(); list(store.list_all(limit=1))" 2>$null
        $endTime = Get-Date
        $duration = [math]::Round(($endTime - $startTime).TotalMilliseconds, 0)
        Write-Host "OK ($duration ms)" -ForegroundColor Green
    } catch {
        Write-Host "FAIL" -ForegroundColor Red
        $ERRORS += "Database query failed"
    }

    Write-Host ""
}

# ============================================================================
# SUMMARY
# ============================================================================

Write-Host "================================================================"
Write-Host "4. Summary"
Write-Host "================================================================"
Write-Host ""

if ($CORE_OK) {
    Write-Host "Core Installation:      WORKING" -ForegroundColor Green
} else {
    Write-Host "Core Installation:      FAILED" -ForegroundColor Red
}

if ($SEARCH_OK) {
    Write-Host "Advanced Search:        ENABLED" -ForegroundColor Green
} else {
    Write-Host "Advanced Search:        DISABLED (optional)" -ForegroundColor Yellow
}

if ($UI_OK) {
    Write-Host "Web Dashboard:          ENABLED" -ForegroundColor Green
} else {
    Write-Host "Web Dashboard:          DISABLED (optional)" -ForegroundColor Yellow
}

Write-Host ""

# Feature Status
Write-Host "Feature Status:"
if ($CORE_OK) {
    Write-Host "  - Basic CLI commands:    Available" -ForegroundColor Green
    Write-Host "  - MCP Server:            Available" -ForegroundColor Green
    Write-Host "  - Skills:                Available" -ForegroundColor Green
} else {
    Write-Host "  - Basic CLI commands:    Not available" -ForegroundColor Red
    Write-Host "  - MCP Server:            Not available" -ForegroundColor Red
    Write-Host "  - Skills:                Not available" -ForegroundColor Red
}

if ($SEARCH_OK) {
    Write-Host "  - Semantic Search:       Enabled" -ForegroundColor Green
} else {
    Write-Host "  - Semantic Search:       Disabled" -ForegroundColor Yellow
}

if ($UI_OK) {
    Write-Host "  - Web Interface:         Enabled" -ForegroundColor Green
} else {
    Write-Host "  - Web Interface:         Disabled" -ForegroundColor Yellow
}

Write-Host ""

# Errors
if ($ERRORS.Count -gt 0) {
    Write-Host "Errors detected:" -ForegroundColor Yellow
    foreach ($error in $ERRORS) {
        Write-Host "  - $error"
    }
    Write-Host ""
}

# Next Steps
Write-Host "Next Steps:"
Write-Host ""

if ($CORE_OK) {
    Write-Host "  Try it now:"
    Write-Host "    python $INSTALL_DIR\memory_store_v2.py"
    Write-Host ""

    if (-not $SEARCH_OK -and -not $UI_OK) {
        Write-Host "  Install optional features:"
        Write-Host "    pip install -r requirements-search.txt  # Advanced search"
        Write-Host "    pip install -r requirements-ui.txt      # Web dashboard"
        Write-Host "    pip install -r requirements-full.txt    # Everything"
        Write-Host ""
    }
} else {
    Write-Host "  Fix installation issues:"
    Write-Host "    Run: .\install.ps1"
    Write-Host ""
}

# Exit code
if ($CORE_OK) {
    Write-Host "Installation verification PASSED" -ForegroundColor Green
    Write-Host ""
    exit 0
} else {
    Write-Host "Installation verification FAILED" -ForegroundColor Red
    Write-Host ""
    exit 1
}
