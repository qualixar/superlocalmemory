# ============================================================================
# SuperLocalMemory V2.7 — Quick Verification Script (PowerShell)
# Copyright (c) 2026 Varun Pratap Bhardwaj
# Licensed under MIT License
# Repository: https://github.com/varun369/SuperLocalMemoryV2
#
# Run this after installation to verify everything works:
#   .\scripts\verify-v27.ps1
# ============================================================================

param()

$ErrorActionPreference = "Stop"

$INSTALL_DIR = Join-Path $env:USERPROFILE ".claude-memory"
$PASS = 0
$WARN = 0
$FAIL = 0

Write-Host ""
Write-Host "SuperLocalMemory v2.7 Verification"
Write-Host "==================================="
Write-Host ""

# ── Check 1: Installation directory ──────────────────────────────────────────
if (Test-Path $INSTALL_DIR) {
    Write-Host "[PASS] Installation directory exists: $INSTALL_DIR" -ForegroundColor Green
    $PASS++
} else {
    Write-Host "[FAIL] Installation directory missing. Run install.ps1 first." -ForegroundColor Red
    $FAIL++
    Write-Host ""
    Write-Host "==================================="
    Write-Host "Result: FAIL — SuperLocalMemory is not installed."
    Write-Host "Run:  .\install.ps1"
    exit 1
}

# ── Check 2: Core modules ────────────────────────────────────────────────────
Write-Host ""
Write-Host "Core Modules:"
$coreModules = @("memory_store_v2.py", "graph_engine.py", "pattern_learner.py", "mcp_server.py", "tree_manager.py")
foreach ($mod in $coreModules) {
    $modPath = Join-Path $INSTALL_DIR $mod
    if (Test-Path $modPath) {
        Write-Host "  [PASS] $mod" -ForegroundColor Green
        $PASS++
    } else {
        Write-Host "  [FAIL] Missing: $mod" -ForegroundColor Red
        $FAIL++
    }
}

# ── Check 3: v2.5 modules ────────────────────────────────────────────────────
Write-Host ""
Write-Host "Event System (v2.5):"
$v25Modules = @("event_bus.py", "subscription_manager.py", "webhook_dispatcher.py", "agent_registry.py", "provenance_tracker.py", "trust_scorer.py", "db_connection_manager.py")
foreach ($mod in $v25Modules) {
    $modPath = Join-Path $INSTALL_DIR $mod
    if (Test-Path $modPath) {
        Write-Host "  [PASS] $mod" -ForegroundColor Green
        $PASS++
    } else {
        Write-Host "  [WARN] Missing: $mod (v2.5 feature)" -ForegroundColor Yellow
        $WARN++
    }
}

# ── Check 4: Learning modules (v2.7) ─────────────────────────────────────────
Write-Host ""
Write-Host "Learning System (v2.7):"
$learningDir = Join-Path $INSTALL_DIR "learning"
if (Test-Path $learningDir) {
    Write-Host "  [PASS] learning/ directory exists" -ForegroundColor Green
    $PASS++

    $learningModules = @("__init__.py", "learning_db.py", "adaptive_ranker.py", "feedback_collector.py",
                         "engagement_tracker.py", "cross_project_aggregator.py", "project_context_manager.py",
                         "workflow_pattern_miner.py", "source_quality_scorer.py", "synthetic_bootstrap.py",
                         "feature_extractor.py")
    foreach ($mod in $learningModules) {
        $modPath = Join-Path $learningDir $mod
        if (Test-Path $modPath) {
            Write-Host "  [PASS] learning/$mod" -ForegroundColor Green
            $PASS++
        } else {
            Write-Host "  [WARN] Missing: learning/$mod" -ForegroundColor Yellow
            $WARN++
        }
    }
} else {
    Write-Host "  [FAIL] learning/ directory missing (v2.7 not fully installed)" -ForegroundColor Red
    $FAIL++
}

# ── Check 5: Learning dependencies ───────────────────────────────────────────
Write-Host ""
Write-Host "Learning Dependencies:"
try {
    python -c "import lightgbm; print(f'  [PASS] LightGBM {lightgbm.__version__}')" 2>$null
    if ($LASTEXITCODE -eq 0) {
        $PASS++
    } else {
        throw
    }
} catch {
    Write-Host "  [INFO] LightGBM not installed (optional — rule-based ranking will be used)" -ForegroundColor Yellow
    $WARN++
}

try {
    python -c "import scipy; print(f'  [PASS] SciPy {scipy.__version__}')" 2>$null
    if ($LASTEXITCODE -eq 0) {
        $PASS++
    } else {
        throw
    }
} catch {
    Write-Host "  [INFO] SciPy not installed (optional — install for full learning features)" -ForegroundColor Yellow
    $WARN++
}

# ── Check 6: Core dependencies ───────────────────────────────────────────────
Write-Host ""
Write-Host "Core Dependencies:"
try {
    python -c "import sklearn; print(f'  [PASS] scikit-learn {sklearn.__version__}')" 2>$null
    if ($LASTEXITCODE -eq 0) {
        $PASS++
    } else {
        throw
    }
} catch {
    Write-Host "  [WARN] scikit-learn not installed (needed for knowledge graph)" -ForegroundColor Yellow
    $WARN++
}

try {
    python -c "import numpy; print(f'  [PASS] numpy {numpy.__version__}')" 2>$null
    if ($LASTEXITCODE -eq 0) {
        $PASS++
    } else {
        throw
    }
} catch {
    Write-Host "  [WARN] numpy not installed" -ForegroundColor Yellow
    $WARN++
}

try {
    python -c "import igraph; print(f'  [PASS] python-igraph {igraph.__version__}')" 2>$null
    if ($LASTEXITCODE -eq 0) {
        $PASS++
    } else {
        throw
    }
} catch {
    Write-Host "  [WARN] python-igraph not installed (needed for graph clustering)" -ForegroundColor Yellow
    $WARN++
}

# ── Check 7: Database ────────────────────────────────────────────────────────
Write-Host ""
Write-Host "Databases:"
$memoryDb = Join-Path $INSTALL_DIR "memory.db"
if (Test-Path $memoryDb) {
    try {
        $MEMORY_COUNT = sqlite3 $memoryDb "SELECT COUNT(*) FROM memories;" 2>$null
        $DB_SIZE = [math]::Round((Get-Item $memoryDb).Length / 1KB, 2)
        Write-Host "  [PASS] memory.db exists ($MEMORY_COUNT memories, $DB_SIZE KB)" -ForegroundColor Green
        $PASS++
    } catch {
        Write-Host "  [INFO] memory.db exists but cannot query (sqlite3 not available)" -ForegroundColor Yellow
    }
} else {
    Write-Host "  [INFO] memory.db not yet created (will auto-create on first use)" -ForegroundColor Yellow
}

$learningDb = Join-Path $INSTALL_DIR "learning.db"
if (Test-Path $learningDb) {
    try {
        $FEEDBACK_COUNT = sqlite3 $learningDb "SELECT COUNT(*) FROM ranking_feedback;" 2>$null
        Write-Host "  [PASS] learning.db exists ($FEEDBACK_COUNT feedback signals)" -ForegroundColor Green
        $PASS++
    } catch {
        Write-Host "  [INFO] learning.db exists but cannot query (sqlite3 not available)" -ForegroundColor Yellow
    }
} else {
    Write-Host "  [INFO] learning.db not yet created (will auto-create on first recall)" -ForegroundColor Yellow
}

# ── Check 8: CLI ──────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "CLI:"
$slmCommand = Get-Command slm -ErrorAction SilentlyContinue
if ($slmCommand) {
    Write-Host "  [PASS] slm command available in PATH" -ForegroundColor Green
    $PASS++
} else {
    $slmBin = Join-Path $INSTALL_DIR "bin\slm"
    if (Test-Path $slmBin) {
        Write-Host "  [WARN] slm exists at $slmBin but not in PATH" -ForegroundColor Yellow
        Write-Host "         Add to PATH: `$env:PATH = `"`$env:USERPROFILE\.claude-memory\bin;`$env:PATH`""
        $WARN++
    } else {
        Write-Host "  [FAIL] slm command not found" -ForegroundColor Red
        $FAIL++
    }
}

# ── Check 9: MCP server ──────────────────────────────────────────────────────
Write-Host ""
Write-Host "MCP Server:"
$mcpServer = Join-Path $INSTALL_DIR "mcp_server.py"
if (Test-Path $mcpServer) {
    Write-Host "  [PASS] mcp_server.py installed" -ForegroundColor Green
    $PASS++
} else {
    Write-Host "  [FAIL] mcp_server.py missing" -ForegroundColor Red
    $FAIL++
}

try {
    python -c "from mcp.server.fastmcp import FastMCP" 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  [PASS] MCP SDK installed" -ForegroundColor Green
        $PASS++
    } else {
        throw
    }
} catch {
    Write-Host "  [WARN] MCP SDK not installed (install: pip install mcp)" -ForegroundColor Yellow
    $WARN++
}

# ── Check 10: Import chain verification ───────────────────────────────────────
Write-Host ""
Write-Host "Import Chain:"
$importTest = @"
import sys
sys.path.insert(0, '$($INSTALL_DIR -replace '\\', '\\')')
try:
    from learning import get_learning_db, get_status, FULL_LEARNING_AVAILABLE, ML_RANKING_AVAILABLE
    status = get_status()
    ml = 'yes' if status['ml_ranking_available'] else 'no'
    full = 'yes' if status['learning_available'] else 'no'
    print(f'OK ml_ranking={ml} full_learning={full}')
except ImportError as e:
    print(f'IMPORT_ERROR {e}')
except Exception as e:
    print(f'ERROR {e}')
"@

$IMPORT_RESULT = python -c $importTest 2>&1

if ($IMPORT_RESULT -like "OK*") {
    Write-Host "  [PASS] Learning system imports successfully" -ForegroundColor Green
    Write-Host "         $IMPORT_RESULT"
    $PASS++
} elseif ($IMPORT_RESULT -like "IMPORT_ERROR*") {
    Write-Host "  [WARN] Learning import failed: $($IMPORT_RESULT -replace 'IMPORT_ERROR ', '')" -ForegroundColor Yellow
    Write-Host "         This may be normal if learning modules are not yet installed."
    $WARN++
} else {
    Write-Host "  [WARN] Learning check: $IMPORT_RESULT" -ForegroundColor Yellow
    $WARN++
}

# ── Summary ───────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "==================================="
Write-Host "Verification Summary"
Write-Host "  Passed:   $PASS"
Write-Host "  Warnings: $WARN"
Write-Host "  Failed:   $FAIL"
Write-Host "==================================="
Write-Host ""

if ($FAIL -eq 0) {
    Write-Host "Status: READY" -ForegroundColor Green
    Write-Host ""
    Write-Host "Quick start:"
    Write-Host "  slm remember `"My first memory`""
    Write-Host "  slm recall `"first`""
    Write-Host "  slm status"
    Write-Host ""
    if ($WARN -gt 0) {
        Write-Host "Some optional features may not be available."
        Write-Host "Install missing dependencies to enable them:"
        Write-Host "  pip install lightgbm scipy        # Learning system"
        Write-Host "  pip install scikit-learn igraph   # Knowledge graph"
        Write-Host ""
    }
} else {
    Write-Host "Status: INCOMPLETE" -ForegroundColor Red
    Write-Host ""
    Write-Host "Fix the failed checks above, then re-run:"
    Write-Host "  .\scripts\verify-v27.ps1"
    Write-Host ""
    exit 1
}
