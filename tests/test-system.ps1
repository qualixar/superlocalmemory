# SuperLocalMemory V2 - Comprehensive System Test (PowerShell)
# Tests all components for production readiness

param()

$ErrorActionPreference = "Stop"

$MEMORY_DIR = Join-Path $env:USERPROFILE ".claude-memory"
$VENV_PYTHON = Join-Path $MEMORY_DIR "venv\Scripts\python.exe"

Write-Host "╔══════════════════════════════════════════════════════════╗"
Write-Host "║                                                          ║"
Write-Host "║   SuperLocalMemory V2 - Production Readiness Test       ║"
Write-Host "║                                                          ║"
Write-Host "╚══════════════════════════════════════════════════════════╝"
Write-Host ""

$TESTS_PASSED = 0
$TESTS_FAILED = 0

function Test-Passed {
    param([string]$Message)
    Write-Host "✓ PASSED: $Message" -ForegroundColor Green
    $script:TESTS_PASSED++
}

function Test-Failed {
    param([string]$Message, [string]$Error)
    Write-Host "✗ FAILED: $Message" -ForegroundColor Red
    Write-Host "  Error: $Error"
    $script:TESTS_FAILED++
}

function Test-Warning {
    param([string]$Message)
    Write-Host "⚠ WARNING: $Message" -ForegroundColor Yellow
}

Write-Host "════════════════════════════════════════════════════════════"
Write-Host "TEST 1: Database Schema Verification"
Write-Host "════════════════════════════════════════════════════════════"

# Check if database exists
$dbPath = Join-Path $MEMORY_DIR "memory.db"
if (Test-Path $dbPath) {
    Test-Passed "Database file exists"
} else {
    Test-Failed "Database file missing" "Expected: $dbPath"
    exit 1
}

# Check V2 tables exist
$tables = @("memories", "memory_tree", "graph_nodes", "graph_edges", "graph_clusters", "identity_patterns", "pattern_examples", "memory_archive")
foreach ($table in $tables) {
    try {
        $result = sqlite3 $dbPath "SELECT name FROM sqlite_master WHERE type='table' AND name='$table';"
        if ($result) {
            Test-Passed "Table '$table' exists"
        } else {
            Test-Failed "Table '$table' missing" "V2 schema incomplete"
        }
    } catch {
        Test-Failed "Table '$table' check failed" $_.Exception.Message
    }
}

# Check V2 columns in memories table
$columns = @("tier", "cluster_id", "tree_path", "parent_id", "depth", "category", "last_accessed", "access_count")
foreach ($column in $columns) {
    try {
        $result = sqlite3 $dbPath "PRAGMA table_info(memories);" | Select-String -Pattern "^[^|]*\|$column\|"
        if ($result) {
            Test-Passed "Column 'memories.$column' exists"
        } else {
            Test-Failed "Column 'memories.$column' missing" "V2 migration incomplete"
        }
    } catch {
        Test-Failed "Column check failed for '$column'" $_.Exception.Message
    }
}

Write-Host ""
Write-Host "════════════════════════════════════════════════════════════"
Write-Host "TEST 2: Memory Store Operations"
Write-Host "════════════════════════════════════════════════════════════"

# Test reading memories
try {
    $MEMORY_COUNT = sqlite3 $dbPath "SELECT COUNT(*) FROM memories;"
    if ([int]$MEMORY_COUNT -gt 0) {
        Test-Passed "Can read memories (found $MEMORY_COUNT)"
    } else {
        Test-Warning "No memories found (database empty)"
    }
} catch {
    Test-Failed "Cannot read memories" $_.Exception.Message
}

# Test V2 memory store
$v2Script = Join-Path $MEMORY_DIR "memory_store_v2.py"
if (Test-Path $v2Script) {
    try {
        & $VENV_PYTHON $v2Script stats 2>&1 | Out-Null
        if ($LASTEXITCODE -eq 0) {
            Test-Passed "V2 memory_store_v2.py works"
        } else {
            Test-Failed "V2 memory_store_v2.py broken" "Check Python syntax"
        }
    } catch {
        Test-Failed "V2 memory_store_v2.py broken" $_.Exception.Message
    }
} else {
    Test-Failed "V2 memory_store_v2.py missing" "Core component not found"
}

Write-Host ""
Write-Host "════════════════════════════════════════════════════════════"
Write-Host "TEST 3: Graph Engine"
Write-Host "════════════════════════════════════════════════════════════"

$graphScript = Join-Path $MEMORY_DIR "graph_engine.py"
if (-not (Test-Path $graphScript)) {
    Test-Failed "graph_engine.py missing" "Core component not found"
} else {
    try {
        & $VENV_PYTHON $graphScript stats 2>&1 | Out-Null
        if ($LASTEXITCODE -eq 0) {
            Test-Passed "Graph engine runs"

            # Check if graph data exists
            $NODE_COUNT = sqlite3 $dbPath "SELECT COUNT(*) FROM graph_nodes;"
            $EDGE_COUNT = sqlite3 $dbPath "SELECT COUNT(*) FROM graph_edges;"
            $CLUSTER_COUNT = sqlite3 $dbPath "SELECT COUNT(*) FROM graph_clusters;"

            if ([int]$NODE_COUNT -gt 0) {
                Test-Passed "Graph has nodes ($NODE_COUNT)"
            } else {
                Test-Warning "No graph nodes (run: graph_engine.py build)"
            }

            if ([int]$EDGE_COUNT -gt 0) {
                Test-Passed "Graph has edges ($EDGE_COUNT)"
            } else {
                Test-Warning "No graph edges (run: graph_engine.py build)"
            }

            if ([int]$CLUSTER_COUNT -gt 0) {
                Test-Passed "Graph has clusters ($CLUSTER_COUNT)"
            } else {
                Test-Warning "No clusters detected (run: graph_engine.py build)"
            }
        } else {
            Test-Failed "Graph engine broken" "Check dependencies (python-igraph, leidenalg)"
        }
    } catch {
        Test-Failed "Graph engine broken" $_.Exception.Message
    }
}

Write-Host ""
Write-Host "════════════════════════════════════════════════════════════"
Write-Host "TEST 4: Pattern Learning"
Write-Host "════════════════════════════════════════════════════════════"

$patternScript = Join-Path $MEMORY_DIR "pattern_learner.py"
if (-not (Test-Path $patternScript)) {
    Test-Failed "pattern_learner.py missing" "Core component not found"
} else {
    try {
        & $VENV_PYTHON $patternScript stats 2>&1 | Out-Null
        if ($LASTEXITCODE -eq 0) {
            Test-Passed "Pattern learner runs"

            # Check if patterns exist
            $PATTERN_COUNT = sqlite3 $dbPath "SELECT COUNT(*) FROM identity_patterns;"

            if ([int]$PATTERN_COUNT -gt 0) {
                Test-Passed "Patterns learned ($PATTERN_COUNT)"
            } else {
                Test-Warning "No patterns learned (run: pattern_learner.py update)"
            }
        } else {
            Test-Failed "Pattern learner broken" "Check Python syntax"
        }
    } catch {
        Test-Failed "Pattern learner broken" $_.Exception.Message
    }
}

Write-Host ""
Write-Host "════════════════════════════════════════════════════════════"
Write-Host "TEST 5: Tree Manager"
Write-Host "════════════════════════════════════════════════════════════"

$treeScript = Join-Path $MEMORY_DIR "tree_manager.py"
if (-not (Test-Path $treeScript)) {
    Test-Failed "tree_manager.py missing" "Core component not found"
} else {
    try {
        & $VENV_PYTHON $treeScript stats 2>&1 | Out-Null
        if ($LASTEXITCODE -eq 0) {
            Test-Passed "Tree manager runs"

            $TREE_NODE_COUNT = sqlite3 $dbPath "SELECT COUNT(*) FROM memory_tree;"
            if ([int]$TREE_NODE_COUNT -gt 0) {
                Test-Passed "Tree structure exists ($TREE_NODE_COUNT nodes)"
            } else {
                Test-Warning "No tree nodes (run: tree_manager.py build_tree)"
            }
        } else {
            Test-Failed "Tree manager broken" "Check Python syntax"
        }
    } catch {
        Test-Failed "Tree manager broken" $_.Exception.Message
    }
}

Write-Host ""
Write-Host "════════════════════════════════════════════════════════════"
Write-Host "TEST 6: Dependencies"
Write-Host "════════════════════════════════════════════════════════════"

# Check Python version
try {
    $PYTHON_VERSION = & $VENV_PYTHON --version 2>&1
    Test-Passed "Python accessible ($PYTHON_VERSION)"
} catch {
    Test-Failed "Python not found" "Check venv installation"
}

# Check critical dependencies
$packages = @("sklearn", "numpy", "igraph", "leidenalg")
foreach ($package in $packages) {
    try {
        & $VENV_PYTHON -c "import $package" 2>&1 | Out-Null
        if ($LASTEXITCODE -eq 0) {
            Test-Passed "Package '$package' installed"
        } else {
            Test-Failed "Package '$package' missing" "Run: pip install $package"
        }
    } catch {
        Test-Failed "Package '$package' missing" "Run: pip install $package"
    }
}

Write-Host ""
Write-Host "╔══════════════════════════════════════════════════════════╗"
Write-Host "║                    TEST SUMMARY                          ║"
Write-Host "╚══════════════════════════════════════════════════════════╝"
Write-Host ""
Write-Host "Tests Passed: $TESTS_PASSED" -ForegroundColor Green
Write-Host "Tests Failed: $TESTS_FAILED" -ForegroundColor Red
Write-Host ""

if ($TESTS_FAILED -eq 0) {
    Write-Host "✓ ALL CRITICAL TESTS PASSED" -ForegroundColor Green
    Write-Host ""
    Write-Host "SuperLocalMemory V2 is PRODUCTION READY!"
    Write-Host ""
    Write-Host "Next steps:"
    Write-Host "  1. Start using memory commands in your AI tools"
    Write-Host "  2. Build graph: python $graphScript build"
    Write-Host "  3. Learn patterns: python $patternScript update"
    Write-Host ""
    exit 0
} else {
    Write-Host "✗ SOME TESTS FAILED" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please fix the failed tests before using in production."
    Write-Host "Check errors above for details."
    Write-Host ""
    exit 1
}
