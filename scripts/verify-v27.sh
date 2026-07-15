#!/bin/bash
# ============================================================================
# SuperLocalMemory V2.7 — Quick Verification Script
# Copyright (c) 2026 Varun Pratap Bhardwaj
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Repository: https://github.com/qualixar/superlocalmemory
#
# Run this after installation to verify everything works:
#   bash scripts/verify-v27.sh
# ============================================================================

INSTALL_DIR="${HOME}/.superlocalmemory"
PASS=0
WARN=0
FAIL=0

echo ""
echo "SuperLocalMemory v2.7 Verification"
echo "==================================="
echo ""

# ── Check 1: Installation directory ──────────────────────────────────────────
if [ -d "$INSTALL_DIR" ]; then
    echo "[PASS] Installation directory exists: $INSTALL_DIR"
    PASS=$((PASS + 1))
else
    echo "[FAIL] Installation directory missing. Run install.sh first."
    FAIL=$((FAIL + 1))
    echo ""
    echo "==================================="
    echo "Result: FAIL — SuperLocalMemory is not installed."
    echo "Run:  bash install.sh"
    exit 1
fi

# ── Check 2: Core modules ────────────────────────────────────────────────────
echo ""
echo "Core Modules:"
for mod in memory_store_v2.py graph_engine.py pattern_learner.py mcp_server.py tree_manager.py; do
    if [ -f "$INSTALL_DIR/$mod" ]; then
        echo "  [PASS] $mod"
        PASS=$((PASS + 1))
    else
        echo "  [FAIL] Missing: $mod"
        FAIL=$((FAIL + 1))
    fi
done

# ── Check 3: v2.5 modules ────────────────────────────────────────────────────
echo ""
echo "Event System (v2.5):"
for mod in event_bus.py subscription_manager.py webhook_dispatcher.py agent_registry.py provenance_tracker.py trust_scorer.py db_connection_manager.py; do
    if [ -f "$INSTALL_DIR/$mod" ]; then
        echo "  [PASS] $mod"
        PASS=$((PASS + 1))
    else
        echo "  [WARN] Missing: $mod (v2.5 feature)"
        WARN=$((WARN + 1))
    fi
done

# ── Check 4: Learning modules (v2.7) ─────────────────────────────────────────
echo ""
echo "Learning System (v2.7):"
if [ -d "$INSTALL_DIR/learning" ]; then
    echo "  [PASS] learning/ directory exists"
    PASS=$((PASS + 1))

    for mod in __init__.py learning_db.py adaptive_ranker.py feedback_collector.py \
               engagement_tracker.py cross_project_aggregator.py project_context_manager.py \
               workflow_pattern_miner.py source_quality_scorer.py synthetic_bootstrap.py \
               feature_extractor.py; do
        if [ -f "$INSTALL_DIR/learning/$mod" ]; then
            echo "  [PASS] learning/$mod"
            PASS=$((PASS + 1))
        else
            echo "  [WARN] Missing: learning/$mod"
            WARN=$((WARN + 1))
        fi
    done
else
    echo "  [FAIL] learning/ directory missing (v2.7 not fully installed)"
    FAIL=$((FAIL + 1))
fi

# ── Check 5: Learning dependencies ───────────────────────────────────────────
echo ""
echo "Learning Dependencies:"
python3 -c "import lightgbm; print(f'  [PASS] LightGBM {lightgbm.__version__}')" 2>/dev/null || {
    echo "  [INFO] LightGBM not installed (optional — rule-based ranking will be used)"
    WARN=$((WARN + 1))
}
python3 -c "import scipy; print(f'  [PASS] SciPy {scipy.__version__}')" 2>/dev/null || {
    echo "  [INFO] SciPy not installed (optional — install for full learning features)"
    WARN=$((WARN + 1))
}

# ── Check 6: Core dependencies ───────────────────────────────────────────────
echo ""
echo "Core Dependencies:"
python3 -c "import sklearn; print(f'  [PASS] scikit-learn {sklearn.__version__}')" 2>/dev/null || {
    echo "  [WARN] scikit-learn not installed (needed for knowledge graph)"
    WARN=$((WARN + 1))
}
python3 -c "import numpy; print(f'  [PASS] numpy {numpy.__version__}')" 2>/dev/null || {
    echo "  [WARN] numpy not installed"
    WARN=$((WARN + 1))
}
python3 -c "import igraph; print(f'  [PASS] python-igraph {igraph.__version__}')" 2>/dev/null || {
    echo "  [WARN] python-igraph not installed (needed for graph clustering)"
    WARN=$((WARN + 1))
}

# ── Check 7: Database ────────────────────────────────────────────────────────
echo ""
echo "Databases:"
if [ -f "$INSTALL_DIR/memory.db" ]; then
    MEMORY_COUNT=$(sqlite3 "$INSTALL_DIR/memory.db" "SELECT COUNT(*) FROM memories;" 2>/dev/null || echo "0")
    DB_SIZE=$(du -h "$INSTALL_DIR/memory.db" 2>/dev/null | cut -f1)
    echo "  [PASS] memory.db exists ($MEMORY_COUNT memories, $DB_SIZE)"
    PASS=$((PASS + 1))
else
    echo "  [INFO] memory.db not yet created (will auto-create on first use)"
fi

if [ -f "$INSTALL_DIR/learning.db" ]; then
    FEEDBACK_COUNT=$(sqlite3 "$INSTALL_DIR/learning.db" "SELECT COUNT(*) FROM ranking_feedback;" 2>/dev/null || echo "0")
    echo "  [PASS] learning.db exists ($FEEDBACK_COUNT feedback signals)"
    PASS=$((PASS + 1))
else
    echo "  [INFO] learning.db not yet created (will auto-create on first recall)"
fi

# ── Check 8: CLI ──────────────────────────────────────────────────────────────
echo ""
echo "CLI:"
if command -v slm &> /dev/null; then
    echo "  [PASS] slm command available in PATH"
    PASS=$((PASS + 1))
else
    if [ -f "$INSTALL_DIR/bin/slm" ]; then
        echo "  [WARN] slm exists at $INSTALL_DIR/bin/slm but not in PATH"
        echo "         Add to PATH: export PATH=\"\$HOME/.superlocalmemory/bin:\$PATH\""
        WARN=$((WARN + 1))
    else
        echo "  [FAIL] slm command not found"
        FAIL=$((FAIL + 1))
    fi
fi

# ── Check 9: MCP server ──────────────────────────────────────────────────────
echo ""
echo "MCP Server:"
if [ -f "$INSTALL_DIR/mcp_server.py" ]; then
    echo "  [PASS] mcp_server.py installed"
    PASS=$((PASS + 1))
else
    echo "  [FAIL] mcp_server.py missing"
    FAIL=$((FAIL + 1))
fi

if python3 -c "from mcp.server.fastmcp import FastMCP" 2>/dev/null; then
    echo "  [PASS] MCP SDK installed"
    PASS=$((PASS + 1))
else
    echo "  [WARN] MCP SDK not installed (install: pip3 install mcp)"
    WARN=$((WARN + 1))
fi

# ── Check 10: Import chain verification ───────────────────────────────────────
echo ""
echo "Import Chain:"
IMPORT_RESULT=$(python3 -c "
import sys
sys.path.insert(0, '$INSTALL_DIR')
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
" 2>&1)

if [[ "$IMPORT_RESULT" == OK* ]]; then
    echo "  [PASS] Learning system imports successfully"
    echo "         $IMPORT_RESULT"
    PASS=$((PASS + 1))
elif [[ "$IMPORT_RESULT" == IMPORT_ERROR* ]]; then
    echo "  [WARN] Learning import failed: ${IMPORT_RESULT#IMPORT_ERROR }"
    echo "         This may be normal if learning modules are not yet installed."
    WARN=$((WARN + 1))
else
    echo "  [WARN] Learning check: $IMPORT_RESULT"
    WARN=$((WARN + 1))
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "==================================="
echo "Verification Summary"
echo "  Passed:   $PASS"
echo "  Warnings: $WARN"
echo "  Failed:   $FAIL"
echo "==================================="
echo ""

if [ $FAIL -eq 0 ]; then
    echo "Status: READY"
    echo ""
    echo "Quick start:"
    echo "  slm remember \"My first memory\""
    echo "  slm recall \"first\""
    echo "  slm status"
    echo ""
    if [ $WARN -gt 0 ]; then
        echo "Some optional features may not be available."
        echo "Install missing dependencies to enable them:"
        echo "  pip3 install lightgbm scipy        # Learning system"
        echo "  pip3 install scikit-learn igraph    # Knowledge graph"
        echo ""
    fi
else
    echo "Status: INCOMPLETE"
    echo ""
    echo "Fix the failed checks above, then re-run:"
    echo "  bash scripts/verify-v27.sh"
    echo ""
    exit 1
fi
