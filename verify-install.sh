#!/bin/bash
# ============================================================================
# SuperLocalMemory - Installation Verification Script
# Copyright (c) 2026 Varun Pratap Bhardwaj
# Licensed under MIT License
# Repository: https://github.com/varun369/SuperLocalMemoryV2
# ============================================================================

set -e

INSTALL_DIR="${HOME}/.claude-memory"

# Print banner
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  SuperLocalMemory - Installation Verification         ║"
echo "║  by Varun Pratap Bhardwaj                                    ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Track status
CORE_OK=true
SEARCH_OK=false
UI_OK=false
ERRORS=()

# ============================================================================
# CORE INSTALLATION CHECK
# ============================================================================

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "1. Core Installation"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check Python version
echo -n "Python 3.8+               "
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PYTHON_MAJOR=$(python3 -c 'import sys; print(sys.version_info.major)')
PYTHON_MINOR=$(python3 -c 'import sys; print(sys.version_info.minor)')

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
    echo "✗ FAIL (found $PYTHON_VERSION)"
    CORE_OK=false
    ERRORS+=("Python 3.8+ required")
else
    echo "✓ OK (v$PYTHON_VERSION)"
fi

# Check installation directory
echo -n "Installation directory    "
if [ -d "${INSTALL_DIR}" ]; then
    echo "✓ OK (${INSTALL_DIR})"
else
    echo "✗ FAIL (not found)"
    CORE_OK=false
    ERRORS+=("Installation directory missing")
fi

# Check core files
echo -n "Core scripts              "
MISSING_FILES=()
for file in memory_store_v2.py graph_engine.py pattern_learner.py; do
    if [ ! -f "${INSTALL_DIR}/${file}" ]; then
        MISSING_FILES+=("${file}")
    fi
done

if [ ${#MISSING_FILES[@]} -eq 0 ]; then
    echo "✓ OK"
else
    echo "✗ FAIL (missing: ${MISSING_FILES[*]})"
    CORE_OK=false
    ERRORS+=("Core scripts missing")
fi

# Check CLI wrappers
echo -n "CLI wrappers              "
if [ -d "${INSTALL_DIR}/bin" ] && [ -x "${INSTALL_DIR}/bin/slm" ]; then
    echo "✓ OK"
else
    echo "✗ FAIL"
    CORE_OK=false
    ERRORS+=("CLI wrappers missing or not executable")
fi

# Check PATH configuration
echo -n "PATH configuration        "
if command -v slm &>/dev/null; then
    echo "✓ OK (commands globally available)"
elif grep -q ".claude-memory/bin" "${HOME}/.zshrc" 2>/dev/null || grep -q ".claude-memory/bin" "${HOME}/.bashrc" 2>/dev/null || grep -q ".claude-memory/bin" "${HOME}/.bash_profile" 2>/dev/null; then
    echo "○ PARTIAL (configured but needs shell restart)"
else
    echo "⚠️  WARNING (not in PATH)"
    ERRORS+=("PATH not configured - run: source ~/.zshrc (or ~/.bashrc)")
fi

# Check database
echo -n "Database                  "
if [ -f "${INSTALL_DIR}/memory.db" ]; then
    DB_SIZE=$(du -h "${INSTALL_DIR}/memory.db" | cut -f1)
    echo "✓ OK (${DB_SIZE})"
else
    echo "○ NOT CREATED (will be created on first use)"
fi

# Check config
echo -n "Configuration             "
if [ -f "${INSTALL_DIR}/config.json" ]; then
    echo "✓ OK"
else
    echo "⚠️  WARNING (using defaults)"
fi

echo ""

# ============================================================================
# OPTIONAL FEATURES CHECK
# ============================================================================

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "2. Optional Features"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Advanced Search
echo -n "Advanced Search           "
if python3 -c "import sentence_transformers; import hnswlib" 2>/dev/null; then
    echo "✓ ENABLED"
    SEARCH_OK=true
else
    echo "○ DISABLED"
    echo "  Install: pip3 install -r requirements-search.txt"
fi

# Web Dashboard
echo -n "Web Dashboard             "
if python3 -c "import fastapi; import uvicorn" 2>/dev/null; then
    echo "✓ ENABLED"
    UI_OK=true
    if [ -f "${INSTALL_DIR}/api_server.py" ]; then
        echo "  Start: python3 ~/.claude-memory/api_server.py"
        echo "  URL:   http://127.0.0.1:8000"
    fi
else
    echo "○ DISABLED"
    echo "  Install: pip3 install -r requirements-ui.txt"
fi

echo ""

# ============================================================================
# PERFORMANCE QUICK TEST
# ============================================================================

if $CORE_OK; then
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "3. Performance Quick Test"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""

    # Test memory store initialization
    echo -n "Memory store init         "
    if python3 -c "import sys; sys.path.insert(0, '${INSTALL_DIR}'); from memory_store_v2 import MemoryStoreV2; store = MemoryStoreV2()" 2>/dev/null; then
        echo "✓ OK"
    else
        echo "✗ FAIL"
        ERRORS+=("Memory store initialization failed")
    fi

    # Test database query
    echo -n "Database query            "
    START_TIME=$(python3 -c "import time; print(time.time())")
    if python3 -c "import sys; sys.path.insert(0, '${INSTALL_DIR}'); from memory_store_v2 import MemoryStoreV2; store = MemoryStoreV2(); list(store.list_all(limit=1))" 2>/dev/null; then
        END_TIME=$(python3 -c "import time; print(time.time())")
        DURATION=$(python3 -c "print(f'{($END_TIME - $START_TIME) * 1000:.0f}ms')" 2>/dev/null || echo "OK")
        echo "✓ OK ($DURATION)"
    else
        echo "✗ FAIL"
        ERRORS+=("Database query failed")
    fi

    echo ""
fi

# ============================================================================
# SUMMARY
# ============================================================================

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "4. Summary"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if $CORE_OK; then
    echo "✓ Core Installation:      WORKING"
else
    echo "✗ Core Installation:      FAILED"
fi

if $SEARCH_OK; then
    echo "✓ Advanced Search:        ENABLED"
else
    echo "○ Advanced Search:        DISABLED (optional)"
fi

if $UI_OK; then
    echo "✓ Web Dashboard:          ENABLED"
else
    echo "○ Web Dashboard:          DISABLED (optional)"
fi

echo ""

# Feature Status
echo "Feature Status:"
echo "  • Basic CLI commands:    $(if $CORE_OK; then echo '✓ Available'; else echo '✗ Not available'; fi)"
echo "  • MCP Server:            $(if $CORE_OK; then echo '✓ Available'; else echo '✗ Not available'; fi)"
echo "  • Skills:                $(if $CORE_OK; then echo '✓ Available'; else echo '✗ Not available'; fi)"
echo "  • Semantic Search:       $(if $SEARCH_OK; then echo '✓ Enabled'; else echo '○ Disabled'; fi)"
echo "  • Web Interface:         $(if $UI_OK; then echo '✓ Enabled'; else echo '○ Disabled'; fi)"
echo ""

# Errors
if [ ${#ERRORS[@]} -gt 0 ]; then
    echo "⚠️  Errors detected:"
    for error in "${ERRORS[@]}"; do
        echo "  • $error"
    done
    echo ""
fi

# Next Steps
echo "Next Steps:"
echo ""

if $CORE_OK; then
    echo "  Try it now:"
    echo "    slm status"
    echo "    slm remember 'My first memory'"
    echo "    slm recall 'first'"
    echo ""

    if ! $SEARCH_OK && ! $UI_OK; then
        echo "  Install optional features:"
        echo "    pip3 install -r requirements-search.txt  # Advanced search"
        echo "    pip3 install -r requirements-ui.txt      # Web dashboard"
        echo "    pip3 install -r requirements-full.txt    # Everything"
        echo ""
    fi
else
    echo "  Fix installation issues:"
    echo "    ./install.sh"
    echo ""
fi

# Exit code
if $CORE_OK; then
    echo "✓ Installation verification PASSED"
    echo ""
    exit 0
else
    echo "✗ Installation verification FAILED"
    echo ""
    exit 1
fi
