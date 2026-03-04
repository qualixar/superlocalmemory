#!/bin/bash
# SuperLocalMemory V2 - SL_MEMORY_PATH Integration Test
# Tests that bin/slm respects the SL_MEMORY_PATH environment variable
#
# Run with:
#   bash tests/test-sl-memory-path.sh

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"

TESTS_PASSED=0
TESTS_FAILED=0

GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

pass() { echo -e "${GREEN}PASS${NC}: $1"; ((TESTS_PASSED++)); }
fail() { echo -e "${RED}FAIL${NC}: $1 -- $2"; ((TESTS_FAILED++)); }

TMPDIR_CUSTOM=$(mktemp -d)
trap 'rm -rf "$TMPDIR_CUSTOM"' EXIT

echo "SL_MEMORY_PATH Integration Tests"
echo "================================="

# Test 1: bin/slm respects SL_MEMORY_PATH when set
# We export SL_MEMORY_PATH pointing to an existing temp dir so the
# directory-existence check passes and slm reaches the version handler.
export SL_MEMORY_PATH="$TMPDIR_CUSTOM"
output=$(bash "$REPO_DIR/bin/slm" version 2>&1) || true
if echo "$output" | grep -qF "$TMPDIR_CUSTOM"; then
    pass "bin/slm uses SL_MEMORY_PATH path in output"
else
    fail "bin/slm uses SL_MEMORY_PATH path in output" "Expected '$TMPDIR_CUSTOM' in: $output"
fi

# Test 2: bin/slm does NOT mention default .claude-memory when SL_MEMORY_PATH overrides it
if echo "$output" | grep -qF ".claude-memory"; then
    fail "bin/slm ignores SL_MEMORY_PATH and still shows .claude-memory" "$output"
else
    pass "bin/slm does not fall back to .claude-memory when SL_MEMORY_PATH is set"
fi

# Test 3: bin/slm falls back to ~/.claude-memory when SL_MEMORY_PATH is unset
unset SL_MEMORY_PATH
output=$(bash "$REPO_DIR/bin/slm" version 2>&1) || true
if echo "$output" | grep -qF ".claude-memory"; then
    pass "bin/slm falls back to .claude-memory when SL_MEMORY_PATH is unset"
else
    fail "bin/slm falls back to .claude-memory when SL_MEMORY_PATH is unset" "Expected .claude-memory in: $output"
fi

# Test 4: bin/slm error message mentions custom path when dir does not exist
NONEXISTENT="/nonexistent/custom/memory/path"
export SL_MEMORY_PATH="$NONEXISTENT"
output=$(bash "$REPO_DIR/bin/slm" version 2>&1) || true
if echo "$output" | grep -qF "$NONEXISTENT"; then
    pass "bin/slm error message references SL_MEMORY_PATH value when dir missing"
else
    fail "bin/slm error message references SL_MEMORY_PATH value when dir missing" "Expected '$NONEXISTENT' in: $output"
fi
unset SL_MEMORY_PATH

echo ""
echo "================================="
echo "Passed: $TESTS_PASSED  Failed: $TESTS_FAILED"

[ "$TESTS_FAILED" -eq 0 ] && exit 0 || exit 1
