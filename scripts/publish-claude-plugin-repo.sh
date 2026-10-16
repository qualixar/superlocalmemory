#!/usr/bin/env bash
# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
#
# publish-claude-plugin-repo.sh — kept for the release checklist and muscle
# memory; the release workflow now does this on every tag. The logic lives in
# scripts/release/claude-plugin-repo.sh (same behaviour, plus --dry-run).
#
#   scripts/publish-claude-plugin-repo.sh [--dry-run] [--version X.Y.Z]

set -euo pipefail
exec "$(cd "$(dirname "$0")" && pwd)/release/claude-plugin-repo.sh" "$@"
