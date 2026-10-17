#!/usr/bin/env bash
# The one CI gate: pull requests, pushes to main and the release job all run
# this script, so what a green PR proved is exactly what a release checks.
#
# It is the release contract, not the whole suite: packaging, plugins and
# versions, the CLI, MCP and hook surfaces, migrations, the LLM and encoding
# paths. It must stay fast (about 10 minutes on a GitHub macOS runner) and
# deterministic. The full suite runs once, locally, before every release
# (see CONTRIBUTING.md); it is not a CI job, because on shared runners its
# timing and process tests measured the runner, not SLM, and kept every
# release blocked.
#
# Usage: bash scripts/ci-gate.sh            (from the repository root)
set -euo pipefail

WORKERS="${CI_GATE_WORKERS:-3}"

# GitHub's macOS runner has 7 GB of memory and SLM refuses images and
# documents below 15 GiB, so the media tool contracts fail at setup there. The
# tests use fake media models; the documented developer override lets them
# run the same on the runner as on a 16 GB+ Mac.
export SLM_MEDIA_ALLOW_LOW_RAM=1

GATE_TESTS=(
  tests/test_version_consistency.py
  tests/test_ci_guards
  tests/test_plugin
  tests/test_packaging
  tests/test_architecture
  tests/test_cli_subparsers
  tests/test_llm_provider.py
  tests/test_mode_switch_preservation.py
  tests/test_dashboard_preserve_embedding.py
  tests/test_encoding
  tests/test_mcp
  tests/test_hooks
  tests/test_migrations
)

# The slowest files of these folders (about a fifth of the gate's time, measured
# 17 Oct 2026) run in the local full suite only: real virtualenv builds, media
# fixtures over remote MCP, and a test that times a deliberate hang.
SLOW_IN_FULL_SUITE_ONLY=(
  --ignore=tests/test_mcp/test_remote_by_id_writes_hide_media.py
  --ignore=tests/test_mcp/test_remote_more_reads_hide_media.py
  --ignore=tests/test_mcp/test_remote_delete_hides_media.py
  --ignore=tests/test_plugin/test_wp06_ensure_venv.py
  --ignore=tests/test_hooks/test_native_hook_event_evidence.py
  --ignore=tests/test_ci_guards/test_a_hung_test_is_named.py
)

python -m pytest "${GATE_TESTS[@]}" "${SLOW_IN_FULL_SUITE_ONLY[@]}" \
  -q --tb=short -n "$WORKERS" --durations=15
npm test
npm run check:plugin
