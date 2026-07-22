#!/usr/bin/env bash
# ensure-venv.sh (Codex edition) — SuperLocalMemory plugin venv bootstrap
#
# Idempotent: fast-path exits in <100ms on repeat invocations when requirements.txt
# is unchanged.
#
# Difference from plugin/scripts/ensure-venv.sh (Claude Code edition):
#   - Uses SLM_DATA_DIR instead of CLAUDE_PLUGIN_DATA.
#   - Resolves requirements.txt relative to this script's location instead of
#     CLAUDE_PLUGIN_ROOT (which is a Claude Code runtime variable with no Codex equivalent).
#
# Environment:
#   SLM_DATA_DIR — persistent data dir (venv lives here)
#                  (default: ~/.superlocalmemory)
#
# Exit codes: 0 = venv ready   non-0 = failure (logged to stderr)
# All output goes to stderr only (stdout reserved for MCP stdio protocol).

set -euo pipefail

# Redirect all output to stderr (MCP uses stdout for protocol messages)
exec 1>&2

# ---------------------------------------------------------------------------
# Resolve paths — script-relative so no runtime var needed
# ---------------------------------------------------------------------------
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# requirements.txt lives one level up from scripts/ (in codex-plugin/)
REQ="${_SCRIPT_DIR}/../requirements.txt"
_SLM_DATA_DIR="${SLM_DATA_DIR:-${HOME}/.superlocalmemory}"
VENV="${_SLM_DATA_DIR}/venv"
SENTINEL="${_SLM_DATA_DIR}/.venv-reqs.sha256"
VENV_TMP="${_SLM_DATA_DIR}/venv.tmp"

# ---------------------------------------------------------------------------
# Python >= 3.11 guard
# ---------------------------------------------------------------------------
if ! python3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)" 2>/dev/null; then
    PY_VER=$(python3 --version 2>&1 || echo "unknown")
    echo "ERROR: SuperLocalMemory plugin requires Python >= 3.11, found: ${PY_VER}" >&2
    echo "Install Python 3.11+ and ensure it is first on PATH." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Validate requirements.txt exists
# ---------------------------------------------------------------------------
if [ ! -f "${REQ}" ]; then
    echo "ERROR: requirements.txt not found at ${REQ}" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Compute sha256 of requirements.txt (cross-platform: prefer sha256sum, fall back to shasum)
# ---------------------------------------------------------------------------
hash_req() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "${REQ}" | awk '{print $1}'
    else
        shasum -a 256 "${REQ}" | awk '{print $1}'
    fi
}

NEW_HASH=$(hash_req)

# ---------------------------------------------------------------------------
# Fast-path: venv python exists AND sentinel matches current requirements hash
# ---------------------------------------------------------------------------
VENV_PYTHON="${VENV}/bin/python3"
if [ ! -x "${VENV_PYTHON}" ]; then
    VENV_PYTHON="${VENV}/bin/python"
fi
if [ -x "${VENV_PYTHON}" ] && [ -f "${SENTINEL}" ] && [ "$(cat "${SENTINEL}")" = "${NEW_HASH}" ]; then
    echo "SLM plugin: venv up-to-date (sha256=${NEW_HASH:0:12}…), skipping install." >&2
    exit 0
fi

# ---------------------------------------------------------------------------
# Rebuild venv atomically: install to venv.tmp, rename to venv, write sentinel LAST
# ---------------------------------------------------------------------------
echo "SLM plugin: bootstrapping Python venv at ${VENV} …" >&2
echo "  requirements: ${REQ}" >&2
echo "  python3: $(python3 --version 2>&1)" >&2

rm -rf "${VENV_TMP}"
python3 -m venv "${VENV_TMP}"
"${VENV_TMP}/bin/pip" install --upgrade pip --prefer-binary --quiet
"${VENV_TMP}/bin/pip" install \
    --require-virtualenv \
    --prefer-binary \
    --quiet \
    -r "${REQ}"

echo "SLM plugin: install complete, activating venv." >&2

rm -rf "${VENV}"
mv "${VENV_TMP}" "${VENV}"
echo "${NEW_HASH}" > "${SENTINEL}"

echo "SLM plugin: venv ready at ${VENV}/bin/slm" >&2
