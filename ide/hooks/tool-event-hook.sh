#!/usr/bin/env bash
# SuperLocalMemory V3.4.10 — Enriched Tool Event Learning Hook
# Copyright (c) 2026 Varun Pratap Bhardwaj
# Licensed under AGPL-3.0-or-later
#
# PostToolUse hook that logs RICH tool events to SLM for behavioral learning.
# Captures: tool_name, input_summary (500 chars), output_summary (500 chars),
#           session_id, project_path. Scrubs secrets before storing.
#
# Modeled after ECC observe.sh patterns — SLM is self-sufficient for observation.
# An end user with only SLM (no ECC) gets full learning quality.
#
# Installation: slm init (auto-registers) or manually add to settings.json:
#   { "type": "PostToolUse", "matcher": "*",
#     "command": "bash /path/to/tool-event-hook.sh",
#     "timeout": 5000 }

set -euo pipefail

# Read stdin (Claude Code passes JSON with tool_name, input, output)
INPUT=$(cat)

# Exit if no input
if [ -z "$INPUT" ]; then
  exit 0
fi

# Extract tool name (lightweight — bash-only fallback if python unavailable)
TOOL_NAME=""
if command -v python3 >/dev/null 2>&1; then
  PYTHON_CMD="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_CMD="python"
else
  # No python — fallback to basic extraction
  TOOL_NAME=$(echo "$INPUT" | grep -o '"tool_name"[[:space:]]*:[[:space:]]*"[^"]*"' | head -1 | sed 's/.*"tool_name"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/' 2>/dev/null || echo "unknown")
  PYTHON_CMD=""
fi

# Skip logging for our own tools (avoid recursion)
if [ -n "$PYTHON_CMD" ]; then
  TOOL_NAME=$(echo "$INPUT" | "$PYTHON_CMD" -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print(d.get('tool_name', d.get('tool', 'unknown')))
except:
    print('unknown')
" 2>/dev/null || echo "unknown")
fi

case "$TOOL_NAME" in
    log_tool_event|get_assertions|reinforce_assertion|contradict_assertion|\
    mcp__superlocalmemory__*|session_init|recall|remember|observe|mesh_*)
        exit 0
        ;;
esac

# Get daemon port
PORT=8765
DATA_ROOT="${SLM_DATA_DIR:-${SL_MEMORY_PATH:-${SLM_HOME:-$HOME/.superlocalmemory}}}"
PORT_FILE="$DATA_ROOT/daemon.port"
[ -f "$PORT_FILE" ] && PORT=$(cat "$PORT_FILE" 2>/dev/null || echo 8765)

# If no python, send basic event (backward compatible)
if [ -z "$PYTHON_CMD" ]; then
  curl -s -m 2 -X POST "http://127.0.0.1:${PORT}/api/v3/tool-event" \
      -H "Content-Type: application/json" \
      -d "{\"tool_name\": \"${TOOL_NAME}\", \"event_type\": \"complete\"}" \
      >/dev/null 2>&1 || true
  exit 0
fi

# Parse full JSON and build enriched payload (with secret scrubbing)
ENRICHED=$(echo "$INPUT" | "$PYTHON_CMD" -c '
import json, sys, os, re

# Secret scrubbing pattern (adopted from ECC observe.sh)
_SECRET_RE = re.compile(
    r"(?i)(api[_-]?key|[a-z_]*_key|token|secret|password|pass\b|authorization|credentials?|auth)"
    r"""([\"'"'"'"'"'"'\s:=]+)"""
    r"([A-Za-z]+\s+)?"
    r"([A-Za-z0-9_\-/.+=]{8,})"
)

def scrub(val, max_len=500):
    """Scrub secrets first, then truncate."""
    if val is None:
        return ""
    if isinstance(val, dict):
        s = json.dumps(val, default=str)
    elif isinstance(val, list):
        s = json.dumps(val, default=str)
    else:
        s = str(val)
    s = _SECRET_RE.sub(
        lambda m: m.group(1) + m.group(2) + (m.group(3) or "") + "[REDACTED]", s
    )
    return s[:max_len]

try:
    data = json.load(sys.stdin)

    tool_name = data.get("tool_name", data.get("tool", "unknown"))

    # Extract input — Claude Code uses tool_input or input
    raw_input = data.get("tool_input", data.get("input", ""))
    input_summary = scrub(raw_input, 500)

    # Extract output — Claude Code uses tool_response, tool_output, or output
    raw_output = data.get("tool_response")
    if raw_output is None:
        raw_output = data.get("tool_output", data.get("output", ""))
    output_summary = scrub(raw_output, 500)

    session_id = data.get("session_id", "")
    project_path = data.get("cwd", "")

    payload = {
        "tool_name": tool_name,
        "event_type": "complete",
        "input_summary": input_summary,
        "output_summary": output_summary,
        "session_id": session_id,
        "project_path": project_path,
    }
    print(json.dumps(payload))
except Exception:
    # Fallback — send basic event
    print(json.dumps({"tool_name": "unknown", "event_type": "complete"}))
' 2>/dev/null || echo '{"tool_name":"unknown","event_type":"complete"}')

# Send enriched event to SLM daemon (fire and forget, 2s timeout)
echo "$ENRICHED" | curl -s -m 2 -X POST "http://127.0.0.1:${PORT}/api/v3/tool-event" \
    -H "Content-Type: application/json" \
    -d @- \
    >/dev/null 2>&1 || true
