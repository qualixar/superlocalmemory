#!/usr/bin/env bash
# Refuse to publish internal process detail or private correspondence.
#
# This repository is public. Three separate things have leaked into it and been
# removed by hand after the fact: the owner's verbatim words about marketing
# positioning, the names and roles of the agents used to build it, and the name
# of the internal development method. None of that helps a reader of the code and
# all of it is expensive to un-publish, because git history keeps a copy.
#
# So the check runs before a push rather than after a discovery. It is
# deliberately narrow: it looks for the specific phrasings that leaked, not for
# every use of an ordinary word. Review annotations (CRIT, CRIT-1) and design-doc
# references (LLD §5) are NOT flagged — they read as normal engineering shorthand
# and sweeping them would be a large diff for no protection.
#
# Usage:
#   scripts/check-no-internal-leaks.sh            # scan tracked files
#   scripts/check-no-internal-leaks.sh --staged   # scan what is about to commit
#
# Optional private denylist: set SLM_LEAK_DENYLIST to the path of a file kept
# OUTSIDE this repository, one extended regex per line (case-insensitive, blank
# lines and lines starting with # ignored). Each line is scanned like the
# built-in patterns. Unset, or a missing file, changes nothing.
#
# Exit 0 clean, 1 if anything matched.

set -uo pipefail
cd "$(git rev-parse --show-toplevel)" || exit 2

MODE="${1:-tracked}"

# Each entry: <regex>|<why it must not ship>
PATTERNS=(
'[Aa]uthored by the release coordinator|NOT by implementers|Implementation agents may NOT|Any agent that edits this file|the implementer'"'"'s choice|OWNER CONTEXT|OWNER REQUIREMENT|OWNER DECISION|owner, verbatim|MEASURED ON THE OWNER|the owner runs |the owner set |owner-specified|owner reviews the|release coordinator|internal strategy, waves|internal roles or the agents used to build this'
)
WHY="internal roles, agent orchestration, or the owner's private words"

METHOD='iron[- ]?pattern|Iron Pattern|Qualixar Iron|Qualixar Dev Pattern|Varun Method|Stage [0-9]+/[0-9]+ audit|13-stage'
METHOD_WHY="the name or structure of the internal development method"

PROCESS='\b[Ww]ave [0-9]+\b|\bWave-[0-9]+\b|\bsprint [0-9]+\b|acceptance gates?\b|harsh.audit|re-?audit-?100x|pre-?release-?gate'
PROCESS_WHY="internal process vocabulary (waves, sprints, gates, audit stages)"

# Deliberately NOT a bare list of model names. This project legitimately names
# models as supported hosts and providers — "## Kimi" in the IDE setup docs, a
# "DeepSeek V3 (Free)" option in the settings UI, a certification test listing
# Cursor/Gemini/Grok/Muse as hosts. Flagging those trains the reader to ignore
# this script, which is worse than no script. What must not ship is a model
# named as one of the AGENTS THAT BUILT THIS, and that reads differently.
AGENTS='audited by (Grok|Muse|Kimi|DeepSeek|Claude)|(Grok|Muse|Kimi|DeepSeek) (audit|found|reported|flagged)|cross-audit|dual audit|tap__tap_|ultrathink|\bmegathink\b|\bAvenger|\bJarvis\b|Iron Man'
AGENTS_WHY="a model named as one of the agents that built this, or agent-orchestration vocabulary"

MARKETING='being marketed on|we are marketing it as|marketed on GDPR'
MARKETING_WHY="marketing positioning that is not a product fact"

if [ "$MODE" = "--staged" ]; then
  FILES=$(git diff --cached --name-only --diff-filter=ACMR)
else
  FILES=$(git ls-files)
fi
[ -z "$FILES" ] && { echo "no files to scan"; exit 0; }

FOUND=0
scan_with() {
  local flags="$1" pattern="$2" why="$3"
  local hits
  hits=$(printf '%s\n' "$FILES" \
    | grep -vE '(^|/)(\.backup|node_modules|dist|build)/' \
    | grep -vE '^scripts/check-no-internal-leaks\.sh$' \
    | tr '\n' '\0' \
    | xargs -0 grep "$flags" -- "$pattern" 2>/dev/null | head -20)
  if [ -n "$hits" ]; then
    FOUND=1
    echo ""
    echo "BLOCKED — $why:"
    printf '%s\n' "$hits" | sed 's/^/    /' | cut -c1-160
  fi
}
scan()    { scan_with -InE  "$1" "$2"; }
scan_ci() { scan_with -InEi "$1" "$2"; }

# Private denylist (optional). The names live in the caller's file, never here.
if [ -n "${SLM_LEAK_DENYLIST:-}" ] && [ -f "$SLM_LEAK_DENYLIST" ]; then
  while IFS= read -r line || [ -n "$line" ]; do
    line="${line%$'\r'}"
    case "$line" in ''|'#'*) continue ;; esac
    scan_ci "$line" "a term from your private denylist"
  done < "$SLM_LEAK_DENYLIST"
fi

scan "${PATTERNS[0]}" "$WHY"
scan "$METHOD"        "$METHOD_WHY"
scan "$PROCESS"       "$PROCESS_WHY"
scan "$AGENTS"        "$AGENTS_WHY"
scan "$MARKETING"     "$MARKETING_WHY"

if [ "$FOUND" -eq 1 ]; then
  cat <<'EOF'

Nothing was pushed. Rewrite the flagged text to state the technical fact and
drop the framing — what the code does, not who asked for it or which stage of
which process produced it. The one permitted attribution is:

    Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>

To override for a single push (only when a match is a genuine false positive):
    SLM_SKIP_LEAK_CHECK=1 git push ...
EOF
  exit 1
fi

echo "leak check: clean ($(printf '%s\n' "$FILES" | wc -l | tr -d ' ') files scanned)"
exit 0
