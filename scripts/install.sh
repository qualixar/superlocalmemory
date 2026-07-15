#!/usr/bin/env bash
# SuperLocalMemory isolated installer for macOS and Linux.
#
# This script manages application code only. Runtime state is owned by the
# SuperLocalMemory application and is never read, copied, moved, or removed
# here. An existing uv or pipx installation is required; this script does not
# bootstrap package managers or modify the operating system.

set -euo pipefail

readonly PACKAGE_NAME="superlocalmemory"

ACTION="install"
ACTION_SET=false
REQUESTED_MANAGER=""
PACKAGE_OVERRIDE=""
PACKAGE_OVERRIDDEN=false
DRY_RUN=false

usage() {
    cat <<'EOF'
Usage: install.sh [install|upgrade|uninstall] [OPTIONS]

Install SuperLocalMemory in an isolated, user-scoped tool environment.

Actions:
  install                 Install the selected package source (default)
  upgrade                 Upgrade an existing isolated installation
  uninstall               Remove application code; preserve memory data

Options:
  --manager uv|pipx       Select an already-installed tool manager
  --package SPEC          Install a wheel, project path, or package spec
  --dry-run               Print the mutating command without running it
  --non-interactive       Compatibility flag; the installer never prompts
  --yes, -y               Alias for --non-interactive
  --help, -h              Show this help

When both managers own an installation, upgrade and uninstall require an
explicit --manager selection. Install prefers uv when both are available.
EOF
}

fail() {
    local message="$1"
    local status="${2:-2}"
    printf 'Error: %s\n' "$message" >&2
    exit "$status"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        install|upgrade|uninstall)
            if [[ "$ACTION_SET" == true ]]; then
                fail "only one action may be specified"
            fi
            ACTION="$1"
            ACTION_SET=true
            shift
            ;;
        --manager)
            [[ $# -ge 2 ]] || fail "--manager requires uv or pipx"
            REQUESTED_MANAGER="$2"
            shift 2
            ;;
        --manager=*)
            REQUESTED_MANAGER="${1#--manager=}"
            shift
            ;;
        --package)
            [[ $# -ge 2 ]] || fail "--package requires a wheel, project path, or package spec"
            PACKAGE_OVERRIDE="$2"
            PACKAGE_OVERRIDDEN=true
            shift 2
            ;;
        --package=*)
            PACKAGE_OVERRIDE="${1#--package=}"
            PACKAGE_OVERRIDDEN=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --non-interactive|--yes|-y)
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        --*)
            fail "unknown option: $1"
            ;;
        *)
            fail "unknown action: $1"
            ;;
    esac
done

if [[ -n "$REQUESTED_MANAGER" ]] && \
   [[ "$REQUESTED_MANAGER" != "uv" ]] && \
   [[ "$REQUESTED_MANAGER" != "pipx" ]]; then
    fail "--manager must be uv or pipx"
fi

if [[ "$PACKAGE_OVERRIDDEN" == true && -z "$PACKAGE_OVERRIDE" ]]; then
    fail "--package must not be empty"
fi

if [[ "$ACTION" == "uninstall" && "$PACKAGE_OVERRIDDEN" == true ]]; then
    fail "--package is not valid for uninstall; removal uses the application identity"
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
if [[ "$PACKAGE_OVERRIDDEN" == true ]]; then
    PACKAGE_SPEC="$PACKAGE_OVERRIDE"
elif [[ -f "$PROJECT_ROOT/pyproject.toml" ]]; then
    PACKAGE_SPEC="$PROJECT_ROOT"
else
    PACKAGE_SPEC="$PACKAGE_NAME"
fi

manager_available() {
    command -v "$1" >/dev/null 2>&1
}

manager_owns_package() {
    local manager="$1"
    local listing=""

    case "$manager" in
        uv)
            listing="$(uv tool list 2>/dev/null || true)"
            ;;
        pipx)
            listing="$(pipx list --short 2>/dev/null || true)"
            ;;
    esac

    printf '%s\n' "$listing" | grep -Eq \
        '^[[:space:]]*superlocalmemory([[:space:]]|$)'
}

select_manager() {
    if [[ -n "$REQUESTED_MANAGER" ]]; then
        manager_available "$REQUESTED_MANAGER" || \
            fail "$REQUESTED_MANAGER is not available on PATH" 127
        printf '%s\n' "$REQUESTED_MANAGER"
        return
    fi

    if [[ "$ACTION" == "install" ]]; then
        if manager_available uv; then
            printf '%s\n' "uv"
            return
        fi
        if manager_available pipx; then
            printf '%s\n' "pipx"
            return
        fi
        fail "install uv or pipx with your operating-system package manager, then retry" 127
    fi

    local uv_owns=false
    local pipx_owns=false
    if manager_available uv && manager_owns_package uv; then
        uv_owns=true
    fi
    if manager_available pipx && manager_owns_package pipx; then
        pipx_owns=true
    fi

    if [[ "$uv_owns" == true && "$pipx_owns" == true ]]; then
        fail "both uv and pipx own an installation; choose one with --manager"
    fi
    if [[ "$uv_owns" == true ]]; then
        printf '%s\n' "uv"
        return
    fi
    if [[ "$pipx_owns" == true ]]; then
        printf '%s\n' "pipx"
        return
    fi

    fail "no isolated installation was found; run install first or specify --manager"
}

MANAGER="$(select_manager)"

case "$MANAGER:$ACTION" in
    uv:install)
        COMMAND=(uv tool install "$PACKAGE_SPEC")
        ;;
    uv:upgrade)
        if [[ "$PACKAGE_SPEC" == "$PACKAGE_NAME" ]]; then
            COMMAND=(uv tool upgrade "$PACKAGE_NAME")
        else
            COMMAND=(uv tool install "$PACKAGE_SPEC")
        fi
        ;;
    uv:uninstall)
        COMMAND=(uv tool uninstall "$PACKAGE_NAME")
        ;;
    pipx:install)
        COMMAND=(pipx install "$PACKAGE_SPEC")
        ;;
    pipx:upgrade)
        if [[ "$PACKAGE_SPEC" == "$PACKAGE_NAME" ]]; then
            COMMAND=(pipx upgrade "$PACKAGE_NAME")
        else
            COMMAND=(pipx install --force "$PACKAGE_SPEC")
        fi
        ;;
    pipx:uninstall)
        COMMAND=(pipx uninstall "$PACKAGE_NAME")
        ;;
    *)
        fail "unsupported installer state"
        ;;
esac

if [[ "$DRY_RUN" == true ]]; then
    printf 'Dry run:'
    printf ' %q' "${COMMAND[@]}"
    printf '\n'
    exit 0
fi

printf 'Using %s isolated tool environment.\n' "$MANAGER"
set +e
"${COMMAND[@]}"
status=$?
set -e

if [[ $status -ne 0 ]]; then
    printf 'Error: %s failed through %s (exit %s).\n' \
        "$ACTION" "$MANAGER" "$status" >&2
    exit "$status"
fi

case "$ACTION" in
    install)
        printf 'SuperLocalMemory application installed successfully.\n'
        printf 'Run: slm setup\n'
        ;;
    upgrade)
        printf 'SuperLocalMemory application upgraded successfully.\n'
        ;;
    uninstall)
        printf 'SuperLocalMemory application removed; memory data was preserved.\n'
        ;;
esac
