#!/bin/sh
# SuperLocalMemory one-line installer for macOS (Apple Silicon) and 64-bit Linux.
#
#   curl -fsSLo get-slm.sh https://raw.githubusercontent.com/qualixar/superlocalmemory/main/scripts/get-slm.sh && sh get-slm.sh
#   sh get-slm.sh --media          start the pictures and documents download now (on by default)
#
# No Python and no Node are needed.  The script installs `uv` (Astral's tool
# installer) when it is missing, lets uv fetch its own Python 3.12 for
# SuperLocalMemory, then sets everything up, starts it and connects the AI
# tools it finds.  It never uses sudo and never touches the system Python.
#
# Options
#   --media     also run `slm media enable --yes` (same as SLM_MEDIA=1)
#   -h, --help  show this text
#
# Environment
#   SLM_VERSION         pin an exact release, for example SLM_VERSION=4.2.2
#   SLM_MEDIA=1         same as --media
#   SLM_GET_SKIP_START=1  test hatch: skip starting the service, connecting
#                       agents, turning on media and opening the dashboard
#                       (steps 5, 6, 7 and 8) so nothing live is touched
#   SLM_GET_DRY_RUN=1   test hatch: print each command as "DRY-RUN: <command>"
#                       instead of running it
#
# Steps: 1 check the computer, 2 get uv, 3 install SuperLocalMemory,
# 4 set it up, 5 start it and load the text model, 6 connect your AI tools,
# 7 (optional) pictures and documents, then finish and open the dashboard.

UV_INSTALL_URL="https://astral.sh/uv/install.sh"
PYTHON_VERSION="3.12"
PACKAGE="superlocalmemory"
DEFAULT_PORT="8765"

set -eu

STEP=0
TOTAL=7
MEDIA=0
PATH_CHANGED=0
PARTIAL=0
TMPD=""
SLM_BIN=""

say() { printf '%s\n' "$*"; }
warn() { printf 'Note: %s\n' "$*" >&2; }
die() {
    printf '\nSorry, the install stopped: %s\n' "$*" >&2
    exit 1
}

cleanup() {
    if [ -n "${TMPD:-}" ] && [ -d "$TMPD" ]; then
        rm -rf "$TMPD"
    fi
}

is_on() {
    case "${1:-}" in
        1 | true | TRUE | True | yes | YES | Yes | on | ON | On) return 0 ;;
        *) return 1 ;;
    esac
}

is_dry() { is_on "${SLM_GET_DRY_RUN:-}"; }
skip_start() { is_on "${SLM_GET_SKIP_START:-}"; }

step() {
    STEP=$((STEP + 1))
    say ""
    say "Step $STEP of $TOTAL: $1"
}

usage() {
    # Static text: when piped from curl, $0 is the shell, not this file.
    say "SuperLocalMemory one-line installer"
    say ""
    say "  curl -fsSLo get-slm.sh https://raw.githubusercontent.com/qualixar/superlocalmemory/main/scripts/get-slm.sh && sh get-slm.sh"
    say "  sh get-slm.sh --media    start the pictures and documents download now (they are on by default)"
    say ""
    say "Environment: SLM_VERSION=<x.y.z> pins a release, SLM_MEDIA=1 is the same as --media."
}

# run_uv ARGS...  - run uv, or print the command in dry-run mode.
run_uv() {
    if is_dry; then
        printf 'DRY-RUN: uv %s\n' "$*"
        return 0
    fi
    uv "$@" </dev/null
}

# run_slm ARGS...  - run the installed slm by full path (never a look-alike on
# PATH), or print the command in dry-run mode.
run_slm() {
    if is_dry; then
        printf 'DRY-RUN: slm %s\n' "$*"
        return 0
    fi
    "$SLM_BIN" "$@" </dev/null
}

parse_args() {
    if is_on "${SLM_MEDIA:-}"; then
        MEDIA=1
    fi
    for arg in "$@"; do
        case "$arg" in
            --media) MEDIA=1 ;;
            -h | --help)
                usage
                exit 0
                ;;
            *) die "I do not know the option '$arg'. The only option is --media." ;;
        esac
    done
    if [ "$MEDIA" = 1 ]; then
        TOTAL=8
    fi
    if [ -n "${SLM_VERSION:-}" ]; then
        case "$SLM_VERSION" in
            *[!A-Za-z0-9.+_-]*) die "SLM_VERSION may only contain letters, digits and . + _ - (got '$SLM_VERSION')." ;;
        esac
    fi
}

# ---- Step 1 ---------------------------------------------------------------

check_platform() {
    os=$(uname -s 2>/dev/null || echo unknown)
    arch=$(uname -m 2>/dev/null || echo unknown)
    case "$os" in
        Darwin)
            case "$arch" in
                arm64) ;;
                x86_64)
                    if [ "$(sysctl -n hw.optional.arm64 2>/dev/null || echo 0)" = 1 ]; then
                        die "this Terminal is running under Rosetta (Intel emulation) on an Apple Silicon Mac. Open a normal Terminal window (Get Info on Terminal, untick 'Open using Rosetta') and paste the line again."
                    fi
                    die "Intel Macs are not supported. SuperLocalMemory needs an Apple Silicon Mac (M1 or newer)."
                    ;;
                *) die "this Mac type ($arch) is not supported. SuperLocalMemory needs an Apple Silicon Mac (M1 or newer)." ;;
            esac
            ;;
        Linux)
            case "$arch" in
                x86_64 | amd64 | aarch64 | arm64) ;;
                i?86 | armv6* | armv7* | arm) die "32-bit systems are not supported. SuperLocalMemory needs 64-bit Linux (x86-64 or ARM64)." ;;
                *) die "this processor type ($arch) is not supported. SuperLocalMemory needs 64-bit Linux (x86-64 or ARM64)." ;;
            esac
            ;;
        MINGW* | MSYS* | CYGWIN*)
            die "this looks like Windows. Paste the agent prompt from the README into your AI agent, or install with npm or pip (see the README)."
            ;;
        *) die "$os is not supported. SuperLocalMemory runs on Apple Silicon macOS, 64-bit Linux and 64-bit Windows." ;;
    esac
    say "This computer is supported ($os $arch)."
}

# ---- Step 2 ---------------------------------------------------------------

# First directory that holds a `uv` binary, in the order Astral's installer
# chooses: UV_INSTALL_DIR, UV_UNMANAGED_INSTALL, XDG_BIN_HOME,
# XDG_DATA_HOME/../bin, then $HOME/.local/bin ($HOME/.cargo/bin on old releases).
find_uv_dir() {
    for dir in \
        "${UV_INSTALL_DIR:-}" \
        "${UV_UNMANAGED_INSTALL:-}" \
        "${XDG_BIN_HOME:-}" \
        "${XDG_DATA_HOME:+${XDG_DATA_HOME}/../bin}" \
        "$HOME/.local/bin" \
        "$HOME/.cargo/bin"; do
        if [ -n "$dir" ] && [ -x "$dir/uv" ]; then
            printf '%s\n' "$dir"
            return 0
        fi
    done
    return 1
}

path_has() {
    case ":$PATH:" in
        *":$1:"*) return 0 ;;
        *) return 1 ;;
    esac
}

fetch_uv_installer() {
    target="$1"
    if command -v curl >/dev/null 2>&1; then
        curl -LsSf "$UV_INSTALL_URL" -o "$target" || return 1
    elif command -v wget >/dev/null 2>&1; then
        wget -qO "$target" "$UV_INSTALL_URL" || return 1
    else
        die "I need curl or wget to download uv, and found neither. Install curl with your package manager, then paste the line again."
    fi
}

install_uv() {
    if is_dry; then
        printf 'DRY-RUN: download %s, then run it with sh\n' "$UV_INSTALL_URL"
        return 0
    fi
    TMPD=$(mktemp -d "${TMPDIR:-/tmp}/get-slm.XXXXXX") || die "I could not make a temporary folder."
    fetch_uv_installer "$TMPD/uv-install.sh" || die "I could not download uv from $UV_INSTALL_URL. Check your internet connection and try again."
    sh "$TMPD/uv-install.sh" </dev/null || die "the official uv installer failed (messages above)."
    uv_dir=$(find_uv_dir) || die "uv was installed but I cannot find it. Open a new terminal and paste the line again."
    PATH="$uv_dir:$PATH"
    export PATH
    PATH_CHANGED=1
}

ensure_uv() {
    if command -v uv >/dev/null 2>&1; then
        say "uv is already installed."
        return 0
    fi
    if uv_dir=$(find_uv_dir); then
        # installed earlier but its folder is not on PATH yet
        PATH="$uv_dir:$PATH"
        export PATH
        say "uv is already installed (found in $uv_dir)."
        return 0
    fi
    say "Installing uv, the tool that fetches Python for you (no admin password needed)..."
    install_uv
    if ! is_dry; then
        command -v uv >/dev/null 2>&1 || die "uv still cannot be found after installing it."
    fi
}

# ---- Step 3 ---------------------------------------------------------------

tool_bin_dir() {
    dir=""
    if ! is_dry || command -v uv >/dev/null 2>&1; then
        dir=$(uv tool dir --bin 2>/dev/null </dev/null || true)
    fi
    if [ -z "$dir" ]; then
        dir="${XDG_BIN_HOME:-$HOME/.local/bin}"
    fi
    printf '%s\n' "$dir"
}

already_installed() {
    if ! command -v uv >/dev/null 2>&1; then
        return 1
    fi
    uv tool list 2>/dev/null </dev/null | grep -q "^$PACKAGE "
}

install_slm() {
    TOOL_BIN=$(tool_bin_dir)
    if [ -n "${SLM_VERSION:-}" ]; then
        say "Installing SuperLocalMemory $SLM_VERSION (uv fetches Python $PYTHON_VERSION itself)..."
        run_uv tool install --python "$PYTHON_VERSION" --force "$PACKAGE==$SLM_VERSION" ||
            die "uv could not install $PACKAGE==$SLM_VERSION (messages above). Check the version number and your internet connection."
    elif already_installed; then
        say "SuperLocalMemory is already installed. Upgrading it..."
        run_uv tool upgrade "$PACKAGE" ||
            die "uv could not upgrade $PACKAGE (messages above)."
    else
        say "Installing SuperLocalMemory (uv fetches Python $PYTHON_VERSION itself; this can take a few minutes)..."
        run_uv tool install --python "$PYTHON_VERSION" "$PACKAGE" ||
            die "uv could not install $PACKAGE (messages above). Check your internet connection and try again."
    fi
    if ! path_has "$TOOL_BIN"; then
        # update-shell first: uv skips the edit when the folder is already on PATH.
        run_uv tool update-shell ||
            warn "I could not update your shell settings automatically. Add $TOOL_BIN to your PATH yourself."
        PATH="$TOOL_BIN:$PATH"
        export PATH
        PATH_CHANGED=1
    fi
    SLM_BIN="$TOOL_BIN/slm"
    if ! is_dry && [ ! -x "$SLM_BIN" ]; then
        die "the install finished but the slm command is missing from $TOOL_BIN."
    fi
}

# ---- Steps 4 to 8 ----------------------------------------------------------

setup_slm() {
    run_slm setup --auto || die "'slm setup --auto' failed (messages above). Run 'slm doctor' to see what is wrong."
}

start_slm() {
    run_slm serve start || die "SuperLocalMemory would not start (messages above). Run 'slm doctor' to see what is wrong."
    say "Loading the text model (the first time this downloads about 500 MB)..."
    run_slm warmup || die "the text model did not load (messages above). Run 'slm doctor' to see what is wrong."
}

connect_agents() {
    if ! run_slm connect; then
        PARTIAL=1
        warn "I could not connect every AI tool. Run 'slm connect' again later, or 'slm connect --list' to see what is supported."
    fi
}

enable_media() {
    if run_slm media enable --yes; then
        say "Pictures and documents: the extra models download in the background. Check with 'slm media status'."
    else
        warn "Pictures and documents were not turned on (reason above). Text memory works fine without them."
    fi
}

has_display() {
    if [ "$(uname -s 2>/dev/null || echo unknown)" = Darwin ]; then
        return 0
    fi
    [ -n "${DISPLAY:-}" ] || [ -n "${WAYLAND_DISPLAY:-}" ]
}

dashboard_url() {
    port=""
    py="$(uv tool dir 2>/dev/null </dev/null || true)/$PACKAGE/bin/python"
    if [ -x "$py" ]; then
        port=$("$py" -c 'from superlocalmemory.cli.deployment import runtime_port; print(runtime_port())' 2>/dev/null </dev/null || true)
    fi
    case "$port" in
        '' | *[!0-9]*) port="${SLM_DAEMON_PORT:-$DEFAULT_PORT}" ;;
    esac
    printf 'http://localhost:%s\n' "$port"
}

finish() {
    summary=""
    if is_dry; then
        run_slm doctor --quick
    else
        report=$("$SLM_BIN" doctor --quick </dev/null 2>&1) || warn "the health check reported a problem. Run 'slm doctor' to see it."
        summary=$(printf '%s\n' "$report" | grep '^Summary:' | tail -n 1 || true)
        if [ -z "$summary" ]; then
            summary=$(printf '%s\n' "$report" | tail -n 1)
        fi
        say "Health check: $summary"
    fi
    if is_dry; then
        run_slm dashboard
        return 0
    fi
    url=$(dashboard_url)
    if has_display; then
        if ! "$SLM_BIN" dashboard </dev/null >/dev/null 2>&1; then
            warn "I could not open the dashboard in your browser. Open it yourself:"
        fi
    fi
    say "Dashboard: $url"
}

main() {
    parse_args "$@"
    trap cleanup EXIT
    if [ -z "${HOME:-}" ]; then
        die "your HOME folder is not set, so I do not know where to install. Open a normal terminal and try again."
    fi

    say "SuperLocalMemory installer"

    step "checking your computer..."
    check_platform

    step "getting uv (the tool that installs SuperLocalMemory)..."
    ensure_uv

    step "installing SuperLocalMemory..."
    install_slm

    step "setting up SuperLocalMemory on this computer..."
    setup_slm

    if skip_start; then
        step "starting SuperLocalMemory (skipped: SLM_GET_SKIP_START)"
        step "connecting your AI tools (skipped: SLM_GET_SKIP_START)"
        if [ "$MEDIA" = 1 ]; then
            step "pictures and documents (skipped: SLM_GET_SKIP_START)"
        fi
        step "finishing (skipped: SLM_GET_SKIP_START)"
    else
        step "starting SuperLocalMemory and loading the text model..."
        start_slm

        step "connecting the AI tools I can find (Claude, Cursor, Codex and others)..."
        connect_agents

        if [ "$MEDIA" = 1 ]; then
            step "turning on pictures and documents..."
            enable_media
        fi

        step "finishing up..."
        finish
    fi

    say ""
    if is_dry; then
        say "Dry run finished: nothing was installed or changed."
    elif [ "$PARTIAL" = 1 ]; then
        say "SuperLocalMemory is installed, but one step needs attention (see the note above)."
    else
        say "Done. SuperLocalMemory is installed."
    fi
    if [ "$PATH_CHANGED" = 1 ]; then
        say "Open a new terminal window so the 'slm' command works everywhere."
    fi
    if [ "$PARTIAL" = 1 ]; then
        exit 1
    fi
}

main "$@"
