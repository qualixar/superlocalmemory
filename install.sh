#!/bin/bash
# ============================================================================
# SuperLocalMemory V2 Installation Script
# Copyright (c) 2026 Varun Pratap Bhardwaj
# Licensed under MIT License
# Repository: https://github.com/varun369/SuperLocalMemoryV2
# ============================================================================

set -e

INSTALL_DIR="${HOME}/.claude-memory"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"

# Parse command line arguments
NON_INTERACTIVE=false

# Auto-detect non-interactive environment (Docker, CI/CD, pipes)
if [ ! -t 0 ] || [ ! -t 1 ]; then
    NON_INTERACTIVE=true
fi

for arg in "$@"; do
    case $arg in
        --non-interactive|--auto|--yes|-y)
            NON_INTERACTIVE=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --non-interactive, --auto, --yes, -y"
            echo "                    Skip interactive prompts (for scripts/automation)"
            echo "  --help, -h        Show this help message"
            echo ""
            echo "Note: Non-interactive mode is auto-detected when running in"
            echo "      Docker, CI/CD, or piped environments."
            echo ""
            exit 0
            ;;
    esac
done

# Print banner
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  SuperLocalMemory V2 - Installation                          ║"
echo "║  by Varun Pratap Bhardwaj                                    ║"
echo "║  https://github.com/varun369/SuperLocalMemoryV2              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Show mode if non-interactive
if [ "$NON_INTERACTIVE" = true ]; then
    echo "🤖 Running in non-interactive mode (auto-detected)"
    echo "   Skipping optional prompts, using defaults"
    echo ""
fi

# Check Python version — install if missing (non-tech user friendly)
echo "Checking Python version..."

install_python() {
    echo ""
    echo "Python 3 not found. Attempting automatic installation..."
    if [ "$(uname)" = "Darwin" ]; then
        # macOS: try Homebrew first, then Xcode CLI tools
        if command -v brew &> /dev/null; then
            echo "Installing Python via Homebrew..."
            brew install python3 && return 0
        fi
        # Try installing Xcode Command Line Tools (includes Python 3)
        echo "Installing Xcode Command Line Tools (includes Python 3)..."
        echo "A system dialog may appear — click 'Install' to continue."
        xcode-select --install 2>/dev/null
        # Wait for user to complete the install dialog
        echo "Waiting for Xcode CLI tools installation to complete..."
        echo "Press Enter after the installation finishes."
        if [ "$NON_INTERACTIVE" = false ]; then
            read -r
        else
            # In non-interactive mode, wait and retry
            sleep 30
        fi
        if command -v python3 &> /dev/null; then
            return 0
        fi
        # Last resort: direct Python.org installer
        echo ""
        echo "Automatic installation could not complete."
        echo "Please install Python 3.10+ from: https://www.python.org/downloads/"
        echo "Then re-run this installer."
        return 1
    elif [ -f /etc/debian_version ]; then
        # Debian/Ubuntu
        echo "Installing Python via apt..."
        sudo apt-get update -qq && sudo apt-get install -y python3 python3-pip && return 0
    elif [ -f /etc/redhat-release ]; then
        # RHEL/CentOS/Fedora
        echo "Installing Python via dnf..."
        sudo dnf install -y python3 python3-pip && return 0
    elif [ -f /etc/arch-release ]; then
        # Arch Linux
        sudo pacman -S --noconfirm python python-pip && return 0
    fi
    echo "Could not auto-install Python. Please install Python 3.8+ manually."
    echo "  macOS:  brew install python3"
    echo "  Ubuntu: sudo apt install python3 python3-pip"
    echo "  Fedora: sudo dnf install python3 python3-pip"
    return 1
}

if ! command -v python3 &> /dev/null; then
    install_python || exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PYTHON_MAJOR=$(python3 -c 'import sys; print(sys.version_info.major)')
PYTHON_MINOR=$(python3 -c 'import sys; print(sys.version_info.minor)')

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
    echo "Python $PYTHON_VERSION found but 3.8+ required."
    install_python || exit 1
    # Re-check after install
    PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    PYTHON_MAJOR=$(python3 -c 'import sys; print(sys.version_info.major)')
    PYTHON_MINOR=$(python3 -c 'import sys; print(sys.version_info.minor)')
    if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
        echo "✗ Error: Python 3.8+ still not available after install attempt"
        exit 1
    fi
fi
echo "✓ Python $PYTHON_VERSION"

# Ensure pip3 is available
if ! command -v pip3 &> /dev/null; then
    echo "Installing pip..."
    python3 -m ensurepip --upgrade 2>/dev/null || python3 -c "import urllib.request; urllib.request.urlretrieve('https://bootstrap.pypa.io/get-pip.py', '/tmp/get-pip.py')" && python3 /tmp/get-pip.py 2>/dev/null || true
fi

# Create installation directory
echo ""
echo "Creating installation directory..."
mkdir -p "${INSTALL_DIR}"
echo "✓ Directory: ${INSTALL_DIR}"

# Create universal symlink for non-Claude users
UNIVERSAL_LINK="${HOME}/.superlocalmemory"
if [ ! -e "${UNIVERSAL_LINK}" ]; then
    ln -s "${INSTALL_DIR}" "${UNIVERSAL_LINK}" 2>/dev/null && \
        echo "✓ Universal link created: ~/.superlocalmemory → ~/.claude-memory" || true
fi

# Copy source files
echo ""
echo "Copying source files..."
cp -r "${REPO_DIR}/src/"* "${INSTALL_DIR}/"
echo "✓ Source files copied"

# Copy learning modules explicitly (v2.7+ — ensures nested dir is handled)
if [ -d "${REPO_DIR}/src/learning" ]; then
    mkdir -p "${INSTALL_DIR}/learning"
    cp -r "${REPO_DIR}/src/learning/"* "${INSTALL_DIR}/learning/"
    echo "✓ Learning modules copied"
fi

# Copy lifecycle modules (v2.8+)
if [ -d "${REPO_DIR}/src/lifecycle" ]; then
    mkdir -p "${INSTALL_DIR}/lifecycle"
    cp -r "${REPO_DIR}/src/lifecycle/"* "${INSTALL_DIR}/lifecycle/"
    echo "✓ Lifecycle modules copied"
fi

# Copy behavioral modules (v2.8+)
if [ -d "${REPO_DIR}/src/behavioral" ]; then
    mkdir -p "${INSTALL_DIR}/behavioral"
    cp -r "${REPO_DIR}/src/behavioral/"* "${INSTALL_DIR}/behavioral/"
    echo "✓ Behavioral modules copied"
fi

# Copy compliance modules (v2.8+)
if [ -d "${REPO_DIR}/src/compliance" ]; then
    mkdir -p "${INSTALL_DIR}/compliance"
    cp -r "${REPO_DIR}/src/compliance/"* "${INSTALL_DIR}/compliance/"
    echo "✓ Compliance modules copied"
fi

# Copy hooks
echo "Copying hooks..."
mkdir -p "${INSTALL_DIR}/hooks"
if [ -d "${REPO_DIR}/hooks" ] && [ "$(ls -A ${REPO_DIR}/hooks)" ]; then
    cp -r "${REPO_DIR}/hooks/"* "${INSTALL_DIR}/hooks/"
    echo "✓ Hooks copied"
else
    echo "○ No hooks to copy"
fi

# Copy CLI wrappers
echo "Copying CLI wrappers..."
mkdir -p "${INSTALL_DIR}/bin"
cp -r "${REPO_DIR}/bin/"* "${INSTALL_DIR}/bin/"
chmod +x "${INSTALL_DIR}/bin/"*
echo "✓ CLI wrappers installed"

# Copy API server
if [ -f "${REPO_DIR}/api_server.py" ]; then
    cp "${REPO_DIR}/api_server.py" "${INSTALL_DIR}/"
    echo "✓ API server copied"
fi

# Copy UI server + dashboard files
if [ -f "${REPO_DIR}/ui_server.py" ]; then
    cp "${REPO_DIR}/ui_server.py" "${INSTALL_DIR}/"
    echo "✓ UI server copied"
fi

if [ -d "${REPO_DIR}/ui" ]; then
    mkdir -p "${INSTALL_DIR}/ui/js"
    cp "${REPO_DIR}/ui/index.html" "${INSTALL_DIR}/ui/" 2>/dev/null || true
    cp "${REPO_DIR}/ui/js/"*.js "${INSTALL_DIR}/ui/js/" 2>/dev/null || true
    echo "✓ Dashboard UI copied"
fi

# Copy route modules (v2.5+ dashboard API)
if [ -d "${REPO_DIR}/routes" ]; then
    mkdir -p "${INSTALL_DIR}/routes"
    cp "${REPO_DIR}/routes/"*.py "${INSTALL_DIR}/routes/"
    echo "✓ Dashboard routes copied"
fi

# Copy MCP server
if [ -f "${REPO_DIR}/mcp_server.py" ]; then
    cp "${REPO_DIR}/mcp_server.py" "${INSTALL_DIR}/"
    echo "✓ MCP server copied"
fi

# Copy config if not exists
if [ ! -f "${INSTALL_DIR}/config.json" ]; then
    echo "Creating default config..."
    cp "${REPO_DIR}/config.json" "${INSTALL_DIR}/config.json"
    echo "✓ Config created"
else
    echo "○ Config exists (keeping existing)"
fi

# Create necessary directories
echo ""
echo "Creating directories..."
mkdir -p "${INSTALL_DIR}/backups"
mkdir -p "${INSTALL_DIR}/profiles"
mkdir -p "${INSTALL_DIR}/vectors"
mkdir -p "${INSTALL_DIR}/cold-storage"
mkdir -p "${INSTALL_DIR}/jobs"
mkdir -p "${INSTALL_DIR}/policies"
echo "✓ Directories created"

# Initialize audit database (v2.8+ — compliance engine)
if [ ! -f "${INSTALL_DIR}/audit.db" ]; then
    python3 -c "
import sqlite3
from pathlib import Path
audit_path = Path.home() / '.claude-memory' / 'audit.db'
conn = sqlite3.connect(audit_path)
cursor = conn.cursor()
cursor.execute('''CREATE TABLE IF NOT EXISTS audit_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL,
    actor TEXT,
    target TEXT,
    action TEXT,
    detail TEXT,
    hash TEXT,
    prev_hash TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)''')
cursor.execute('''CREATE TABLE IF NOT EXISTS retention_policies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    policy_name TEXT UNIQUE NOT NULL,
    retention_days INTEGER,
    auto_delete INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)''')
conn.commit()
conn.close()
print('Audit database ready')
" 2>/dev/null && echo "✓ Audit database initialized (v2.8)" || echo "○ Audit database skipped (optional)"
fi

# Make Python scripts executable
chmod +x "${INSTALL_DIR}/"*.py 2>/dev/null || true

# Initialize database
echo ""
echo "Initializing database..."
if python3 "${INSTALL_DIR}/setup_validator.py" --init > /dev/null 2>&1; then
    echo "✓ Database initialized"
else
    # Fallback: create basic tables
    python3 -c "
import sqlite3
from pathlib import Path
db_path = Path.home() / '.claude-memory' / 'memory.db'
conn = sqlite3.connect(db_path)
cursor = conn.cursor()
cursor.execute('''CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content TEXT NOT NULL,
    summary TEXT,
    project_path TEXT,
    project_name TEXT,
    tags TEXT DEFAULT '[]',
    category TEXT,
    parent_id INTEGER,
    tree_path TEXT DEFAULT '/',
    depth INTEGER DEFAULT 0,
    memory_type TEXT DEFAULT 'session',
    importance INTEGER DEFAULT 5,
    content_hash TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_accessed TIMESTAMP,
    access_count INTEGER DEFAULT 0,
    compressed_at TIMESTAMP,
    tier INTEGER DEFAULT 1,
    cluster_id INTEGER
)''')
cursor.execute('CREATE TABLE IF NOT EXISTS graph_nodes (id INTEGER PRIMARY KEY, memory_id INTEGER UNIQUE, entities TEXT DEFAULT \"[]\", embedding_vector BLOB, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)')
cursor.execute('CREATE TABLE IF NOT EXISTS graph_edges (id INTEGER PRIMARY KEY, source_memory_id INTEGER, target_memory_id INTEGER, similarity REAL, relationship_type TEXT, shared_entities TEXT DEFAULT \"[]\", created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)')
cursor.execute('CREATE TABLE IF NOT EXISTS graph_clusters (id INTEGER PRIMARY KEY, cluster_name TEXT, description TEXT, memory_count INTEGER DEFAULT 0, avg_importance REAL DEFAULT 5.0, top_entities TEXT DEFAULT \"[]\", created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)')
cursor.execute('CREATE TABLE IF NOT EXISTS identity_patterns (id INTEGER PRIMARY KEY, pattern_type TEXT, pattern_key TEXT, pattern_value TEXT, confidence REAL DEFAULT 0.0, frequency INTEGER DEFAULT 1, last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)')
cursor.execute('CREATE TABLE IF NOT EXISTS pattern_examples (id INTEGER PRIMARY KEY, pattern_id INTEGER, memory_id INTEGER, context TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)')
cursor.execute('CREATE TABLE IF NOT EXISTS memory_tree (id INTEGER PRIMARY KEY, node_type TEXT, name TEXT, parent_id INTEGER, tree_path TEXT DEFAULT \"/\", depth INTEGER DEFAULT 0, memory_count INTEGER DEFAULT 0, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)')
cursor.execute('CREATE TABLE IF NOT EXISTS memory_archive (id INTEGER PRIMARY KEY, original_memory_id INTEGER, compressed_content TEXT, compression_type TEXT DEFAULT \"tier2\", original_size INTEGER, compressed_size INTEGER, archived_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)')
cursor.execute('CREATE TABLE IF NOT EXISTS system_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
cursor.execute(\"INSERT OR REPLACE INTO system_metadata (key, value) VALUES ('product', 'SuperLocalMemory V2'), ('author', 'Varun Pratap Bhardwaj'), ('repository', 'https://github.com/varun369/SuperLocalMemoryV2'), ('license', 'MIT'), ('schema_version', '2.0.0')\")
conn.commit()
conn.close()
print('Database ready')
" && echo "✓ Database initialized (fallback)"
fi

# Run v2.8 schema migration (adds lifecycle columns to existing databases)
echo ""
echo "Running schema migration..."
cd "$INSTALL_DIR"
python3 -c "
import sys
sys.path.insert(0, '.')
try:
    from memory_store_v2 import MemoryStoreV2
    MemoryStoreV2()
    print('  Schema migration complete')
except Exception as e:
    print(f'  Migration note: {e}')
" 2>/dev/null || echo "  Migration will run on first use"

# Install core dependencies (required for graph & dashboard)
echo ""
echo "Installing core dependencies..."
echo "⏳ This ensures graph visualization and patterns work out-of-box"

# Detect pip installation method
if pip3 install --help | grep -q "break-system-packages"; then
    PIP_FLAGS="--break-system-packages"
else
    PIP_FLAGS=""
fi

if [ -f "${REPO_DIR}/requirements-core.txt" ]; then
    if pip3 install $PIP_FLAGS -q -r "${REPO_DIR}/requirements-core.txt"; then
        echo "✓ Core dependencies installed (graph, dashboard, patterns)"
    else
        echo "⚠️  Core dependency installation failed. Some features may not work."
        echo "   Install manually: pip3 install -r ${REPO_DIR}/requirements-core.txt"
    fi
else
    echo "⚠️  requirements-core.txt not found, skipping dependency installation"
fi

# Install learning dependencies (v2.7+)
echo ""
echo "Installing learning dependencies..."
echo "  Enables intelligent pattern learning and personalized recall"

if [ -f "${REPO_DIR}/requirements-learning.txt" ]; then
    if pip3 install $PIP_FLAGS -q -r "${REPO_DIR}/requirements-learning.txt" 2>/dev/null; then
        echo "✓ Learning dependencies installed (personalized ranking enabled)"
    else
        echo "○ Learning dependencies skipped (core features unaffected)"
        echo "  To install later: pip3 install lightgbm scipy"
    fi
else
    echo "○ requirements-learning.txt not found (learning features will use rule-based ranking)"
fi

# Initialize knowledge graph and pattern learning
echo ""
echo "Initializing advanced features..."

# Add sample memories if database is empty (for first-time users)
MEMORY_COUNT=$(python3 -c "
import sqlite3
from pathlib import Path
db_path = Path.home() / '.claude-memory' / 'memory.db'
conn = sqlite3.connect(db_path)
cursor = conn.cursor()
cursor.execute('SELECT COUNT(*) FROM memories')
print(cursor.fetchone()[0])
conn.close()
" 2>/dev/null || echo "0")

if [ "$MEMORY_COUNT" -eq 0 ]; then
    echo "○ Adding sample memories for demonstration..."
    python3 "${INSTALL_DIR}/memory_store_v2.py" add "SuperLocalMemory V2 is a local-first, privacy-focused memory system for AI assistants. All data stays on your machine." --tags "supermemory,system,intro" --importance 8 > /dev/null 2>&1 || true
    python3 "${INSTALL_DIR}/memory_store_v2.py" add "Knowledge graph uses TF-IDF for entity extraction and Leiden clustering for community detection." --tags "architecture,graph" --importance 7 > /dev/null 2>&1 || true
    python3 "${INSTALL_DIR}/memory_store_v2.py" add "Pattern learning analyzes your coding preferences, style, and terminology to provide better context." --tags "architecture,patterns" --importance 7 > /dev/null 2>&1 || true
fi

# Build knowledge graph (Layer 3)
echo "○ Building knowledge graph..."
if python3 "${INSTALL_DIR}/graph_engine.py" build > /dev/null 2>&1; then
    echo "  ✓ Knowledge graph initialized"
else
    echo "  ⚠️  Graph build skipped (dependencies not installed)"
fi

# Run pattern learning (Layer 4)
echo "○ Learning patterns..."
if python3 "${INSTALL_DIR}/pattern_learner.py" update > /dev/null 2>&1; then
    PATTERN_COUNT=$(python3 -c "
import sqlite3
from pathlib import Path
db_path = Path.home() / '.claude-memory' / 'memory.db'
conn = sqlite3.connect(db_path)
cursor = conn.cursor()
cursor.execute('SELECT COUNT(*) FROM identity_patterns')
count = cursor.fetchone()[0]
conn.close()
print(count)
" 2>/dev/null || echo "0")
    echo "  ✓ Pattern learning complete ($PATTERN_COUNT patterns found)"
else
    echo "  ⚠️  Pattern learning skipped (dependencies not installed)"
fi

# Check optional dependencies
echo ""
echo "Checking optional dependencies..."
python3 -c "import sklearn" 2>/dev/null && echo "✓ scikit-learn (Knowledge Graph)" || echo "○ scikit-learn not installed (optional)"
python3 -c "import numpy" 2>/dev/null && echo "✓ numpy (Vector Operations)" || echo "○ numpy not installed (optional)"
python3 -c "import igraph" 2>/dev/null && echo "✓ python-igraph (Clustering)" || echo "○ python-igraph not installed (optional)"
python3 -c "import fastapi" 2>/dev/null && echo "✓ fastapi (UI Server)" || echo "○ fastapi not installed (optional)"

# Auto-configure PATH
echo ""
echo "Configuring PATH..."

# Detect user's default shell (not installer script's shell)
USER_SHELL="${SHELL:-/bin/bash}"
SHELL_CONFIG=""

if [[ "$USER_SHELL" == *"zsh"* ]]; then
    SHELL_CONFIG="${HOME}/.zshrc"
    # Create .zshrc if it doesn't exist
    touch "${SHELL_CONFIG}" 2>/dev/null
elif [[ "$USER_SHELL" == *"bash"* ]]; then
    # For bash, prefer .bash_profile on macOS, .bashrc on Linux
    if [[ "$(uname)" == "Darwin" ]] && [ -f "${HOME}/.bash_profile" ]; then
        SHELL_CONFIG="${HOME}/.bash_profile"
    else
        SHELL_CONFIG="${HOME}/.bashrc"
        touch "${SHELL_CONFIG}" 2>/dev/null
    fi
else
    # Fallback: check which config exists
    if [ -f "${HOME}/.zshrc" ]; then
        SHELL_CONFIG="${HOME}/.zshrc"
    elif [ -f "${HOME}/.bash_profile" ]; then
        SHELL_CONFIG="${HOME}/.bash_profile"
    elif [ -f "${HOME}/.bashrc" ]; then
        SHELL_CONFIG="${HOME}/.bashrc"
    else
        # Default to zsh on modern macOS
        SHELL_CONFIG="${HOME}/.zshrc"
        touch "${SHELL_CONFIG}"
    fi
fi

# Check if PATH already configured
PATH_EXPORT="export PATH=\"\${HOME}/.claude-memory/bin:\${PATH}\""
if grep -q ".claude-memory/bin" "${SHELL_CONFIG}" 2>/dev/null; then
    echo "○ PATH already configured in ${SHELL_CONFIG}"
else
    # Add PATH export to shell config
    echo "" >> "${SHELL_CONFIG}"
    echo "# SuperLocalMemory V2 - Added by installer on $(date '+%Y-%m-%d')" >> "${SHELL_CONFIG}"
    echo "${PATH_EXPORT}" >> "${SHELL_CONFIG}"
    echo "✓ PATH configured in ${SHELL_CONFIG}"
fi

# Add to current session PATH
export PATH="${HOME}/.claude-memory/bin:${PATH}"
echo "✓ Commands available in current session"

# ============================================================================
# UNIVERSAL INTEGRATION - Auto-detect and configure IDEs/CLI tools
# ============================================================================

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Universal Integration - Auto-Detection                      ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Detecting installed tools..."
echo ""

DETECTED_TOOLS=()

# Function to configure MCP for an IDE
configure_mcp() {
    local tool_name="$1"
    local config_source="$2"
    local config_dest="$3"

    # Replace {{INSTALL_DIR}} with actual path
    sed "s|{{INSTALL_DIR}}|${INSTALL_DIR}|g" "${config_source}" > /tmp/slm-config-$$.json

    # Create config directory if needed
    mkdir -p "$(dirname "${config_dest}")"

    # Backup existing config
    if [ -f "${config_dest}" ]; then
        cp "${config_dest}" "${config_dest}.backup.$(date +%Y%m%d-%H%M%S)"
        echo "  ✓ Backed up existing ${tool_name} config"
    fi

    # Install config
    cp /tmp/slm-config-$$.json "${config_dest}"
    rm /tmp/slm-config-$$.json

    echo "  ✓ ${tool_name} MCP configured"
}

# Copy MCP server to install directory
if [ -f "${REPO_DIR}/mcp_server.py" ]; then
    cp "${REPO_DIR}/mcp_server.py" "${INSTALL_DIR}/"
    chmod +x "${INSTALL_DIR}/mcp_server.py"
    echo "✓ MCP Server installed"
fi

# Detect Claude Desktop
if [ -d "${HOME}/Library/Application Support/Claude" ] || [ -d "${HOME}/.config/Claude" ]; then
    DETECTED_TOOLS+=("Claude Desktop")

    if [ -f "${REPO_DIR}/configs/claude-desktop-mcp.json" ]; then
        # Determine config path based on OS
        if [ -d "${HOME}/Library/Application Support/Claude" ]; then
            CONFIG_PATH="${HOME}/Library/Application Support/Claude/claude_desktop_config.json"
        else
            CONFIG_PATH="${HOME}/.config/Claude/claude_desktop_config.json"
        fi

        configure_mcp "Claude Desktop" \
            "${REPO_DIR}/configs/claude-desktop-mcp.json" \
            "${CONFIG_PATH}"
    fi
fi

# Detect Cursor
if [ -d "${HOME}/.cursor" ] || command -v cursor &>/dev/null; then
    DETECTED_TOOLS+=("Cursor")

    if [ -f "${REPO_DIR}/configs/cursor-mcp.json" ]; then
        configure_mcp "Cursor" \
            "${REPO_DIR}/configs/cursor-mcp.json" \
            "${HOME}/.cursor/mcp_settings.json"
    fi
fi

# Detect Windsurf
if [ -d "${HOME}/.windsurf" ] || command -v windsurf &>/dev/null; then
    DETECTED_TOOLS+=("Windsurf")

    if [ -f "${REPO_DIR}/configs/windsurf-mcp.json" ]; then
        configure_mcp "Windsurf" \
            "${REPO_DIR}/configs/windsurf-mcp.json" \
            "${HOME}/.windsurf/mcp_settings.json"
    fi
fi

# Detect VS Code with Continue
if [ -d "${HOME}/.continue" ]; then
    DETECTED_TOOLS+=("Continue.dev")

    if [ -f "${REPO_DIR}/configs/continue-mcp.yaml" ]; then
        # For Continue, append to config if exists, otherwise create
        CONTINUE_CONFIG="${HOME}/.continue/config.yaml"
        mkdir -p "${HOME}/.continue"

        if [ -f "${CONTINUE_CONFIG}" ]; then
            echo "  ○ Continue.dev config exists - manual merge recommended"
            echo "    See: ${REPO_DIR}/configs/continue-mcp.yaml"
        else
            sed "s|{{INSTALL_DIR}}|${INSTALL_DIR}|g" "${REPO_DIR}/configs/continue-mcp.yaml" > "${CONTINUE_CONFIG}"
            echo "  ✓ Continue.dev MCP configured"
        fi
    fi
fi

# Detect Zed Editor
if [ -d "${HOME}/.config/zed" ] || command -v zed &>/dev/null; then
    DETECTED_TOOLS+=("Zed Editor")

    if [ -f "${REPO_DIR}/configs/zed-mcp.json" ]; then
        configure_mcp "Zed Editor" \
            "${REPO_DIR}/configs/zed-mcp.json" \
            "${HOME}/.config/zed/context_servers.json"
    fi
fi

# Detect OpenCode
if [ -d "${HOME}/.opencode" ]; then
    DETECTED_TOOLS+=("OpenCode")

    if [ -f "${REPO_DIR}/configs/opencode-mcp.json" ]; then
        configure_mcp "OpenCode" \
            "${REPO_DIR}/configs/opencode-mcp.json" \
            "${HOME}/.opencode/mcp.json"
    fi
fi

# Detect Antigravity (Gemini)
if [ -d "${HOME}/.gemini/antigravity" ]; then
    DETECTED_TOOLS+=("Antigravity")

    if [ -f "${REPO_DIR}/configs/antigravity-mcp.json" ]; then
        configure_mcp "Antigravity" \
            "${REPO_DIR}/configs/antigravity-mcp.json" \
            "${HOME}/.gemini/antigravity/mcp_config.json"
    fi
fi

# Detect Perplexity
if [ -d "${HOME}/.perplexity" ]; then
    DETECTED_TOOLS+=("Perplexity")

    if [ -f "${REPO_DIR}/configs/perplexity-mcp.json" ]; then
        configure_mcp "Perplexity" \
            "${REPO_DIR}/configs/perplexity-mcp.json" \
            "${HOME}/.perplexity/mcp.json"
    fi
fi

# Detect ChatGPT Desktop (requires HTTP transport, not stdio)
if [ -d "${HOME}/Library/Application Support/ChatGPT" ] || [ -d "${HOME}/.config/ChatGPT" ]; then
    DETECTED_TOOLS+=("ChatGPT (manual)")
    echo "  ○ ChatGPT Desktop detected - requires HTTP transport"
    echo "    Run: slm serve  (then expose via ngrok for ChatGPT)"
    echo "    Guide: docs/MCP-MANUAL-SETUP.md#chatgpt-desktop-app"
fi

# Detect Cody (VS Code extension) - Works on macOS/Linux/Windows
if [ -d "${HOME}/.vscode/extensions" ] || [ -d "${HOME}/.config/Code/User/extensions" ]; then
    EXTENSIONS_DIR="${HOME}/.vscode/extensions"
    [ -d "${HOME}/.config/Code/User/extensions" ] && EXTENSIONS_DIR="${HOME}/.config/Code/User/extensions"

    if ls "${EXTENSIONS_DIR}" 2>/dev/null | grep -q "sourcegraph.cody"; then
        DETECTED_TOOLS+=("Cody (manual)")
        echo "  ○ Cody detected - manual setup required"
        echo "    See: docs/MCP-MANUAL-SETUP.md#cody-vs-codejetbrains"
    fi
fi

# Detect OpenAI Codex CLI
if [ -d "${HOME}/.codex" ] || command -v codex &>/dev/null; then
    DETECTED_TOOLS+=("Codex CLI")

    # Try native codex mcp add command first
    if command -v codex &>/dev/null; then
        if codex mcp add superlocalmemory-v2 --env "PYTHONPATH=${INSTALL_DIR}" -- python3 "${INSTALL_DIR}/mcp_server.py" 2>/dev/null; then
            echo "  ✓ Codex CLI MCP configured (via codex mcp add)"
        else
            # Fallback: Write TOML config
            CODEX_CONFIG="${HOME}/.codex/config.toml"
            mkdir -p "${HOME}/.codex"
            if [ -f "${CODEX_CONFIG}" ] && grep -q "superlocalmemory-v2" "${CODEX_CONFIG}" 2>/dev/null; then
                echo "  ○ Codex CLI already configured"
            else
                cp "${CODEX_CONFIG}" "${CODEX_CONFIG}.backup.$(date +%Y%m%d-%H%M%S)" 2>/dev/null || true
                cat >> "${CODEX_CONFIG}" <<TOML_EOF

[mcp_servers.superlocalmemory-v2]
command = "python3"
args = ["${INSTALL_DIR}/mcp_server.py"]

[mcp_servers.superlocalmemory-v2.env]
PYTHONPATH = "${INSTALL_DIR}"
TOML_EOF
                echo "  ✓ Codex CLI MCP configured (TOML appended)"
            fi
        fi
    else
        # codex command not in PATH but .codex dir exists
        CODEX_CONFIG="${HOME}/.codex/config.toml"
        mkdir -p "${HOME}/.codex"
        if [ -f "${CODEX_CONFIG}" ] && grep -q "superlocalmemory-v2" "${CODEX_CONFIG}" 2>/dev/null; then
            echo "  ○ Codex CLI already configured"
        else
            cp "${CODEX_CONFIG}" "${CODEX_CONFIG}.backup.$(date +%Y%m%d-%H%M%S)" 2>/dev/null || true
            cat >> "${CODEX_CONFIG}" <<TOML_EOF

[mcp_servers.superlocalmemory-v2]
command = "python3"
args = ["${INSTALL_DIR}/mcp_server.py"]

[mcp_servers.superlocalmemory-v2.env]
PYTHONPATH = "${INSTALL_DIR}"
TOML_EOF
            echo "  ✓ Codex CLI MCP configured (TOML appended)"
        fi
    fi
fi

# Detect VS Code / GitHub Copilot
if command -v code &>/dev/null || command -v code-insiders &>/dev/null; then
    DETECTED_TOOLS+=("VS Code/Copilot")

    if [ -f "${REPO_DIR}/configs/vscode-copilot-mcp.json" ]; then
        # VS Code user-level MCP config
        VSCODE_MCP="${HOME}/.vscode/mcp.json"
        mkdir -p "${HOME}/.vscode"

        if [ -f "${VSCODE_MCP}" ] && grep -q "superlocalmemory-v2" "${VSCODE_MCP}" 2>/dev/null; then
            echo "  ○ VS Code/Copilot already configured"
        else
            if [ -f "${VSCODE_MCP}" ]; then
                cp "${VSCODE_MCP}" "${VSCODE_MCP}.backup.$(date +%Y%m%d-%H%M%S)"
                echo "  ✓ Backed up existing VS Code MCP config"
            fi
            sed "s|{{INSTALL_DIR}}|${INSTALL_DIR}|g" "${REPO_DIR}/configs/vscode-copilot-mcp.json" > "${VSCODE_MCP}"
            echo "  ✓ VS Code/Copilot MCP configured"
        fi
    fi
fi

# Detect Gemini CLI (separate from Antigravity)
if command -v gemini &>/dev/null || [ -f "${HOME}/.gemini/settings.json" ]; then
    # Only add if not already detected as Antigravity
    if [[ ! " ${DETECTED_TOOLS[*]} " =~ " Antigravity " ]]; then
        DETECTED_TOOLS+=("Gemini CLI")
    else
        DETECTED_TOOLS+=("Gemini CLI")
    fi

    if [ -f "${REPO_DIR}/configs/gemini-cli-mcp.json" ]; then
        GEMINI_CONFIG="${HOME}/.gemini/settings.json"
        mkdir -p "${HOME}/.gemini"

        if [ -f "${GEMINI_CONFIG}" ] && grep -q "superlocalmemory-v2" "${GEMINI_CONFIG}" 2>/dev/null; then
            echo "  ○ Gemini CLI already configured"
        else
            if [ -f "${GEMINI_CONFIG}" ]; then
                cp "${GEMINI_CONFIG}" "${GEMINI_CONFIG}.backup.$(date +%Y%m%d-%H%M%S)"
                echo "  ✓ Backed up existing Gemini CLI config"
            fi
            sed "s|{{INSTALL_DIR}}|${INSTALL_DIR}|g" "${REPO_DIR}/configs/gemini-cli-mcp.json" > "${GEMINI_CONFIG}"
            echo "  ✓ Gemini CLI MCP configured"
        fi
    fi
fi

# Detect JetBrains IDEs (manual setup required - GUI-based)
if [ -d "${HOME}/Library/Application Support/JetBrains" ] || [ -d "${HOME}/.config/JetBrains" ]; then
    DETECTED_TOOLS+=("JetBrains (manual)")
    echo "  ○ JetBrains IDE detected - manual setup via GUI"
    echo "    Settings → AI Assistant → MCP Servers → Add"
    echo "    Template: configs/jetbrains-mcp.json"
fi

# Install Universal Skills (SKILL.md for all detected tools)
if [ -f "${REPO_DIR}/install-skills.sh" ]; then
    echo ""
    echo "Installing Universal Skills..."
    bash "${REPO_DIR}/install-skills.sh" --auto 2>/dev/null || echo "  ○ Skills installation skipped (optional)"
fi

# Install MCP Python package if not present
if ! python3 -c "import mcp" 2>/dev/null; then
    echo ""
    echo "Installing MCP SDK..."
    pip3 install mcp --quiet 2>/dev/null && echo "✓ MCP SDK installed" || echo "○ MCP SDK install failed (manual install: pip3 install mcp)"
fi

# Install bash completions
if [ -d "/usr/local/etc/bash_completion.d" ] && [ -f "${REPO_DIR}/completions/slm.bash" ]; then
    sudo cp "${REPO_DIR}/completions/slm.bash" /usr/local/etc/bash_completion.d/slm 2>/dev/null && echo "✓ Bash completion installed" || true
fi

# Summary of detected tools
echo ""
if [ ${#DETECTED_TOOLS[@]} -gt 0 ]; then
    echo "✓ Detected and configured:"
    for tool in "${DETECTED_TOOLS[@]}"; do
        echo "  • $tool"
    done
    echo ""
    echo "These tools now have native access to SuperLocalMemory!"
    echo "Restart them to use the new MCP integration."
else
    echo "○ No additional tools detected"
    echo "  MCP server is available if you install Cursor, Windsurf, etc."
fi

echo ""
echo "Universal CLI commands also available:"
echo "  slm remember <content>  - Simple command (anywhere)"
echo "  slm recall <query>      - Search from any terminal"
echo "  slm status              - Check system status"
echo ""

# Manual setup guide for other tools
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Manual Setup for Other Apps                                 ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Want to use SuperLocalMemory in other apps?"
echo ""
echo "  • ChatGPT Desktop - Add via Settings → MCP"
echo "  • Perplexity - Add via Settings → Integrations"
echo "  • Zed Editor - Configure in settings.json"
echo "  • Cody - Configure in VS Code settings"
echo "  • Custom tools - See integration guide"
echo ""
echo "Full manual setup guide:"
echo "  docs/MCP-MANUAL-SETUP.md"
echo "  https://github.com/varun369/SuperLocalMemoryV2/blob/main/docs/MCP-MANUAL-SETUP.md"
echo ""

# Summary
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Installation Complete!                                       ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "✓ Commands are now globally available!"
echo ""
echo "  You can use them immediately:"
echo ""
echo "Available commands (two ways to use them):"
echo ""
echo "OPTION 1: Original commands (still work):"
echo "  superlocalmemoryv2-remember  - Save a new memory"
echo "  superlocalmemoryv2-recall    - Search memories"
echo "  superlocalmemoryv2-list      - List recent memories"
echo "  superlocalmemoryv2-status    - Check system status"
echo ""
echo "OPTION 2: New simple commands:"
echo "  slm remember <content>       - Save (simpler syntax)"
echo "  slm recall <query>           - Search"
echo "  slm list                     - List recent"
echo "  slm status                   - System status"
echo ""
echo "Quick start (try now):"
echo "  slm status"
echo "  slm remember 'My first memory'"
echo "  slm recall 'first'"
echo ""
echo "Learning System (v2.7+):"
echo "  slm learning status              - Check learning system"
echo "  slm engagement                   - View engagement metrics"
echo ""
echo "Lifecycle & Compliance (v2.8+):"
echo "  slm lifecycle-status             - View memory lifecycle states"
echo "  slm compact --dry-run            - Preview lifecycle transitions"
echo "  slm behavioral-patterns          - View learned patterns"
echo ""
# Optional: Offer to install optional features
if [ "$NON_INTERACTIVE" = true ]; then
    INSTALL_CHOICE="N"
    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║  Non-Interactive Mode: Skipping Optional Features            ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo ""
    echo "To install optional features later:"
    echo "  Advanced Search: pip3 install -r ${REPO_DIR}/requirements-search.txt"
    echo "  Web Dashboard:   pip3 install -r ${REPO_DIR}/requirements-ui.txt"
    echo "  Full Package:    pip3 install -r ${REPO_DIR}/requirements-full.txt"
    echo ""
else
    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║  Optional Features Available                                  ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo ""
    echo "Core features already installed:"
    echo "  ✓ Knowledge graph with Leiden clustering"
    echo "  ✓ Pattern learning and identity profiles"
    echo "  ✓ Web dashboard at http://localhost:8000"
    echo ""
    echo "Optional advanced search features:"
    echo ""
    echo "  1) Advanced Semantic Search (~1.5GB, 5-10 min)"
    echo "     • Sentence transformers for better search quality"
    echo "     • Vector similarity with HNSWLIB"
    echo "     • Recommended for large memory databases (>500 items)"
    echo ""
    echo "  N) Skip (install later)"
    echo ""
    echo -n "Choose option [1/N]: "
    read -r INSTALL_CHOICE
fi

case "$INSTALL_CHOICE" in
    1)
        echo ""
        echo "Installing Advanced Search features..."
        echo "⏳ Downloading ~1.5GB (ML models)..."
        if pip3 install $PIP_FLAGS -r "${REPO_DIR}/requirements-search.txt"; then
            echo "✓ Advanced Search installed successfully"
            echo ""
            echo "Search now uses semantic embeddings for better quality!"
        else
            echo "⚠️  Installation failed. Install manually later:"
            echo "  pip3 install -r ${REPO_DIR}/requirements-search.txt"
        fi
        ;;
    [Nn]|*)
        echo ""
        echo "Skipping advanced search."
        echo ""
        echo "To install later:"
        echo "  pip3 install -r ${REPO_DIR}/requirements-search.txt"
        echo ""
        echo "Start Web Dashboard:"
        echo "  python3 ~/.claude-memory/ui_server.py"
        echo "  Then open: http://localhost:8000"
        ;;
esac

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  ATTRIBUTION NOTICE (REQUIRED BY MIT LICENSE)                ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Created by: Varun Pratap Bhardwaj                           ║"
echo "║  Role: Solution Architect & Original Creator                 ║"
echo "║  Repository: github.com/varun369/SuperLocalMemoryV2          ║"
echo "║  License: MIT (attribution must be preserved)                ║"
echo "║                                                              ║"
echo "║  See ATTRIBUTION.md for full attribution requirements        ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
