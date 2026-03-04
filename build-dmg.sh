#!/usr/bin/env bash
#
# SuperLocalMemory V2 - DMG Installer Build Script
#
# Copyright (c) 2026 Varun Pratap Bhardwaj
# Solution Architect & Original Creator
#
# Licensed under MIT License (see LICENSE file)
# Repository: https://github.com/varun369/SuperLocalMemoryV2
#
# ATTRIBUTION REQUIRED: This notice must be preserved in all copies.
#

set -e  # Exit on any error

# ============================================================================
# CONFIGURATION
# ============================================================================

VERSION="2.7.4"
APP_NAME="SuperLocalMemory"
DMG_NAME="SuperLocalMemory-v${VERSION}-macos"
BUILD_DIR="build/dmg"
INSTALLER_DIR="${BUILD_DIR}/${APP_NAME}"
TEMP_DMG="${BUILD_DIR}/temp.dmg"
FINAL_DMG="${DMG_NAME}.dmg"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ============================================================================
# FUNCTIONS
# ============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

cleanup_build() {
    log_info "Cleaning up previous build artifacts..."
    rm -rf "${BUILD_DIR}"
    rm -f "${FINAL_DMG}"
}

create_directory_structure() {
    log_info "Creating directory structure..."
    mkdir -p "${INSTALLER_DIR}"
}

copy_installer_files() {
    log_info "Copying installer files..."
    
    # Core installation script
    cp install.sh "${INSTALLER_DIR}/"
    
    # Source files
    log_info "  → Copying src/ directory..."
    cp -r src "${INSTALLER_DIR}/"
    
    # Binary files
    log_info "  → Copying bin/ directory..."
    cp -r bin "${INSTALLER_DIR}/"
    
    # MCP configurations
    log_info "  → Copying configs/ directory..."
    cp -r configs "${INSTALLER_DIR}/"
    
    # Universal skills
    log_info "  → Copying skills/ directory..."
    cp -r skills "${INSTALLER_DIR}/"
    
    # MCP server
    log_info "  → Copying mcp_server.py..."
    cp mcp_server.py "${INSTALLER_DIR}/"
    
    # Skills installer
    log_info "  → Copying install-skills.sh..."
    cp install-skills.sh "${INSTALLER_DIR}/"
    
    # Requirements files
    log_info "  → Copying requirements files..."
    cp requirements.txt "${INSTALLER_DIR}/"
    cp requirements-core.txt "${INSTALLER_DIR}/"
    cp requirements-full.txt "${INSTALLER_DIR}/"
    cp requirements-search.txt "${INSTALLER_DIR}/"
    cp requirements-ui.txt "${INSTALLER_DIR}/"
    cp requirements-learning.txt "${INSTALLER_DIR}/" 2>/dev/null || true
    
    # UI server
    log_info "  → Copying ui_server.py..."
    cp ui_server.py "${INSTALLER_DIR}/"
    
    # Start scripts
    log_info "  → Copying start scripts..."
    cp start-dashboard.sh "${INSTALLER_DIR}/"
    
    # Verification script
    log_info "  → Copying verify-install.sh..."
    cp verify-install.sh "${INSTALLER_DIR}/"
    
    # Documentation
    log_info "  → Copying documentation..."
    cp README.md "${INSTALLER_DIR}/" 2>/dev/null || true
    cp LICENSE "${INSTALLER_DIR}/" 2>/dev/null || true
    cp ATTRIBUTION.md "${INSTALLER_DIR}/" 2>/dev/null || true
    cp CHANGELOG.md "${INSTALLER_DIR}/" 2>/dev/null || true
    if [ -d "docs" ]; then
        cp -r docs "${INSTALLER_DIR}/"
    fi
    
    # Shell completions
    if [ -d "completions" ]; then
        log_info "  → Copying shell completions..."
        cp -r completions "${INSTALLER_DIR}/"
    fi
}

create_readme() {
    log_info "Creating README-INSTALLATION.txt..."
    
    cat > "${INSTALLER_DIR}/README-INSTALLATION.txt" << 'READMEEOF'
================================================================================
  SuperLocalMemory V2.7.0 - macOS Installation
================================================================================

Thank you for downloading SuperLocalMemory V2!

QUICK START (2 minutes):
------------------------

1. Open Terminal (Applications → Utilities → Terminal)

2. Navigate to this folder:
   cd /Volumes/SuperLocalMemory

3. Run the installer:
   ./install.sh

4. Follow the on-screen prompts

5. Restart your AI tools (Claude Desktop, Cursor, etc.)

WHAT GETS INSTALLED:
-------------------
• Core memory system → ~/.claude-memory/
• MCP server auto-configured for 11+ IDEs
• 6 universal skills (Claude Code, Continue.dev, Cody, etc.)
• CLI command: slm
• Dashboard server: start-dashboard.sh

VERIFICATION:
------------
After installation, verify by running:
  slm status

NEED HELP?
---------
• Documentation: README.md
• Quick Start: docs/CLI-COMMANDS-REFERENCE.md
• GitHub Wiki: https://github.com/varun369/SuperLocalMemoryV2/wiki
• Issues: https://github.com/varun369/SuperLocalMemoryV2/issues

UNINSTALL:
---------
To uninstall:
  rm -rf ~/.claude-memory
  rm -f /usr/local/bin/slm

================================================================================
Created by: Varun Pratap Bhardwaj
License: MIT (Attribution Required)
Repository: https://github.com/varun369/SuperLocalMemoryV2
================================================================================
READMEEOF
}

create_installer_wrapper() {
    log_info "Creating installer wrapper script..."
    
    cat > "${INSTALLER_DIR}/INSTALL" << 'WRAPPEREOF'
#!/usr/bin/env bash
#
# SuperLocalMemory V2 - Installer Wrapper
# Double-click this file to install, or run from Terminal
#

# Detect if running from Finder (double-click)
if [ -t 0 ]; then
    # Running in terminal
    TERMINAL_MODE=true
else
    # Running from Finder
    TERMINAL_MODE=false
    # Open Terminal and run installer
    osascript <<APPLESCRIPT
tell application "Terminal"
    activate
    do script "cd \"$(dirname \"$0\")\" && ./install.sh && echo '\\n\\nPress Enter to close...' && read"
end tell
APPLESCRIPT
    exit 0
fi

# If in terminal mode, just run installer
cd "$(dirname "$0")"
./install.sh

echo ""
echo "Installation complete!"
echo "Press Enter to close..."
read
WRAPPEREOF

    chmod +x "${INSTALLER_DIR}/INSTALL"
}

create_dmg_background() {
    log_info "Creating DMG background image..."
    
    # Create a simple background with instructions
    # (This uses ImageMagick if available, otherwise creates a simple text file)
    
    if command -v convert &> /dev/null; then
        log_info "  → Using ImageMagick to create background..."
        convert -size 640x480 xc:white \
            -font Helvetica -pointsize 20 \
            -fill black -gravity north \
            -annotate +0+50 "SuperLocalMemory" \
            -pointsize 14 -gravity center \
            -annotate +0+0 "Drag to Applications folder\nor double-click INSTALL to begin" \
            "${BUILD_DIR}/.background.png" 2>/dev/null || true
    else
        log_warning "  → ImageMagick not found, skipping background image"
    fi
}

create_symlink() {
    log_info "Creating Applications folder symlink..."
    ln -s /Applications "${BUILD_DIR}/Applications"
}

create_dmg() {
    log_info "Creating temporary DMG..."
    
    # Calculate size needed (add 50MB buffer)
    SIZE=$(du -sm "${INSTALLER_DIR}" | awk '{print $1}')
    SIZE=$((SIZE + 50))
    
    log_info "  → Allocating ${SIZE}MB for DMG..."
    
    # Create writable DMG
    hdiutil create \
        -volname "${APP_NAME}" \
        -srcfolder "${BUILD_DIR}" \
        -ov \
        -format UDRW \
        -size ${SIZE}m \
        "${TEMP_DMG}"
    
    log_info "Mounting temporary DMG..."
    DEVICE=$(hdiutil attach -readwrite -noverify -noautoopen "${TEMP_DMG}" | \
             grep -E '^/dev/' | sed 1q | awk '{print $1}')
    
    log_info "  → Mounted at: ${DEVICE}"
    
    # Wait for mount
    sleep 2
    
    MOUNT_POINT="/Volumes/${APP_NAME}"
    
    # Set custom icon positions (if possible)
    log_info "Configuring Finder view settings..."
    
    osascript <<APPLESCRIPT || log_warning "Could not set Finder view settings"
tell application "Finder"
    tell disk "${APP_NAME}"
        open
        set current view of container window to icon view
        set toolbar visible of container window to false
        set statusbar visible of container window to false
        set the bounds of container window to {100, 100, 740, 580}
        set theViewOptions to the icon view options of container window
        set arrangement of theViewOptions to not arranged
        set icon size of theViewOptions to 128
        set background picture of theViewOptions to file ".background.png"
        
        -- Position installer folder
        set position of item "${APP_NAME}" of container window to {160, 240}
        
        -- Position Applications symlink
        set position of item "Applications" of container window to {480, 240}
        
        -- Position README
        if exists item "README-INSTALLATION.txt" then
            set position of item "README-INSTALLATION.txt" of container window to {320, 400}
        end if
        
        update without registering applications
        delay 2
        close
    end tell
end tell
APPLESCRIPT
    
    # Ensure everything is written
    sync
    sleep 2
    
    log_info "Unmounting temporary DMG..."
    hdiutil detach "${DEVICE}" -quiet || {
        log_warning "Normal unmount failed, forcing..."
        hdiutil detach "${DEVICE}" -force -quiet
    }
    
    log_info "Converting to compressed read-only DMG..."
    hdiutil convert "${TEMP_DMG}" \
        -format UDZO \
        -imagekey zlib-level=9 \
        -o "${FINAL_DMG}"
    
    # Clean up temp DMG
    rm -f "${TEMP_DMG}"
}

verify_dmg() {
    log_info "Verifying DMG..."
    
    if [ ! -f "${FINAL_DMG}" ]; then
        log_error "DMG file not found: ${FINAL_DMG}"
        return 1
    fi
    
    # Check DMG integrity
    hdiutil verify "${FINAL_DMG}" && log_success "DMG verification passed"
    
    # Show file size
    SIZE=$(du -h "${FINAL_DMG}" | awk '{print $1}')
    log_info "DMG size: ${SIZE}"
}

show_summary() {
    echo ""
    echo "============================================================================"
    log_success "DMG BUILD COMPLETE!"
    echo "============================================================================"
    echo ""
    echo "  Output file: ${FINAL_DMG}"
    echo "  Size: $(du -h "${FINAL_DMG}" | awk '{print $1}')"
    echo ""
    echo "NEXT STEPS:"
    echo "  1. Test the DMG:"
    echo "     open ${FINAL_DMG}"
    echo ""
    echo "  2. Upload to GitHub Releases:"
    echo "     gh release create v${VERSION} ${FINAL_DMG} \\"
    echo "       --title \"SuperLocalMemory v${VERSION}\" \\"
    echo "       --notes \"See CHANGELOG.md for details\""
    echo ""
    echo "  3. Or upload manually:"
    echo "     https://github.com/varun369/SuperLocalMemoryV2/releases/new"
    echo ""
    echo "============================================================================"
}

# ============================================================================
# MAIN EXECUTION
# ============================================================================

main() {
    log_info "Starting DMG build for SuperLocalMemory v${VERSION}..."
    echo ""
    
    # Check we're in the right directory
    if [ ! -f "install.sh" ]; then
        log_error "Must run from SuperLocalMemoryV2 repository root"
        exit 1
    fi
    
    # Check required tools
    if ! command -v hdiutil &> /dev/null; then
        log_error "hdiutil not found (required for macOS DMG creation)"
        exit 1
    fi
    
    # Build process
    cleanup_build
    create_directory_structure
    copy_installer_files
    create_readme
    create_installer_wrapper
    create_dmg_background
    create_symlink
    create_dmg
    verify_dmg
    show_summary
    
    log_success "Build complete!"
}

# Run main function
main "$@"
