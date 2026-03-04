# ============================================================================
# SuperLocalMemory - Dashboard Startup Script (PowerShell)
# Starts the web dashboard on http://localhost:8765
# Copyright (c) 2026 Varun Pratap Bhardwaj
# ============================================================================

$ErrorActionPreference = "Stop"

$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $SCRIPT_DIR

Write-Host "=================================================================="
Write-Host "  SuperLocalMemory - Dashboard                            "
Write-Host "  by Varun Pratap Bhardwaj                                       "
Write-Host "=================================================================="
Write-Host ""

# Check if Python is available
try {
    $pythonVersion = & python --version 2>&1
    Write-Host "Python: $pythonVersion"
} catch {
    Write-Host "ERROR: Python not found in PATH" -ForegroundColor Red
    Write-Host "Install Python 3.8+ from https://www.python.org/downloads/" -ForegroundColor Yellow
    exit 1
}

# Check if FastAPI is installed
try {
    & python -c "import fastapi" 2>$null
} catch {
    Write-Host "WARNING: FastAPI not installed (optional dependency)" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "To use the dashboard, install web dependencies:"
    Write-Host "  pip install -r requirements-ui.txt"
    Write-Host ""
    Write-Host "Or install all features:"
    Write-Host "  pip install -r requirements-full.txt"
    Write-Host ""
    exit 1
}

Write-Host "Starting dashboard server..." -ForegroundColor Green
Write-Host ""
Write-Host "   Dashboard: http://localhost:8765"
Write-Host "   API Docs:  http://localhost:8765/docs"
Write-Host ""
Write-Host "Press Ctrl+C to stop"
Write-Host ""

# Start server
& python ui_server.py
