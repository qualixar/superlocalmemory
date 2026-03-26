@echo off
REM SuperLocalMemory V3 - Windows CLI Wrapper
REM Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
REM Licensed under MIT License
REM Repository: https://github.com/qualixar/superlocalmemory

setlocal enabledelayedexpansion

REM Resolve the package src/ directory for PYTHONPATH
set "SLM_PKG_DIR=%~dp0..\src"

REM Handle --version / -v directly (fast path, no Python needed)
if "%~1"=="--version" goto :show_version
if "%~1"=="-v" goto :show_version

REM Find Python 3
where python3 >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    set PYTHON_CMD=python3
    goto :run
)
where python >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    set PYTHON_CMD=python
    goto :run
)
where py >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    set PYTHON_CMD=py -3
    goto :run
)

echo Error: Python 3.11+ not found.
echo Install from: https://python.org/downloads/
exit /b 1

:show_version
REM Read version from package.json via findstr
for /f "tokens=2 delims=:," %%a in ('findstr /C:"\"version\"" "%~dp0..\package.json"') do (
    set "VER=%%~a"
    set "VER=!VER: =!"
    echo superlocalmemory !VER!
    exit /b 0
)
echo superlocalmemory unknown
exit /b 0

:run
REM Set PYTHONPATH so Python finds the npm package's src/ directory
if defined PYTHONPATH (
    set "PYTHONPATH=%SLM_PKG_DIR%;%PYTHONPATH%"
) else (
    set "PYTHONPATH=%SLM_PKG_DIR%"
)

REM Prevent PyTorch Metal/MPS GPU memory reservation
set "PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0"
set "PYTORCH_MPS_MEM_LIMIT=0"
set "PYTORCH_ENABLE_MPS_FALLBACK=1"
set "TOKENIZERS_PARALLELISM=false"
set "TORCH_DEVICE=cpu"
set "CUDA_VISIBLE_DEVICES="

%PYTHON_CMD% -m superlocalmemory.cli.main %*
exit /b %ERRORLEVEL%
