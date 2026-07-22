@echo off
:: ensure-venv.bat (Codex edition) — SuperLocalMemory plugin venv bootstrap (Windows)
::
:: Idempotent: fast-path exits quickly when requirements.txt is unchanged.
::
:: Difference from plugin/scripts/ensure-venv.bat (Claude Code edition):
::   - Uses SLM_DATA_DIR instead of CLAUDE_PLUGIN_DATA.
::   - Resolves requirements.txt relative to this script's location instead of
::     CLAUDE_PLUGIN_ROOT (which is a Claude Code runtime variable).
::
:: Environment:
::   SLM_DATA_DIR — persistent data dir (venv lives here)
::                  (default: %USERPROFILE%\.superlocalmemory)
::
:: Exit codes: 0 = venv ready   non-0 = failure (logged to stderr)
:: All informational output goes to stderr (stdout reserved for MCP stdio protocol).

setlocal EnableDelayedExpansion

if not defined SLM_DATA_DIR set "SLM_DATA_DIR=%USERPROFILE%\.superlocalmemory"

:: ---------------------------------------------------------------------------
:: Resolve requirements.txt relative to this script's location
:: ---------------------------------------------------------------------------
set "SCRIPT_DIR=%~dp0"
:: Remove trailing backslash
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
set "REQ=%SCRIPT_DIR%\..\requirements.txt"

:: ---------------------------------------------------------------------------
:: Python >= 3.11 guard
:: ---------------------------------------------------------------------------
python -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)" 2>nul
if errorlevel 1 (
    for /f "tokens=*" %%v in ('python --version 2^>^&1') do set PY_VER=%%v
    echo ERROR: SuperLocalMemory plugin requires Python ^>= 3.11, found: !PY_VER! >&2
    echo Install Python 3.11+ and ensure it is first on PATH. >&2
    exit /b 1
)

set "VENV=%SLM_DATA_DIR%\venv"
set "VENV_SCRIPTS=%VENV%\Scripts"
set "SENTINEL=%SLM_DATA_DIR%\.venv-reqs.sha256"
set "VENV_TMP=%SLM_DATA_DIR%\venv.tmp"

:: ---------------------------------------------------------------------------
:: Compute sha256 of requirements.txt using PowerShell
:: ---------------------------------------------------------------------------
for /f "usebackq tokens=*" %%h in (
    `powershell -NoProfile -Command "(Get-FileHash -Algorithm SHA256 '%REQ%').Hash.ToLower()"`
) do set "NEW_HASH=%%h"

if "!NEW_HASH!"=="" (
    echo ERROR: Could not compute sha256 of requirements.txt >&2
    exit /b 1
)

:: ---------------------------------------------------------------------------
:: Fast-path: venv Scripts\python.exe exists AND sentinel matches
:: ---------------------------------------------------------------------------
set "VENV_PYTHON=%VENV_SCRIPTS%\python.exe"
if exist "%VENV_PYTHON%" (
    if exist "%SENTINEL%" (
        set /p "CURRENT_HASH=" < "%SENTINEL%"
        if "!CURRENT_HASH!"=="!NEW_HASH!" (
            echo SLM plugin: venv up-to-date, skipping install. >&2
            exit /b 0
        )
    )
)

:: ---------------------------------------------------------------------------
:: Rebuild venv atomically
:: ---------------------------------------------------------------------------
echo SLM plugin: bootstrapping Python venv at %VENV% ... >&2
echo   requirements: %REQ% >&2

if exist "%VENV_TMP%" rmdir /s /q "%VENV_TMP%"
python -m venv "%VENV_TMP%"
if errorlevel 1 (
    echo ERROR: python -m venv failed >&2
    exit /b 1
)

"%VENV_TMP%\Scripts\pip.exe" install --upgrade pip --prefer-binary --quiet
if errorlevel 1 (
    echo ERROR: pip upgrade failed >&2
    exit /b 1
)

"%VENV_TMP%\Scripts\pip.exe" install --require-virtualenv --prefer-binary --quiet -r "%REQ%"
if errorlevel 1 (
    echo ERROR: pip install requirements failed >&2
    exit /b 1
)

echo SLM plugin: install complete, activating venv. >&2

if exist "%VENV%" rmdir /s /q "%VENV%"
move "%VENV_TMP%" "%VENV%"
if errorlevel 1 (
    echo ERROR: Could not move venv.tmp to venv >&2
    exit /b 1
)

echo !NEW_HASH!>"%SENTINEL%"
echo SLM plugin: venv ready at %VENV%\Scripts\slm.exe >&2
exit /b 0
