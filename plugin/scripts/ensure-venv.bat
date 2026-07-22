@echo off
:: ensure-venv.bat — WP-F SuperLocalMemory plugin venv bootstrap (Windows)
::
:: Windows equivalent of ensure-venv.sh. Called by Claude Code SessionStart hook
:: on Windows hosts. Idempotent: fast-path exits quickly when requirements.txt
:: is unchanged (sentinel file guards reinstall).
::
:: Environment (set by Claude Code plugin runtime):
::   CLAUDE_PLUGIN_ROOT  — plugin installation dir (ephemeral, contains scripts\ + requirements.txt)
::   CLAUDE_PLUGIN_DATA  — persistent data dir (venv lives here, survives plugin updates)
::
:: Exit codes: 0 = venv ready   non-0 = failure (logged to stderr)
:: All informational output goes to stderr (stdout reserved for MCP stdio protocol).
::
:: Requires: Python 3.11+ on PATH, pip, standard Windows cmd.exe

setlocal EnableDelayedExpansion

:: ---------------------------------------------------------------------------
:: Guard: required env vars must be set
:: ---------------------------------------------------------------------------
if not defined CLAUDE_PLUGIN_ROOT (
    echo ERROR: CLAUDE_PLUGIN_ROOT must be set ^(plugin installation directory^) >&2
    exit /b 1
)
if not defined CLAUDE_PLUGIN_DATA (
    echo ERROR: CLAUDE_PLUGIN_DATA must be set ^(plugin persistent data directory^) >&2
    exit /b 1
)

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

:: ---------------------------------------------------------------------------
:: Paths
:: ---------------------------------------------------------------------------
set "REQ=%CLAUDE_PLUGIN_ROOT%\requirements.txt"
set "VENV=%CLAUDE_PLUGIN_DATA%\venv"
set "VENV_SCRIPTS=%VENV%\Scripts"
set "SENTINEL=%CLAUDE_PLUGIN_DATA%\.venv-reqs.sha256"
set "VENV_TMP=%CLAUDE_PLUGIN_DATA%\venv.tmp"

:: ---------------------------------------------------------------------------
:: Compute sha256 of requirements.txt using PowerShell (built-in on Windows 10+)
:: Output: hex string of SHA256 hash
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
:: Rebuild venv atomically: install to venv.tmp, rename to venv, write sentinel LAST
:: ---------------------------------------------------------------------------
echo SLM plugin: bootstrapping Python venv at %VENV% ... >&2
echo   requirements: %REQ% >&2

:: Clean up any partial previous attempt
if exist "%VENV_TMP%" rmdir /s /q "%VENV_TMP%"

:: Create fresh venv
python -m venv "%VENV_TMP%"
if errorlevel 1 (
    echo ERROR: python -m venv failed >&2
    exit /b 1
)

:: Upgrade pip first
"%VENV_TMP%\Scripts\pip.exe" install --upgrade pip --prefer-binary --quiet
if errorlevel 1 (
    echo ERROR: pip upgrade failed >&2
    exit /b 1
)

:: Install requirements
"%VENV_TMP%\Scripts\pip.exe" install --require-virtualenv --prefer-binary --quiet -r "%REQ%"
if errorlevel 1 (
    echo ERROR: pip install requirements failed >&2
    exit /b 1
)

echo SLM plugin: install complete, activating venv. >&2

:: Atomic swap: remove old venv (if any), rename tmp into place
if exist "%VENV%" rmdir /s /q "%VENV%"
move "%VENV_TMP%" "%VENV%"
if errorlevel 1 (
    echo ERROR: Could not move venv.tmp to venv >&2
    exit /b 1
)

:: Write sentinel LAST — guarantees that a crash before this line triggers rebuild
echo !NEW_HASH!>"%SENTINEL%"

echo SLM plugin: venv ready at %VENV%\Scripts\slm.exe >&2
exit /b 0
