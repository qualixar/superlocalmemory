@echo off
REM SuperLocalMemory V2 - Windows CLI Wrapper
REM Copyright (c) 2026 Varun Pratap Bhardwaj
REM Licensed under MIT License
REM Repository: https://github.com/varun369/SuperLocalMemoryV2
REM
REM ATTRIBUTION REQUIRED: This notice must be preserved in all copies.

setlocal enabledelayedexpansion

REM Determine installation location
if exist "%USERPROFILE%\.claude-memory\memory_store_v2.py" (
    set INSTALL_DIR=%USERPROFILE%\.claude-memory
) else if exist "%~dp0..\src\memory_store_v2.py" (
    set INSTALL_DIR=%~dp0..\src
) else (
    echo ERROR: SuperLocalMemory installation not found.
    echo.
    echo Expected locations:
    echo   - %USERPROFILE%\.claude-memory\memory_store_v2.py
    echo   - %~dp0..\src\memory_store_v2.py
    echo.
    echo Run install.ps1 to install SuperLocalMemory V2.
    exit /b 1
)

REM Check Python availability
where python >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    set PYTHON_CMD=python
    goto :python_found
)

where python3 >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    set PYTHON_CMD=python3
    goto :python_found
)

echo ERROR: Python not found in PATH.
echo.
echo Please install Python 3.8 or higher from:
echo   https://www.python.org/downloads/
echo.
echo Make sure to check "Add Python to PATH" during installation.
exit /b 1

:python_found

REM Parse command
if "%1"=="" (
    echo SuperLocalMemory - Universal AI Memory System
    echo.
    echo Usage: slm [command] [options]
    echo.
    echo Commands:
    echo   remember [content]     Store a new memory
    echo   recall [query]         Search memories
    echo   list-recent [N]        List recent memories
    echo   status                 Show system status
    echo   build-graph            Rebuild knowledge graph
    echo   switch-profile [name]  Switch memory profile
    echo   help                   Show this help
    echo.
    echo Examples:
    echo   slm remember "React hooks for state management" --tags frontend,react
    echo   slm recall "authentication patterns"
    echo   slm list-recent 20
    echo   slm status
    echo.
    echo Documentation: https://github.com/varun369/SuperLocalMemoryV2/wiki
    exit /b 0
)

REM Handle special commands
if /i "%1"=="help" goto :show_help
if /i "%1"=="--help" goto :show_help
if /i "%1"=="-h" goto :show_help
if /i "%1"=="version" goto :show_version
if /i "%1"=="--version" goto :show_version
if /i "%1"=="-v" goto :show_version

REM Route commands to appropriate Python module
if /i "%1"=="remember" goto :remember
if /i "%1"=="recall" goto :recall
if /i "%1"=="list-recent" goto :list_recent
if /i "%1"=="list" goto :list_recent
if /i "%1"=="status" goto :status
if /i "%1"=="stats" goto :status
if /i "%1"=="build-graph" goto :build_graph
if /i "%1"=="switch-profile" goto :switch_profile
if /i "%1"=="profile" goto :switch_profile

echo ERROR: Unknown command: %1
echo.
echo Run "slm help" for usage information.
exit /b 1

:remember
shift
%PYTHON_CMD% "%INSTALL_DIR%\memory_store_v2.py" add %*
exit /b %ERRORLEVEL%

:recall
shift
%PYTHON_CMD% "%INSTALL_DIR%\memory_store_v2.py" search %*
exit /b %ERRORLEVEL%

:list_recent
shift
if "%1"=="" (
    %PYTHON_CMD% "%INSTALL_DIR%\memory_store_v2.py" list 20
) else (
    %PYTHON_CMD% "%INSTALL_DIR%\memory_store_v2.py" list %*
)
exit /b %ERRORLEVEL%

:status
%PYTHON_CMD% "%INSTALL_DIR%\memory_store_v2.py" stats
exit /b %ERRORLEVEL%

:build_graph
if exist "%INSTALL_DIR%\graph_engine.py" (
    %PYTHON_CMD% "%INSTALL_DIR%\graph_engine.py" build
) else (
    echo ERROR: graph_engine.py not found.
    exit /b 1
)
exit /b %ERRORLEVEL%

:switch_profile
shift
if exist "%INSTALL_DIR%\memory-profiles.py" (
    if "%1"=="" (
        %PYTHON_CMD% "%INSTALL_DIR%\memory-profiles.py" list
    ) else if "%1"=="--list" (
        %PYTHON_CMD% "%INSTALL_DIR%\memory-profiles.py" list
    ) else (
        %PYTHON_CMD% "%INSTALL_DIR%\memory-profiles.py" switch %*
    )
) else (
    echo ERROR: memory-profiles.py not found.
    exit /b 1
)
exit /b %ERRORLEVEL%

:show_help
echo SuperLocalMemory - Universal AI Memory System
echo.
echo Usage: slm [command] [options]
echo.
echo Commands:
echo   remember [content]              Store a new memory
echo     Options:
echo       --tags TAG1,TAG2            Add tags
echo       --project NAME              Set project
echo       --importance N              Set importance (1-10)
echo.
echo   recall [query]                  Search memories
echo     Options:
echo       --limit N                   Limit results (default: 10)
echo       --tags TAG1,TAG2            Filter by tags
echo       --project NAME              Filter by project
echo.
echo   list-recent [N]                 List N most recent memories
echo   status                          Show system statistics
echo   build-graph                     Rebuild knowledge graph
echo   switch-profile [name]           Switch memory profile
echo     Options:
echo       --list                      List all profiles
echo.
echo   help                            Show this help
echo   version                         Show version
echo.
echo Examples:
echo   slm remember "React hooks for state management" --tags frontend,react
echo   slm recall "authentication patterns" --limit 5
echo   slm list-recent 20
echo   slm status
echo   slm build-graph
echo   slm switch-profile work
echo.
echo Installation: %INSTALL_DIR%
echo Database: %USERPROFILE%\.claude-memory\memory.db
echo.
echo Documentation: https://github.com/varun369/SuperLocalMemoryV2/wiki
echo Report issues: https://github.com/varun369/SuperLocalMemoryV2/issues
exit /b 0

:show_version
echo SuperLocalMemory V2.8.3
echo Copyright (c) 2026 Varun Pratap Bhardwaj
echo Licensed under MIT License
echo Repository: https://github.com/varun369/SuperLocalMemoryV2
exit /b 0
