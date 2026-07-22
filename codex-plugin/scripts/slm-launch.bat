@echo off
:: slm-launch.bat (Codex edition) — SuperLocalMemory MCP launcher (Windows)
::
:: Cross-platform counterpart: slm-launch (POSIX bash)
:: Referenced by codex-plugin/.codex/config.toml as the advanced MCP server command.
::
:: Resolves the correct venv binary for Windows and joins the namespace daemon before
:: opening MCP, preserving one writer for parallel Codex sessions.
::
:: Difference from plugin/scripts/slm-launch.bat (Claude Code edition):
::   - Uses SLM_DATA_DIR (default: %USERPROFILE%\.superlocalmemory) instead of
::     CLAUDE_PLUGIN_DATA.
::   - Falls back to the PATH `slm` binary if the venv binary is not present.
::
:: Environment:
::   SLM_DATA_DIR — persistent data dir where venv lives
::                  (default: %USERPROFILE%\.superlocalmemory)

if not defined SLM_DATA_DIR set "SLM_DATA_DIR=%USERPROFILE%\.superlocalmemory"
set "SLM_BIN=%SLM_DATA_DIR%\venv\Scripts\slm.exe"

if not exist "%SLM_BIN%" set "SLM_BIN=slm"

"%SLM_BIN%" serve start 1>&2
if errorlevel 1 (
    echo SLM plugin: unable to start the owned daemon; refusing a direct MCP writer. 1>&2
    exit /b 1
)

"%SLM_BIN%" mcp
