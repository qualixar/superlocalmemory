@echo off
:: slm-launch.bat — WP-F SuperLocalMemory MCP launcher (Windows)
::
:: Cross-platform counterpart: slm-launch (POSIX bash)
:: Referenced by plugin/.mcp.json as the MCP server command (Windows picks .bat automatically).
::
:: Resolves the correct venv binary for Windows:
::   %CLAUDE_PLUGIN_DATA%\venv\Scripts\slm.exe mcp
::
:: On Windows, Python venv places entry points in Scripts\ (not bin\ like POSIX).
:: This launcher bridges the path difference so ONE .mcp.json command field works
:: cross-platform.
::
:: Environment:
::   CLAUDE_PLUGIN_DATA — persistent data dir where venv lives

"%CLAUDE_PLUGIN_DATA%\venv\Scripts\slm.exe" mcp
