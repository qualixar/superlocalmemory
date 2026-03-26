@echo off
REM SuperLocalMemory V3 - Windows CLI (CMD variant)
REM Delegates to slm.bat which handles PYTHONPATH and --version
call "%~dp0slm.bat" %*
exit /b %ERRORLEVEL%
