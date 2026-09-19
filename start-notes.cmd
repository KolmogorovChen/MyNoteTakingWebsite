@echo off
setlocal
cd /d "%~dp0"
set "NOTES_NODE=node"
where node >nul 2>nul
if errorlevel 1 set "NOTES_NODE=%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe"
echo Notes website: http://127.0.0.1:5173
echo Keep this window open. Press Ctrl+C to stop.
"%NOTES_NODE%" scripts\site.mjs dev
pause
