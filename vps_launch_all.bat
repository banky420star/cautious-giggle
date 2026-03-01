@echo off
echo =========================================================
echo  GROK AGI VPS LAUNCHER (Multi-Window)
echo =========================================================
echo.

:: Ensure we are starting from the bot's root directory
cd /d "%~dp0"

echo [1] Launching Python AGI Server in Window 1...
start powershell -NoExit -Command "$Host.UI.RawUI.WindowTitle = 'AGI Server'; .\start_server.ps1"

echo.
echo [2] Launching n8n Orchestrator in Window 2...
start powershell -NoExit -Command "$Host.UI.RawUI.WindowTitle = 'n8n Orchestrator'; $env:NODES_EXCLUDE='[]'; n8n start"

echo.
echo =========================================================
echo ALL SYSTEMS GO!
echo Two new PowerShell windows have been opened.
echo - Window 1: AGI Server (Wait for "Server running on 0.0.0.0:9090")
echo - Window 2: n8n (Wait for "n8n ready on ::, port 5678")
echo.
echo IMPORTANT: Make sure your MetaTrader 5 Terminal is open!
echo =========================================================
pause
