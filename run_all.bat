@echo off
title AGI Trading System - Launcher
color 0B
echo.
echo  ============================================
echo   AGI Trading System - Full Stack Launcher
echo  ============================================
echo.
echo  [1] Backend  : Python Server_AGI (port 5000)
echo  [2] Frontend : Vite React App   (port 5173)
echo  [3] Socket   : MT5 Bridge       (port 9090)
echo.

cd /d "C:\Users\Administrator\work\cautious-giggle-clone-20260320161357"

:: Kill any existing instances
echo  Cleaning up old processes...
taskkill /F /FI "WINDOWTITLE eq AGI-Backend*" >nul 2>&1
taskkill /F /FI "WINDOWTITLE eq AGI-Frontend*" >nul 2>&1
for /f "tokens=2" %%a in ('wmic process where "commandline like '%%Server_AGI%%' and name='python.exe'" get processid 2^>nul ^| findstr /r "[0-9]"') do (
    taskkill /F /PID %%a >nul 2>&1
)
timeout /t 2 /nobreak >nul

:: Start Backend
echo  Starting Backend Server...
start "AGI-Backend" /min cmd /k "cd /d C:\Users\Administrator\work\cautious-giggle-clone-20260320161357 && .venv312\Scripts\python.exe -m Python.Server_AGI --live"

:: Wait for backend to initialize
echo  Waiting for backend to load models (60s)...
timeout /t 60 /nobreak >nul

:: Start Frontend
echo  Starting Frontend Dev Server...
start "AGI-Frontend" /min cmd /k "cd /d C:\Users\Administrator\work\cautious-giggle-clone-20260320161357\frontend && npx vite --host 0.0.0.0"

:: Wait for Vite
timeout /t 10 /nobreak >nul

echo.
echo  ============================================
echo   All systems launched!
echo  ============================================
echo.
echo   Backend API : http://localhost:5000/api/status
echo   Frontend UI : http://localhost:5173/app/
echo   External UI : http://194.163.171.68:5173/app/
echo.
echo   Press any key to open the dashboard...
pause >nul

start "" "http://localhost:5173/app/"
