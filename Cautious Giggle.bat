@echo off
title Cautious Giggle
cd /d "%~dp0"
python launcher.py
if errorlevel 1 (
    echo.
    echo Failed to start. Make sure Python and dependencies are installed.
    pause
)
