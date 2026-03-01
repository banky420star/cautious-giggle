@echo off
echo ==================================================
echo Grok AGI Bot - Windows VPS Setup Script
echo ==================================================
echo.

:: 1. Create Virtual Environment if it doesn't exist
if not exist ".venv\Scripts\activate.bat" (
    echo [INFO] Creating Python virtual environment...
    python -m venv .venv
) else (
    echo [INFO] Virtual environment already exists.
)

:: 2. Activate VENV
echo [INFO] Activating virtual environment...
call .venv\Scripts\activate.bat

:: 3. Upgrade pip and setuptools (fix for pkg_resources TensorBoard bug)
echo [INFO] Upgrading pip, setuptools (pinned below v82), and wheel...
python -m pip install --upgrade pip
python -m pip install --upgrade "setuptools<82" wheel

:: 4. Verify pkg_resources
echo [INFO] Verifying pkg_resources...
python -c "import pkg_resources; print('pkg_resources OK')"
if %ERRORLEVEL% neq 0 (
    echo [ERROR] pkg_resources check failed. Setup is unstable.
    exit /b 1
)

:: 5. Install dependencies
echo [INFO] Installing project requirements...
pip install -r requirements.txt

:: 6. Create required log/model directories
echo [INFO] Creating required local directories...
if not exist "logs" mkdir logs
if not exist "models" mkdir models
if not exist "data" mkdir data
if not exist "runs" mkdir runs

echo.
echo ==================================================
echo SETUP COMPLETE!
echo Activate the environment anytime using: .\.venv\Scripts\activate
echo Run the smoke test next: powershell -ExecutionPolicy Bypass -File scripts\smoke_test.ps1
echo ==================================================
pause
