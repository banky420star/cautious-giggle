# Grok AGI - VPS Smoke Test
# Run this BEFORE starting live trading or training.

$ErrorActionPreference = "Stop"

Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "Grok AGI Bot - Pre-Flight Smoke Test" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan

# 1. Check for Virtual Environment
if (-Not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "[ERROR] Virtual environment not found. Run scripts\setup_vps.bat first." -ForegroundColor Red
    exit 1
}

# 2. Syntax Check (Compile all Python files)
Write-Host "[INFO] Checking Python syntax across all files..." -ForegroundColor Yellow
& .\.venv\Scripts\python.exe -m py_compile Python\*.py
& .\.venv\Scripts\python.exe -m py_compile training\*.py
& .\.venv\Scripts\python.exe -m py_compile drl\*.py
Write-Host "[OK] Syntax checks passed." -ForegroundColor Green

# 3. Core Component Imports Check
Write-Host "[INFO] Validating core imports..." -ForegroundColor Yellow
$modules = @(
    "Python.Server_AGI",
    "Python.mt5_executor",
    "Python.risk_engine",
    "Python.hybrid_brain",
    "drl.trading_env"
)

foreach ($mod in $modules) {
    Try {
        $output = & .\.venv\Scripts\python.exe -c "import $mod; print('OK')" 2>&1
        if ($output -match "OK") {
            Write-Host "  -> $mod loaded successfully" -ForegroundColor Green
        }
        else {
            Throw $output
        }
    }
    Catch {
        Write-Host "  -> [ERROR] Failed to import $mod" -ForegroundColor Red
        Write-Host $_ -ForegroundColor Red
        exit 1
    }
}

# 4. Config Validation
Write-Host "[INFO] Checking configuration keys..." -ForegroundColor Yellow
Try {
    $script = @"
import yaml, sys
try:
    with open('config.yaml') as f:
        cfg = yaml.safe_load(f)
    trading = cfg.get('trading', {})
    if not trading:
        print('Missing trading config')
        sys.exit(1)
except Exception as e:
    print(f'Config error: {e}')
    sys.exit(1)
print('OK')
"@
    $out = & .\.venv\Scripts\python.exe -c $script 2>&1
    if ($out -match "OK") {
        Write-Host "  -> config.yaml parsed successfully" -ForegroundColor Green
    }
    else {
        Throw $out
    }
}
Catch {
    Write-Host "  -> [ERROR] Invalid or missing config.yaml" -ForegroundColor Red
    exit 1
}

# 5. Check Log Directories
if (-Not (Test-Path "logs")) {
    Write-Host "[WARNING] Logs directory missing. Did you run setup_vps.bat?" -ForegroundColor Yellow
}
if (-Not (Test-Path "models")) {
    Write-Host "[WARNING] Models directory missing." -ForegroundColor Yellow
}

Write-Host "`n==================================================" -ForegroundColor Cyan
Write-Host "ALL SMOKE TESTS PASSED! SERVER IS READY TO FLY." -ForegroundColor Green
Write-Host "==================================================" -ForegroundColor Cyan
exit 0
