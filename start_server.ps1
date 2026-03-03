$env:AGI_HOST = "0.0.0.0"
$env:AGI_PORT = "9090"

if (-not $env:AGI_TOKEN) {
    Write-Host "[WARNING] AGI_TOKEN is not set. Server will accept unsigned requests." -ForegroundColor Yellow
}

$env:AGI_AUTONOMY_AUTO_CANARY = "true"
$env:AGI_PNL_POLL = "true"

$env:AGI_COOLDOWN_SEC = "45"
$env:AGI_MIN_HOLD_SEC = "120"
$env:CANARY_LOT_MULT = "0.25"

$env:AGI_DZ_EURUSD = "0.18"
$env:AGI_DZ_GBPUSD = "0.20"
$env:AGI_DZ_XAUUSD = "0.22"

if (-not $env:AGI_USE_LIMIT_ORDERS) { $env:AGI_USE_LIMIT_ORDERS = "false" }
if (-not $env:AGI_LIMIT_OFFSET_POINTS) { $env:AGI_LIMIT_OFFSET_POINTS = "30" }
if (-not $env:AGI_PPO_DEADZONE) { $env:AGI_PPO_DEADZONE = "0.08" }
if (-not $env:AGI_AUTO_DYNAMIC_ENTRY) { $env:AGI_AUTO_DYNAMIC_ENTRY = "true" }
if (-not $env:AGI_AUTONOMY_TRAIN_LSTM) { $env:AGI_AUTONOMY_TRAIN_LSTM = "true" }
if (-not $env:AGI_AUTONOMY_TRAIN_PPO) { $env:AGI_AUTONOMY_TRAIN_PPO = "true" }

Write-Host "Starting Grok AGI Server on Port $env:AGI_PORT ..."
if (Test-Path ".\.venv\Scripts\Activate.ps1") {
    .\.venv\Scripts\Activate.ps1
}
python -m Python.Server_AGI --live
