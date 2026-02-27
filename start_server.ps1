$env:AGI_HOST="0.0.0.0"
$env:AGI_PORT="9090"
$env:AGI_TOKEN="fuckyou2/"

$env:AGI_AUTONOMY_AUTO_CANARY="true"
$env:AGI_PNL_POLL="true"

$env:AGI_COOLDOWN_SEC="45"
$env:AGI_MIN_HOLD_SEC="120"
$env:CANARY_LOT_MULT="0.25"

$env:AGI_DZ_EURUSD="0.18"
$env:AGI_DZ_GBPUSD="0.20"
$env:AGI_DZ_XAUUSD="0.22"

Write-Host "Starting Grok AGI Server on Port 9090 with Token 'fuckyou2/'..."
python -m Python.Server_AGI --live
