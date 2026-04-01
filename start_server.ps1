function Import-EnvFile {
    param([string]$Path)

    if (-not (Test-Path $Path)) {
        return
    }

    Get-Content $Path | ForEach-Object {
        $line = $_.Trim()
        if (-not $line -or $line.StartsWith("#")) {
            return
        }

        $parts = $line -split "=", 2
        if ($parts.Count -ne 2) {
            return
        }

        $name = $parts[0].Trim()
        $value = $parts[1].Trim()
        if (-not $name) {
            return
        }

        if (-not [System.Environment]::GetEnvironmentVariable($name, "Process")) {
            [System.Environment]::SetEnvironmentVariable($name, $value, "Process")
        }
    }
}

$root = $PSScriptRoot
$envFile = Join-Path $root ".env"
Import-EnvFile -Path $envFile

if (-not $env:AGI_TOKEN) {
    throw "AGI_TOKEN must be set in the environment or in $envFile. See .env.example for the required keys."
}

$env:AGI_HOST = if ($env:AGI_HOST) { $env:AGI_HOST } else { "0.0.0.0" }
$env:AGI_PORT = if ($env:AGI_PORT) { $env:AGI_PORT } else { "9090" }
$env:AGI_AUTONOMY_AUTO_CANARY = if ($env:AGI_AUTONOMY_AUTO_CANARY) { $env:AGI_AUTONOMY_AUTO_CANARY } else { "true" }
$env:AGI_PNL_POLL = if ($env:AGI_PNL_POLL) { $env:AGI_PNL_POLL } else { "true" }
$env:AGI_COOLDOWN_SEC = if ($env:AGI_COOLDOWN_SEC) { $env:AGI_COOLDOWN_SEC } else { "45" }
$env:AGI_MIN_HOLD_SEC = if ($env:AGI_MIN_HOLD_SEC) { $env:AGI_MIN_HOLD_SEC } else { "120" }
$env:CANARY_LOT_MULT = if ($env:CANARY_LOT_MULT) { $env:CANARY_LOT_MULT } else { "0.25" }
$env:AGI_DZ_EURUSD = if ($env:AGI_DZ_EURUSD) { $env:AGI_DZ_EURUSD } else { "0.18" }
$env:AGI_DZ_GBPUSD = if ($env:AGI_DZ_GBPUSD) { $env:AGI_DZ_GBPUSD } else { "0.20" }
$env:AGI_DZ_XAUUSD = if ($env:AGI_DZ_XAUUSD) { $env:AGI_DZ_XAUUSD } else { "0.22" }

if (-not (Test-Path (Join-Path $root "config.yaml"))) {
    Write-Error "FATAL: config.yaml not found. Copy config.yaml.example and configure."
    exit 1
}

if (-not (Test-Path (Join-Path $root "models"))) {
    Write-Warning "models/ directory not found - creating empty. No models available."
    New-Item -ItemType Directory -Path (Join-Path $root "models") -Force | Out-Null
}

$lockFile = Join-Path $root ".tmp" "server_agi.lock"
if (Test-Path $lockFile) {
    $lockPid = Get-Content $lockFile -ErrorAction SilentlyContinue
    $proc = Get-Process -Id $lockPid -ErrorAction SilentlyContinue
    if ($proc) {
        Write-Error "FATAL: Server_AGI is already running (PID $lockPid). Kill it first or remove $lockFile."
        exit 1
    }

    Write-Warning "Removing stale lock file (PID $lockPid is dead)."
    Remove-Item $lockFile -Force
}

Write-Host "Starting Grok AGI Server on Port $($env:AGI_PORT) with AGI_TOKEN sourced from env..."
$py = Join-Path $root ".venv312\\Scripts\\python.exe"
if (-not (Test-Path $py)) {
    $py = Join-Path $root ".venv\\Scripts\\python.exe"
}
if (-not (Test-Path $py)) {
    $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
    if (-not $pythonCmd) {
        throw "No Python interpreter found (.venv312/.venv or global python)."
    }
    $py = $pythonCmd.Source
    Write-Warning "No venv python found. Falling back to global python at $($py)."
}

& $py -c "from Python.config_utils import load_project_config; load_project_config(r'$root', live_mode=True)"
if ($LASTEXITCODE -ne 0) {
    throw "Live config validation failed. Fix config/.env before starting."
}

& $py -m Python.Server_AGI --live
