param(
  [string[]]$Symbols = @(),
  [int]$Timesteps = 120000
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repoRoot

$python = Join-Path $repoRoot ".venv312\Scripts\python.exe"
if (-not (Test-Path $python)) {
  $python = "python"
}

if ($Symbols.Count -eq 0) {
  $cfg = Join-Path $repoRoot "config.yaml"
  if (Test-Path $cfg) {
    $json = & $python -c "import yaml, json, pathlib; p=pathlib.Path('config.yaml'); c=yaml.safe_load(p.read_text(encoding='utf-8')) or {}; print(json.dumps((c.get('trading',{}) or {}).get('symbols',[])))"
    if ($json) {
      $Symbols = @((ConvertFrom-Json $json))
    }
  }
}

if ($Symbols.Count -eq 0) {
  $Symbols = @("EURUSDm")
}

Write-Host "Starting per-symbol DRL training. symbols=$($Symbols -join ',') timesteps=$Timesteps"

foreach ($sym in $Symbols) {
  Write-Host "Training symbol: $sym"
  $env:AGI_DRL_SYMBOL = $sym
  $env:AGI_DRL_TIMESTEPS = "$Timesteps"
  & $python "training\train_drl.py"
  if ($LASTEXITCODE -ne 0) {
    throw "Training failed for symbol $sym"
  }
}

Remove-Item Env:\AGI_DRL_SYMBOL -ErrorAction SilentlyContinue
Remove-Item Env:\AGI_DRL_TIMESTEPS -ErrorAction SilentlyContinue
Write-Host "Per-symbol DRL training completed."
