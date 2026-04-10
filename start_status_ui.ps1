$root = Split-Path -Parent $PSScriptRoot
$py = Join-Path $root '.venv312\Scripts\python.exe'
if (-not (Test-Path $py)) {
  $py = Join-Path $root '.venv\Scripts\python.exe'
}
if (-not (Test-Path $py)) {
  throw "No venv python found (.venv312/.venv). Refusing global python fallback."
}
& $py (Join-Path $root 'tools\project_status_ui.py')
