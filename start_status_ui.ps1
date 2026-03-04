$root = Split-Path -Parent $PSScriptRoot
$py = Join-Path $root '.venv312\Scripts\python.exe'
if (-not (Test-Path $py)) {
  $py = 'python'
}
& $py (Join-Path $root 'tools\project_status_ui.py')
