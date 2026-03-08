param(
  [switch]$StartN8N = $false,
  [string]$UiUrl = "http://127.0.0.1:8088"
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

$pythonCandidates = @(
  (Join-Path $repoRoot ".venv312\Scripts\python.exe"),
  (Join-Path $repoRoot ".venv\Scripts\python.exe"),
  "python"
)

$pythonExe = $null
foreach ($cand in $pythonCandidates) {
  if ($cand -eq "python") {
    $pythonExe = $cand
    break
  }
  if (Test-Path $cand) {
    $pythonExe = $cand
    break
  }
}

if (-not $pythonExe) {
  throw "Python executable not found."
}

function Start-LauncherWindow {
  param(
    [string]$Title,
    [string]$InnerCommand
  )

  $cmd = "Set-Location -Path '$repoRoot'; `$Host.UI.RawUI.WindowTitle = '$Title'; $InnerCommand"
  Start-Process powershell.exe -ArgumentList @(
    "-NoExit",
    "-ExecutionPolicy", "Bypass",
    "-Command", $cmd
  ) | Out-Null
}

$serverCmd = "& '$pythonExe' -m Python.Server_AGI --live"
$uiCmd = "& '$pythonExe' tools\project_status_ui.py"

Start-LauncherWindow -Title "AGI Server" -InnerCommand $serverCmd
Start-Sleep -Seconds 2
Start-LauncherWindow -Title "AGI Status UI" -InnerCommand $uiCmd

if ($StartN8N) {
  Start-LauncherWindow -Title "n8n Orchestrator" -InnerCommand "`$env:NODES_EXCLUDE='[]'; n8n start"
}

Start-Sleep -Seconds 2
Start-Process $UiUrl | Out-Null
Write-Host "AGI Trading launcher started. UI: $UiUrl"
