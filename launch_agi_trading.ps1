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

function Test-PortListening {
  param([int]$Port)
  try {
    $row = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction Stop | Select-Object -First 1
    return $null -ne $row
  } catch {
    $legacy = netstat -ano | Select-String ":$Port\s+.*LISTENING"
    return $null -ne $legacy
  }
}

$serverCmd = "& '$pythonExe' -m Python.Server_AGI --live"
$uiCmd = "& '$pythonExe' tools\project_status_ui.py"

Start-LauncherWindow -Title "AGI Server" -InnerCommand $serverCmd
Start-Sleep -Seconds 2
Start-LauncherWindow -Title "AGI Status UI" -InnerCommand $uiCmd

if ($StartN8N) {
  if (Test-PortListening -Port 5678) {
    Write-Host "n8n already listening on port 5678. Skipping n8n launch."
  } else {
    $n8nCmd = @(
      "`$env:NODES_EXCLUDE='[]'",
      "`$env:N8N_DIAGNOSTICS_ENABLED='false'",
      "`$env:N8N_VERSION_NOTIFICATIONS_ENABLED='false'",
      "`$env:N8N_PERSONALIZATION_ENABLED='false'",
      "n8n start"
    ) -join "; "
    Start-LauncherWindow -Title "n8n Orchestrator" -InnerCommand $n8nCmd
  }
}

Start-Sleep -Seconds 2
Start-Process $UiUrl | Out-Null
Write-Host "AGI Trading launcher started. UI: $UiUrl"
