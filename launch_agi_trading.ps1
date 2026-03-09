param(
  [bool]$StartN8N = $true,
  [bool]$StartTrainingCycle = $true,
  [bool]$AutoBootstrapModel = $true,
  [int]$CycleIntervalMinutes = 30,
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

function Start-DetachedProcess {
  param(
    [string]$FilePath,
    [string[]]$Arguments = @(),
    [hashtable]$Environment = @{}
  )

  $psi = New-Object System.Diagnostics.ProcessStartInfo
  $psi.FileName = $FilePath
  $psi.WorkingDirectory = $repoRoot
  $psi.UseShellExecute = $false
  $psi.CreateNoWindow = $true
  if ($Arguments -and $Arguments.Count -gt 0) {
    $escaped = @()
    foreach ($arg in $Arguments) {
      $s = [string]$arg
      if ($s.Contains(" ") -or $s.Contains('"')) {
        $s = '"' + ($s -replace '"', '\"') + '"'
      }
      $escaped += $s
    }
    $psi.Arguments = ($escaped -join " ")
  }
  foreach ($entry in $Environment.GetEnumerator()) {
    $psi.EnvironmentVariables[[string]$entry.Key] = [string]$entry.Value
  }
  $proc = [System.Diagnostics.Process]::Start($psi)
  if ($null -eq $proc) {
    throw "Failed to start process: $FilePath"
  }
  return [int]$proc.Id
}

function Get-LauncherShellRows {
  $rows = Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" -ErrorAction SilentlyContinue
  if (-not $rows) { return @() }
  $needle = $repoRoot.ToLower().Replace("\", "/")
  $matched = @()
  foreach ($p in $rows) {
    $cmd = ([string]$p.CommandLine).ToLower().Replace("\", "/")
    if ($cmd.Contains($needle) -and $cmd.Contains("-noexit")) {
      $matched += $p
    }
  }
  return $matched
}

function Remove-IdleLauncherShells {
  $shells = Get-LauncherShellRows
  if (-not $shells) { return @() }
  $all = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue
  $killed = @()
  foreach ($p in $shells) {
    $children = @($all | Where-Object { $_.ParentProcessId -eq $p.ProcessId })
    $liveWorkers = @($children | Where-Object { $_.Name -in @("python.exe", "node.exe") })
    if ($liveWorkers.Count -eq 0) {
      try {
        Stop-Process -Id $p.ProcessId -Force -ErrorAction Stop
        $killed += [int]$p.ProcessId
      } catch {
      }
    }
  }
  if ($killed.Count -gt 0) {
    Write-Host "Removed idle launcher PowerShell process(es): $($killed -join ', ')"
  }
  return $killed
}

function Get-NodeExe {
  try {
    $cmd = Get-Command node -ErrorAction Stop
    return [string]$cmd.Source
  } catch {
    return $null
  }
}

function Get-N8NBin {
  $candidates = @(
    (Join-Path $env:APPDATA "npm\node_modules\n8n\bin\n8n"),
    (Join-Path $env:APPDATA "npm\node_modules\n8n\bin\n8n.js")
  )
  foreach ($cand in $candidates) {
    if (Test-Path $cand) {
      return $cand
    }
  }
  return $null
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

function Test-ProcessToken {
  param([string]$Token)
  $rows = Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue
  if (-not $rows) { return $false }
  $needle = $Token.ToLower().Replace("\", "/")
  foreach ($p in $rows) {
    $cmd = ([string]$p.CommandLine).ToLower().Replace("\", "/")
    if ($cmd.Contains($needle)) { return $true }
  }
  return $false
}

function Get-ProcessRowsByToken {
  param([string]$Token)
  $rows = Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue
  if (-not $rows) { return @() }
  $needle = $Token.ToLower().Replace("\", "/")
  $matched = @()
  foreach ($p in $rows) {
    $cmd = ([string]$p.CommandLine).ToLower().Replace("\", "/")
    if ($cmd.Contains($needle)) { $matched += $p }
  }
  return $matched
}

function Get-RootProcessRowsByToken {
  param([string]$Token)
  $rows = Get-ProcessRowsByToken -Token $Token
  if (-not $rows) { return @() }

  $pidSet = @{}
  foreach ($p in $rows) {
    $pidSet[[int]$p.ProcessId] = $true
  }

  $roots = @()
  foreach ($p in $rows) {
    if (-not $pidSet.ContainsKey([int]$p.ParentProcessId)) {
      $roots += $p
    }
  }
  return $roots
}

function Remove-StalePythonDuplicates {
  param(
    [string]$Token,
    [int]$Keep = 1
  )

  $rows = Get-RootProcessRowsByToken -Token $Token
  if (-not $rows -or $rows.Count -le $Keep) { return @() }

  $ordered = $rows | Sort-Object CreationDate
  $toStop = $ordered | Select-Object -First ($ordered.Count - $Keep)
  $killed = @()
  foreach ($p in $toStop) {
    try {
      Stop-Process -Id $p.ProcessId -Force -ErrorAction Stop
      $killed += [int]$p.ProcessId
    } catch {
    }
  }
  if ($killed.Count -gt 0) {
    Write-Host "Stopped duplicate process(es) for token '$Token': $($killed -join ', ')"
  }
  return $killed
}

function Remove-StaleServerLock {
  $lockPath = Join-Path $repoRoot ".tmp\server_agi.lock"
  if (-not (Test-Path $lockPath)) { return }
  $raw = ""
  try {
    $raw = (Get-Content -Path $lockPath -Raw -ErrorAction Stop).Trim()
  } catch {
    return
  }
  if ([string]::IsNullOrWhiteSpace($raw)) {
    Remove-Item -LiteralPath $lockPath -Force -ErrorAction SilentlyContinue
    return
  }

  $lockPid = 0
  [void][int]::TryParse($raw, [ref]$lockPid)
  if ($lockPid -le 0) {
    Remove-Item -LiteralPath $lockPath -Force -ErrorAction SilentlyContinue
    return
  }

  $proc = Get-CimInstance Win32_Process -Filter "ProcessId=$lockPid" -ErrorAction SilentlyContinue
  if (-not $proc) {
    Remove-Item -LiteralPath $lockPath -Force -ErrorAction SilentlyContinue
    Write-Host "Removed stale Server_AGI lock (missing pid $lockPid)."
    return
  }

  $cmd = ([string]$proc.CommandLine).ToLower().Replace("\", "/")
  if (-not $cmd.Contains("python.server_agi")) {
    Remove-Item -LiteralPath $lockPath -Force -ErrorAction SilentlyContinue
    Write-Host "Removed stale Server_AGI lock (pid $lockPid is not Server_AGI)."
  }
}

function Get-RegistryState {
  $code = @"
import json
from Python.model_registry import ModelRegistry
r = ModelRegistry()
a = r._read_active()
latest = None
dirs = []
import os
if os.path.isdir(r.candidates_dir):
    dirs = [os.path.join(r.candidates_dir,d) for d in os.listdir(r.candidates_dir) if os.path.isdir(os.path.join(r.candidates_dir,d))]
if dirs:
    dirs.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    latest = dirs[0]
print(json.dumps({'champion': a.get('champion'), 'canary': a.get('canary'), 'latest': latest}))
"@
  try {
    $raw = & $pythonExe -c $code
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($raw)) {
      return $null
    }
    return ($raw | ConvertFrom-Json)
  } catch {
    return $null
  }
}

function Bootstrap-RegistryIfNeeded {
  param($State)
  if (-not $State) { return }
  if ($State.champion -or $State.canary) { return }
  if (-not $State.latest) { return }

  $latestEsc = [string]$State.latest
  $code = @"
from Python.model_registry import ModelRegistry
r = ModelRegistry()
r.set_canary(r'''$latestEsc''')
print('canary_bootstrapped')
"@
  $null = & $pythonExe -c $code
  if ($LASTEXITCODE -eq 0) {
    Write-Host "Bootstrapped active model: canary <- $latestEsc"
  }
}

if ($AutoBootstrapModel) {
  $state = Get-RegistryState
  Bootstrap-RegistryIfNeeded -State $state
}

# Keep one owner for runtime and training-cycle processes.
Remove-IdleLauncherShells | Out-Null
Remove-StalePythonDuplicates -Token "python.server_agi" | Out-Null
Remove-StalePythonDuplicates -Token "tools/project_status_ui.py" | Out-Null
Remove-StalePythonDuplicates -Token "tools/champion_cycle_loop.py" | Out-Null
Remove-StalePythonDuplicates -Token "tools/champion_cycle.py" | Out-Null
Remove-StalePythonDuplicates -Token "training/train_drl.py" | Out-Null
Remove-StalePythonDuplicates -Token "training/train_lstm.py" | Out-Null
Remove-StaleServerLock

if (-not (Test-ProcessToken -Token "python.server_agi")) {
  $serverPid = Start-DetachedProcess -FilePath $pythonExe -Arguments @("-m", "Python.Server_AGI", "--live")
  Write-Host "Started AGI Server pid=$serverPid"
  Start-Sleep -Seconds 2
} else {
  Write-Host "Server already running; skipping duplicate launch."
}

if (-not (Test-PortListening -Port 8088)) {
  $uiPid = Start-DetachedProcess -FilePath $pythonExe -Arguments @("tools\project_status_ui.py")
  Write-Host "Started Status UI pid=$uiPid"
} else {
  Write-Host "Status UI already listening on 8088; skipping duplicate launch."
}

if ($StartN8N) {
  if (Test-PortListening -Port 5678) {
    Write-Host "n8n already listening on port 5678. Skipping n8n launch."
  } else {
    $nodeExe = Get-NodeExe
    $n8nBin = Get-N8NBin
    if (-not $nodeExe -or -not $n8nBin) {
      Write-Host "n8n executable not found; skipping n8n launch."
    } else {
      $n8nPid = Start-DetachedProcess -FilePath $nodeExe -Arguments @($n8nBin, "start") -Environment @{
        "NODES_EXCLUDE" = "[]"
        "N8N_DIAGNOSTICS_ENABLED" = "false"
        "N8N_VERSION_NOTIFICATIONS_ENABLED" = "false"
        "N8N_PERSONALIZATION_ENABLED" = "false"
      }
      Write-Host "Started n8n pid=$n8nPid"
    }
  }
}

if ($StartTrainingCycle) {
  if (Test-ProcessToken -Token "tools/champion_cycle_loop.py") {
    Write-Host "Champion cycle loop already running; skipping duplicate launch."
  } else {
    $cyclePid = Start-DetachedProcess -FilePath $pythonExe -Arguments @("tools\champion_cycle_loop.py", "--interval-minutes", "$CycleIntervalMinutes")
    Write-Host "Started champion cycle loop pid=$cyclePid"
  }
}

Start-Sleep -Seconds 2
Remove-IdleLauncherShells | Out-Null
Start-Process $UiUrl | Out-Null
Write-Host "AGI Trading launcher started. UI: $UiUrl"
