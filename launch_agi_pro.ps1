<#
.SYNOPSIS
    Cautious Giggle - Professional AGI Trading System Launcher
.DESCRIPTION
    Starts the autonomous trading server (Server_AGI --live), the status
    dashboard (project_status_ui.py :8088), and opens the browser.
    Press Ctrl+C to shut everything down cleanly.
#>

# --------------------------------------------------------------------------- #
#  Configuration
# --------------------------------------------------------------------------- #
$ErrorActionPreference = "Stop"
$RepoRoot     = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvDir      = Join-Path $RepoRoot ".venv312"
$PythonExe    = Join-Path $VenvDir  "Scripts\python.exe"
$ConfigYaml   = Join-Path $RepoRoot "config.yaml"
$LogDir       = Join-Path $RepoRoot "logs"
$LauncherLog  = Join-Path $LogDir   "launcher.log"
$DashboardUrl = "http://127.0.0.1:8088"
$DashboardPort = 8088

# PIDs we own (for shutdown)
$script:OwnedPids = @()

# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #
function Write-Banner {
    $banner = @"

  ============================================================
       CAUTIOUS GIGGLE  -  Autonomous Trading System
  ============================================================
       Server  : Python.Server_AGI --live
       Dashboard: $DashboardUrl
       Venv    : .venv312  (Python 3.12)
  ------------------------------------------------------------

"@
    Write-Host $banner -ForegroundColor Cyan
}

function Write-Log {
    param([string]$Message)
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "[$ts] $Message"
    Write-Host $line
    Add-Content -Path $LauncherLog -Value $line -ErrorAction SilentlyContinue
}

function Test-PortListening {
    param([int]$Port)
    try {
        $conn = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction Stop | Select-Object -First 1
        return $null -ne $conn
    } catch {
        # Fallback to netstat for older systems
        $hit = netstat -ano | Select-String ":$Port\s+.*LISTENING"
        return $null -ne $hit
    }
}

function Test-ProcessAlive {
    param([int]$Pid)
    try {
        $p = Get-Process -Id $Pid -ErrorAction Stop
        return (-not $p.HasExited)
    } catch {
        return $false
    }
}

function Start-BackgroundPython {
    param(
        [string[]]$Arguments,
        [string]$Label
    )
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $PythonExe
    $psi.WorkingDirectory = $RepoRoot
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
    $proc = [System.Diagnostics.Process]::Start($psi)
    if ($null -eq $proc) {
        throw "Failed to start: $Label"
    }
    $script:OwnedPids += [int]$proc.Id
    return [int]$proc.Id
}

function Stop-OwnedProcesses {
    Write-Host ""
    Write-Log "Shutdown requested - stopping owned processes..."
    foreach ($pid in $script:OwnedPids) {
        try {
            if (Test-ProcessAlive -Pid $pid) {
                Stop-Process -Id $pid -Force -ErrorAction Stop
                Write-Log "  Stopped PID $pid"
            }
        } catch {
            Write-Log "  Could not stop PID $pid : $_"
        }
    }
    Write-Log "All processes stopped. Goodbye."
}

# --------------------------------------------------------------------------- #
#  Pre-flight checks
# --------------------------------------------------------------------------- #
Write-Banner

# Ensure log directory
if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
}

Write-Log "=== Launcher started ==="
Write-Log "Repository: $RepoRoot"

# 1. Virtual environment
if (-not (Test-Path $PythonExe)) {
    Write-Log "FATAL: Python venv not found at $VenvDir"
    Write-Host "  Expected: $PythonExe" -ForegroundColor Red
    Write-Host "  Create it with: python -m venv .venv312" -ForegroundColor Yellow
    exit 1
}
$pyVersion = & $PythonExe --version 2>&1
Write-Log "Python: $pyVersion"

# 2. config.yaml
if (-not (Test-Path $ConfigYaml)) {
    Write-Log "FATAL: config.yaml not found at $ConfigYaml"
    Write-Host "  The trading server requires config.yaml in the project root." -ForegroundColor Red
    Write-Host "  Copy config.redacted.yaml and fill in your credentials." -ForegroundColor Yellow
    exit 1
}
Write-Log "Config: config.yaml found"

# 3. Session info
$sessionFile = Join-Path $RepoRoot "runtime\session.json"
if (Test-Path $sessionFile) {
    try {
        $session = Get-Content $sessionFile -Raw | ConvertFrom-Json
        Write-Log "Session: login=$($session.login) server=$($session.server) mode=$($session.mode)"
    } catch {
        Write-Log "Session: runtime\session.json exists but could not be parsed"
    }
} else {
    Write-Log "Session: No existing session (cold start)"
}

# --------------------------------------------------------------------------- #
#  Launch services
# --------------------------------------------------------------------------- #
Write-Host ""
Write-Host "  Starting services..." -ForegroundColor White
Write-Host "  ----------------------------------------------------" -ForegroundColor DarkGray

# --- Dashboard (project_status_ui.py) ---
if (Test-PortListening -Port $DashboardPort) {
    Write-Log "Dashboard: Already listening on port $DashboardPort (skipping)"
} else {
    $dashPid = Start-BackgroundPython -Arguments @("tools\project_status_ui.py") -Label "Dashboard"
    Write-Log "Dashboard: Started (PID $dashPid)"
    # Brief pause to let the dashboard bind
    Start-Sleep -Seconds 2
}

# --- Trading Server (Server_AGI --live) ---
$serverToken = "python.server_agi"
$alreadyRunning = $false
try {
    $rows = Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue
    if ($rows) {
        foreach ($p in $rows) {
            $cmd = ([string]$p.CommandLine).ToLower().Replace("\", "/")
            if ($cmd.Contains("server_agi") -and $cmd.Contains("--live")) {
                $alreadyRunning = $true
                Write-Log "Server: Already running (PID $($p.ProcessId), skipping)"
                break
            }
        }
    }
} catch {}

if (-not $alreadyRunning) {
    $srvPid = Start-BackgroundPython -Arguments @("-m", "Python.Server_AGI", "--live") -Label "Trading Server"
    Write-Log "Server: Started (PID $srvPid)"
}

# --------------------------------------------------------------------------- #
#  Open browser
# --------------------------------------------------------------------------- #
Start-Sleep -Seconds 1
Write-Log "Opening dashboard in browser: $DashboardUrl"
try {
    Start-Process $DashboardUrl | Out-Null
} catch {
    Write-Log "Could not open browser automatically"
}

# --------------------------------------------------------------------------- #
#  Status summary
# --------------------------------------------------------------------------- #
Write-Host ""
Write-Host "  ============================================================" -ForegroundColor Green
Write-Host "       SYSTEM ONLINE" -ForegroundColor Green
Write-Host "  ============================================================" -ForegroundColor Green
Write-Host ""
Write-Host "  Owned processes:" -ForegroundColor White
foreach ($pid in $script:OwnedPids) {
    $status = if (Test-ProcessAlive -Pid $pid) { "running" } else { "exited" }
    $color  = if ($status -eq "running") { "Green" } else { "Red" }
    Write-Host "    PID $pid  [$status]" -ForegroundColor $color
}
Write-Host ""
Write-Host "  Dashboard : $DashboardUrl" -ForegroundColor Cyan
Write-Host "  Logs      : $LauncherLog" -ForegroundColor DarkGray
Write-Host ""
Write-Host "  Press Ctrl+C to shut down all services." -ForegroundColor Yellow
Write-Host ""

Write-Log "Launcher ready. Monitoring $($script:OwnedPids.Count) process(es)."

# --------------------------------------------------------------------------- #
#  Monitor loop - keep alive until Ctrl+C
# --------------------------------------------------------------------------- #
try {
    while ($true) {
        Start-Sleep -Seconds 5

        # Check health of owned processes
        $allDead = $true
        foreach ($pid in $script:OwnedPids) {
            if (Test-ProcessAlive -Pid $pid) {
                $allDead = $false
                break
            }
        }

        if ($allDead -and $script:OwnedPids.Count -gt 0) {
            Write-Log "WARNING: All owned processes have exited unexpectedly."
            Write-Host "  All child processes have exited. Check logs for errors." -ForegroundColor Red
            break
        }
    }
} finally {
    Stop-OwnedProcesses
}

# --------------------------------------------------------------------------- #
#  Desktop shortcut creation helper
# --------------------------------------------------------------------------- #
Write-Host ""
Write-Host "  --- Desktop Shortcut ---" -ForegroundColor Cyan
Write-Host "  Run the following command to create a desktop shortcut:" -ForegroundColor White
Write-Host ""

$shortcutCmd = @"
`$wsh = New-Object -ComObject WScript.Shell
`$sc  = `$wsh.CreateShortcut("`$([Environment]::GetFolderPath('Desktop'))\AGI Trading System.lnk")
`$sc.TargetPath       = "C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe"
`$sc.Arguments         = "-ExecutionPolicy Bypass -File `"$RepoRoot\launch_agi_pro.ps1`""
`$sc.WorkingDirectory  = "$RepoRoot"
`$sc.IconLocation      = "C:\Windows\System32\shell32.dll,44"
`$sc.Description       = "Cautious Giggle - Autonomous Trading System"
`$sc.Save()
Write-Host "Shortcut created on Desktop."
"@

Write-Host $shortcutCmd -ForegroundColor DarkYellow
Write-Host ""
