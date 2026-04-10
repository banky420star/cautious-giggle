param(
    [int]$IntervalSec = 20,
    [string]$OutFile = "logs/review_runtime.jsonl",
    [int]$MaxFileMB = 50
)

$ErrorActionPreference = "SilentlyContinue"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if (-not (Test-Path "logs")) {
    New-Item -ItemType Directory -Path "logs" | Out-Null
}

function TailText([string]$Path, [int]$Lines = 20) {
    if (-not (Test-Path $Path)) { return @() }
    return @(Get-Content -Path $Path -Tail $Lines | ForEach-Object { [string]$_ })
}

function Rotate-OutFileIfNeeded([string]$Path, [int]$MaxMB) {
    if (-not (Test-Path $Path)) { return }
    $it = Get-Item $Path
    $maxBytes = [int64]$MaxMB * 1024 * 1024
    if ([int64]$it.Length -lt $maxBytes) { return }

    $stamp = (Get-Date).ToUniversalTime().ToString("yyyyMMdd_HHmmss")
    $dir = Split-Path -Parent $Path
    $name = Split-Path -Leaf $Path
    $arch = Join-Path $dir ($name + "." + $stamp + ".bak")
    Move-Item -Path $Path -Destination $arch -Force
}

while ($true) {
    Rotate-OutFileIfNeeded -Path $OutFile -MaxMB $MaxFileMB

    $ts = (Get-Date).ToUniversalTime().ToString("o")
    $git = (git -C $root status -sb | Out-String).Trim()

    $proc = Get-CimInstance Win32_Process |
        Where-Object { ($_.Name -match "python|powershell") -and ($_.CommandLine -match "cautious-giggle") } |
        Select-Object ProcessId, Name, CommandLine |
        ForEach-Object {
            [ordered]@{
                pid = [int]$_.ProcessId
                name = [string]$_.Name
                cmd = [string]$_.CommandLine
            }
        }

    $ports = @(
        (netstat -ano | findstr ":9090"),
        (netstat -ano | findstr ":8088")
    ) | Where-Object { $_ -and $_.Trim().Length -gt 0 }

    $files = @(
        "logs\\server.log",
        "logs\\audit_events.jsonl",
        "logs\\lstm_training.log",
        "logs\\ppo_training.log",
        "logs\\backtester.log",
        ".tmp\\runtime_state.json"
    )

    $fileState = @{}
    foreach ($f in $files) {
        $full = Join-Path $root $f
        if (Test-Path $full) {
            $it = Get-Item $full
            $fileState[$f] = @{
                size_bytes = [int64]$it.Length
                last_write_utc = $it.LastWriteTimeUtc.ToString("o")
            }
        } else {
            $fileState[$f] = @{
                missing = $true
            }
        }
    }

    $snapshot = [ordered]@{
        ts_utc = $ts
        git_status = $git
        processes = $proc
        ports = $ports
        env = @{
            MT5_LOGIN = -not [string]::IsNullOrWhiteSpace($env:MT5_LOGIN)
            MT5_PASSWORD = -not [string]::IsNullOrWhiteSpace($env:MT5_PASSWORD)
            MT5_SERVER = -not [string]::IsNullOrWhiteSpace($env:MT5_SERVER)
            TELEGRAM_TOKEN = -not [string]::IsNullOrWhiteSpace($env:TELEGRAM_TOKEN)
            TELEGRAM_CHAT_ID = -not [string]::IsNullOrWhiteSpace($env:TELEGRAM_CHAT_ID)
            AGI_TOKEN = -not [string]::IsNullOrWhiteSpace($env:AGI_TOKEN)
        }
        files = $fileState
        tails = @{
            server = TailText (Join-Path $root "logs\\server.log") 6
            audit = TailText (Join-Path $root "logs\\audit_events.jsonl") 3
            backtester = TailText (Join-Path $root "logs\\backtester.log") 6
        }
    }

    ($snapshot | ConvertTo-Json -Depth 4 -Compress) | Add-Content -Path $OutFile -Encoding UTF8
    Start-Sleep -Seconds $IntervalSec
}
