param(
    [int]$IntervalSec = 20,
    [string]$OutFile = "logs/review_runtime.jsonl"
)

$ErrorActionPreference = "SilentlyContinue"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if (-not (Test-Path "logs")) {
    New-Item -ItemType Directory -Path "logs" | Out-Null
}

function TailText([string]$Path, [int]$Lines = 20) {
    if (-not (Test-Path $Path)) { return @() }
    return Get-Content -Path $Path -Tail $Lines
}

while ($true) {
    $ts = (Get-Date).ToUniversalTime().ToString("o")
    $git = (git -C $root status -sb | Out-String).Trim()

    $proc = Get-CimInstance Win32_Process |
        Where-Object { ($_.Name -match "python|powershell") -and ($_.CommandLine -match "cautious-giggle") } |
        Select-Object ProcessId, Name, CommandLine

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
            server = TailText (Join-Path $root "logs\\server.log") 20
            audit = TailText (Join-Path $root "logs\\audit_events.jsonl") 10
            backtester = TailText (Join-Path $root "logs\\backtester.log") 20
        }
    }

    ($snapshot | ConvertTo-Json -Depth 6 -Compress) | Add-Content -Path $OutFile -Encoding UTF8
    Start-Sleep -Seconds $IntervalSec
}
