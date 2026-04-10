param(
    [int] = 60,
    [int] = 120,
    [switch]
)

Set-StrictMode -Version Latest

 = Split-Path -Parent System.Management.Automation.InvocationInfo.MyCommand.Path
 = Resolve-Path (Join-Path  '..')
 = Join-Path  '.venv312\Scripts\Activate.ps1'
if (Test-Path ) {
    Write-Host  Activating virtual environment from 
    .  | Out-Null
} else {
    Write-Warning Virtual environment activate script not found at 
}

function Run-Helper(, ) {
    Write-Host [2026-03-11T22:09:34.5087077+01:00] Running 
    try {
        & python 
    } catch {
        Write-Warning Helper failed: 
    }
}

 = [timespan]::FromMinutes()
 = [timespan]::FromMinutes()

 = Get-Date
 = Get-Date

while (True) {
     = Get-Date
    if ( -ge ) {
        Run-Helper 'tools/release_summary.py' 'tools/release_summary.py'
         =  + 
    }
    if ( -ge ) {
        Run-Helper 'tools/profit_sweep.py' 'tools/profit_sweep.py'
         =  + 
    }
    if () { break }
    Start-Sleep -Seconds 30
}
