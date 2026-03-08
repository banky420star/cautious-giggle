$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$launcher = Join-Path $repoRoot "launch_agi_trading.ps1"

if (-not (Test-Path $launcher)) {
  throw "Launcher script not found: $launcher"
}

$desktop = [Environment]::GetFolderPath("Desktop")
$shortcutPath = Join-Path $desktop "AGI Trading.lnk"

$wsh = New-Object -ComObject WScript.Shell
$shortcut = $wsh.CreateShortcut($shortcutPath)
$shortcut.TargetPath = "C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe"
$shortcut.Arguments = "-ExecutionPolicy Bypass -File `"$launcher`""
$shortcut.WorkingDirectory = $repoRoot
$shortcut.IconLocation = "C:\Windows\System32\shell32.dll,44"
$shortcut.Description = "Start AGI server + UI"
$shortcut.Save()

Write-Host "Created shortcut: $shortcutPath"
