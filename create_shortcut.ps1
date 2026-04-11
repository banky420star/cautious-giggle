$ErrorActionPreference = "Stop"
$repoRoot = "C:\Users\Administrator\work\cautious-giggle-clone-20260320161357"
$python = Join-Path $repoRoot ".venv312\Scripts\python.exe"
$pngPath = "C:\Users\Administrator\Desktop\Screenshot 2026-03-18 at 04.04.32.png"
$assetsDir = Join-Path $repoRoot "assets"
$icoPath = Join-Path $assetsDir "app_icon.ico"

# Create assets directory
if (-not (Test-Path $assetsDir)) {
    New-Item -ItemType Directory -Path $assetsDir | Out-Null
}

# Convert PNG to ICO using Pillow
$convertScript = @"
from PIL import Image
import sys
img = Image.open(sys.argv[1])
img = img.resize((256, 256), Image.LANCZOS)
img.save(sys.argv[2], format='ICO', sizes=[(256,256),(128,128),(64,64),(48,48),(32,32),(16,16)])
print('Icon created:', sys.argv[2])
"@

$useIcon = $false
if (Test-Path $pngPath) {
    try {
        & $python -c $convertScript "$pngPath" "$icoPath"
        if ($LASTEXITCODE -eq 0 -and (Test-Path $icoPath)) {
            Write-Host "Icon converted successfully: $icoPath"
            $useIcon = $true
        }
    } catch {
        Write-Host "Warning: PNG to ICO conversion failed. Using default icon."
    }
} else {
    Write-Host "Screenshot not found at: $pngPath"
}

# Create desktop shortcut
$desktop = [Environment]::GetFolderPath("Desktop")
$shortcutPath = Join-Path $desktop "Cautious Giggle.lnk"
$batPath = Join-Path $repoRoot "run_all.bat"

$wsh = New-Object -ComObject WScript.Shell
$shortcut = $wsh.CreateShortcut($shortcutPath)
$shortcut.TargetPath = $batPath
$shortcut.WorkingDirectory = $repoRoot
$shortcut.Description = "Launch Cautious Giggle Trading System (Backend + UI + Training)"
$shortcut.WindowStyle = 1

if ($useIcon) {
    $shortcut.IconLocation = "$icoPath,0"
}

$shortcut.Save()
Write-Host "Desktop shortcut created: $shortcutPath"
