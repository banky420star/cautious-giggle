$ws = New-Object -ComObject WScript.Shell
$s = $ws.CreateShortcut("$env:USERPROFILE\Desktop\AGI Trading System.lnk")
$s.TargetPath = "C:\Users\Administrator\work\cautious-giggle-clone-20260320161357\run_all.bat"
$s.WorkingDirectory = "C:\Users\Administrator\work\cautious-giggle-clone-20260320161357"
$s.Description = "Launch AGI Trading System (Backend + Frontend)"
$s.WindowStyle = 1
$s.Save()
Write-Host "Desktop shortcut created."
