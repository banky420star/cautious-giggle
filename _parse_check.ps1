$null=$tokens=$errors=$null
[System.Management.Automation.Language.Parser]::ParseFile((Join-Path $PWD start_server.ps1),[ref]$tokens,[ref]$errors)|Out-Null
if($errors.Count -gt 0){$errors|ForEach-Object{$_.Message};exit 1}
Write-Output ps1_parse_ok
