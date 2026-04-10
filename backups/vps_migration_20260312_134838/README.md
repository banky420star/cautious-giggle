# VPS Migration Backup

Created: 2026-03-12T13:48:38.377277+00:00
Git branch: main
Git commit: 12d99a1457dd0b07d2cc9cf187e400df19e6ecb7

## Contents
- `logs.tar.gz` or split parts: full `logs/` snapshot at backup time
- `models.tar.gz` or split parts: full `models/` snapshot at backup time
- `config.redacted.yaml`: redacted runtime config snapshot for reference
- `manifest.json`: checksums and sizes

## Restore
1. Clone the repo on the new VPS.
2. Copy these backup files from the repo checkout.
3. If an archive is split, reassemble it first:
   ```powershell
   Get-Content .\logs.tar.gz.part* -Encoding Byte | Set-Content .\logs.tar.gz -Encoding Byte
   Get-Content .\models.tar.gz.part* -Encoding Byte | Set-Content .\models.tar.gz -Encoding Byte
   ```
4. Extract:
   ```powershell
   tar -xzf .\logs.tar.gz
   tar -xzf .\models.tar.gz
   ```
5. Recreate `config.yaml` locally from your secure secrets source. The redacted config in this folder is only a reference.

## Safety note
This backup intentionally does not commit raw secrets from `config.yaml` to GitHub.
