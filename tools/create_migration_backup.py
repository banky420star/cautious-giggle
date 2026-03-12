import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path


MAX_PART_BYTES = 95 * 1024 * 1024
REDACT_KEYS = (
    "password",
    "passwd",
    "secret",
    "token",
    "api_key",
    "apikey",
    "key",
    "login",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_value(repo_root: Path, *args: str) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo_root), *args],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or None
    except Exception:
        return None


def redact_config(src: Path, dest: Path) -> bool:
    if not src.exists():
        return False
    out_lines: list[str] = []
    for line in src.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or ":" not in line:
            out_lines.append(line)
            continue
        key = stripped.split(":", 1)[0].strip().lower()
        if any(token in key for token in REDACT_KEYS):
            indent = line[: len(line) - len(line.lstrip(" "))]
            raw_key = line.split(":", 1)[0]
            out_lines.append(f"{indent}{raw_key}: REDACTED")
        else:
            out_lines.append(line)
    dest.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    return True


def add_tree_archive(src_root: Path, archive_path: Path) -> dict:
    with tarfile.open(archive_path, "w:gz") as tf:
        tf.add(src_root, arcname=src_root.name)
    return {
        "source": str(src_root.relative_to(src_root.parent.parent if src_root.parent.parent.exists() else src_root.parent)),
        "archive": archive_path.name,
        "size_bytes": archive_path.stat().st_size,
        "sha256": sha256_file(archive_path),
    }


def split_file(path: Path, max_part_bytes: int = MAX_PART_BYTES) -> list[dict]:
    if path.stat().st_size <= max_part_bytes:
        return [{
            "file": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }]

    parts: list[dict] = []
    idx = 1
    with path.open("rb") as src:
        while True:
            chunk = src.read(max_part_bytes)
            if not chunk:
                break
            part_path = path.with_suffix(path.suffix + f".part{idx:02d}")
            with part_path.open("wb") as out:
                out.write(chunk)
            parts.append({
                "file": part_path.name,
                "size_bytes": part_path.stat().st_size,
                "sha256": sha256_file(part_path),
            })
            idx += 1
    path.unlink()
    return parts


def write_restore_notes(dest: Path, manifest: dict) -> None:
    notes = f"""# VPS Migration Backup

Created: {manifest["created_utc"]}
Git branch: {manifest.get("git_branch") or "unknown"}
Git commit: {manifest.get("git_commit") or "unknown"}

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
   Get-Content .\\logs.tar.gz.part* -Encoding Byte | Set-Content .\\logs.tar.gz -Encoding Byte
   Get-Content .\\models.tar.gz.part* -Encoding Byte | Set-Content .\\models.tar.gz -Encoding Byte
   ```
4. Extract:
   ```powershell
   tar -xzf .\\logs.tar.gz
   tar -xzf .\\models.tar.gz
   ```
5. Recreate `config.yaml` locally from your secure secrets source. The redacted config in this folder is only a reference.

## Safety note
This backup intentionally does not commit raw secrets from `config.yaml` to GitHub.
"""
    dest.write_text(notes, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a GitHub-friendly VPS migration backup snapshot.")
    parser.add_argument("--repo-root", default=".", help="Path to the repository root")
    parser.add_argument("--name", default=None, help="Optional backup folder name")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup_name = args.name or f"vps_migration_{timestamp}"
    backup_root = repo_root / "backups" / backup_name
    backup_root.mkdir(parents=True, exist_ok=True)

    manifest: dict = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_branch": git_value(repo_root, "rev-parse", "--abbrev-ref", "HEAD"),
        "git_commit": git_value(repo_root, "rev-parse", "HEAD"),
        "artifacts": [],
        "notes": [
            "config.yaml is redacted before being committed to GitHub",
            "tracked code is already preserved by git history; this backup adds runtime state",
        ],
    }

    for name in ("logs", "models"):
        src = repo_root / name
        if not src.exists():
            continue
        archive_path = backup_root / f"{name}.tar.gz"
        entry = add_tree_archive(src, archive_path)
        entry["parts"] = split_file(archive_path)
        manifest["artifacts"].append(entry)

    redacted = backup_root / "config.redacted.yaml"
    manifest["config_redacted"] = redact_config(repo_root / "config.yaml", redacted)
    if manifest["config_redacted"]:
        manifest["config_redacted_sha256"] = sha256_file(redacted)

    write_restore_notes(backup_root / "README.md", manifest)
    (backup_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(json.dumps({
        "backup_root": str(backup_root),
        "artifacts": manifest["artifacts"],
        "config_redacted": manifest["config_redacted"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
