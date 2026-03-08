# Public Repo Polish Checklist

## 1. Description

Set repository description to:

`MT5-first autonomous trading research/runtime stack with symbol-scoped model lifecycle and live operator telemetry.`

## 2. Release/Tag Flow

```powershell
git checkout main
git pull --ff-only origin main
git tag -a v0.2.0 -m "Evidence pack + canary survival enforcement + expanded tests"
git push origin v0.2.0
```

Optional GitHub release notes sections:
- Security and configuration hardening
- Promotion gate enforcement
- Walk-forward evidence artifacts
- CI/test coverage expansion

## 3. Evidence Bundle

```powershell
python tools\build_evidence_pack.py
```

Ensure these are committed or attached to release:
- `docs/results/walk_forward_results.csv`
- `docs/results/walk_forward_summary.md`
- `docs/results/evidence_bundle.md`
- `docs/screenshots/*`

## 4. External Reader Validation

- README aligns with code paths and defaults.
- No secrets in tracked files.
- CI badge/workflow visible.
- At least one release tag published.
