import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from Python.model_registry import ModelRegistry

RESULTS_DIR = ROOT / "docs" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_PATH = RESULTS_DIR / "release_summary.md"
PROFIT_LOG = ROOT / "logs" / "profitability.jsonl"


def _read_recent_profit():
    if not PROFIT_LOG.exists():
        return []
    lines = PROFIT_LOG.read_text(encoding="utf-8").strip().splitlines()
    return [json.loads(l) for l in lines[-5:]] if lines else []


def _load_metadata(path: Path) -> dict:
    meta = {"path": str(path)}
    json_path = path / "metadata.json"
    if json_path.exists():
        try:
            meta.update(json.loads(json_path.read_text(encoding="utf-8")))
        except Exception:
            pass
    return meta


def main():
    registry = ModelRegistry()
    active = registry._read_active()
    champion = active.get("champion")
    canary = active.get("canary")

    champion_meta = _load_metadata(Path(champion)) if champion else {}
    canary_meta = _load_metadata(Path(canary)) if canary else {}

    profit_snapshots = _read_recent_profit()

    lines = [
        "# Release Summary",
        "",
        f"**Timestamp:** {json.dumps(__import__('datetime').datetime.utcnow().isoformat())}",
        "",
        "## Champion Overview",
        f"- Path: {champion or 'None'}",
        f"- Metadata: {json.dumps(champion_meta)}",
        "",
        "## Canary Overview",
        f"- Path: {canary or 'None'}",
        f"- Metadata: {json.dumps(canary_meta)}",
        "",
        "## Profitability recent snapshots",
    ]

    if profit_snapshots:
        for snap in profit_snapshots:
            lines.append(f"- `{snap.get('ts')}` equity={snap.get('equity'):.2f} position={snap.get('position')} "
                         f"profitability={json.dumps(snap.get('profitability', {}))}")
    else:
        lines.append("- *(no profitability records yet)*")

    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
