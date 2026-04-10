import csv
import glob
import json
import os
from datetime import datetime, timezone


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = os.path.join(ROOT, "logs")
OUT = os.path.join(ROOT, "docs", "results")
CANDIDATES = os.path.join(ROOT, "models", "registry", "candidates")


def _read_json(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _collect_walk_forward_rows():
    rows = []
    for path in sorted(glob.glob(os.path.join(LOGS, "eval_*.json"))):
        payload = _read_json(path)
        if not isinstance(payload, dict):
            continue
        rows.append(
            {
                "source_file": os.path.basename(path),
                "symbol": str(payload.get("symbol", "")),
                "return_pct": float(payload.get("return_pct", 0.0)),
                "max_drawdown_pct": float(payload.get("max_drawdown_pct", 0.0)),
                "sharpe": float(payload.get("sharpe", 0.0)),
                "trades": int(payload.get("trades", 0)),
                "final_equity": float(payload.get("final_equity", 0.0)),
                "model_dir": str(payload.get("model_dir", "")),
            }
        )
    return rows


def _collect_cycle_rows():
    path = os.path.join(LOGS, "champion_cycle_last_report.json")
    payload = _read_json(path)
    if not isinstance(payload, dict):
        return []
    out = []
    for row in payload.get("symbols", []):
        if not isinstance(row, dict):
            continue
        out.append(
            {
                "symbol": str(row.get("symbol", "")),
                "wins": bool(row.get("wins", False)),
                "passes_thresholds": bool(row.get("passes_thresholds", False)),
                "pass_rate": float(row.get("pass_rate", 0.0) or 0.0),
                "candidate": str(row.get("candidate", "")),
                "champion": str(row.get("champion", "")),
            }
        )
    return out


def _collect_candidate_rows(limit: int = 12):
    rows = []
    if not os.path.isdir(CANDIDATES):
        return rows
    dirs = [os.path.join(CANDIDATES, d) for d in os.listdir(CANDIDATES) if os.path.isdir(os.path.join(CANDIDATES, d))]
    dirs.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    for d in dirs[:limit]:
        scorecard = _read_json(os.path.join(d, "scorecard.json")) or {}
        rows.append(
            {
                "candidate_dir": os.path.basename(d),
                "symbol": scorecard.get("symbol") or ",".join(scorecard.get("symbols", [])),
                "timeframe": scorecard.get("timeframe") or scorecard.get("interval"),
                "candles": scorecard.get("candles", ""),
                "timesteps": scorecard.get("timesteps", ""),
                "feature_set_version": scorecard.get("feature_set_version", ""),
                "reward_version": scorecard.get("reward_version", ""),
                "date": scorecard.get("date", ""),
            }
        )
    return rows


def _write_csv(path: str, rows: list[dict], fieldnames: list[str]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def _md_table(rows: list[dict], cols: list[str]) -> str:
    if not rows:
        return "_No rows available._"
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join([str(row.get(c, "")) for c in cols]) + " |")
    return "\n".join([head, sep] + body)


def build():
    os.makedirs(OUT, exist_ok=True)
    ts = datetime.now(timezone.utc).isoformat()

    wf_rows = _collect_walk_forward_rows()
    cycle_rows = _collect_cycle_rows()
    candidate_rows = _collect_candidate_rows()

    wf_csv = os.path.join(OUT, "walk_forward_results.csv")
    _write_csv(
        wf_csv,
        wf_rows,
        ["source_file", "symbol", "return_pct", "max_drawdown_pct", "sharpe", "trades", "final_equity", "model_dir"],
    )

    summary_md = os.path.join(OUT, "walk_forward_summary.md")
    with open(summary_md, "w", encoding="utf-8") as f:
        f.write("# Walk-Forward Summary\n\n")
        f.write(f"Generated (UTC): `{ts}`\n\n")
        f.write("## Walk-Forward Rows\n\n")
        f.write(
            _md_table(
                wf_rows,
                ["source_file", "symbol", "return_pct", "max_drawdown_pct", "sharpe", "trades", "final_equity"],
            )
        )
        f.write("\n\n## Champion Cycle Snapshot\n\n")
        f.write(_md_table(cycle_rows, ["symbol", "wins", "passes_thresholds", "pass_rate", "candidate", "champion"]))
        if wf_rows:
            labels = ", ".join([f"\"{str(r['symbol'])}\"" for r in wf_rows])
            returns = ", ".join([str(round(float(r["return_pct"]), 2)) for r in wf_rows])
            f.write("\n\n## Return Chart (Mermaid)\n\n")
            f.write("```mermaid\n")
            f.write("xychart-beta\n")
            f.write("  title \"Walk-Forward Return %\"\n")
            f.write(f"  x-axis [{labels}]\n")
            f.write("  y-axis \"Return %\" -20 --> 60\n")
            f.write(f"  bar [{returns}]\n")
            f.write("```\n")
        f.write("\n\n## Notes\n\n")
        f.write("- If this file is empty, run evaluation and champion cycle first.\n")
        f.write("- Source files: `logs/eval_*.json`, `logs/champion_cycle_last_report.json`.\n")
        f.write("\n\n## Candidate Registry Snapshot\n\n")
        f.write(
            _md_table(
                candidate_rows,
                [
                    "candidate_dir",
                    "symbol",
                    "timeframe",
                    "candles",
                    "timesteps",
                    "feature_set_version",
                    "reward_version",
                ],
            )
        )

    bundle_md = os.path.join(OUT, "evidence_bundle.md")
    with open(bundle_md, "w", encoding="utf-8") as f:
        f.write("# Evidence Bundle\n\n")
        f.write(f"Generated (UTC): `{ts}`\n\n")
        f.write("## Included Artifacts\n\n")
        f.write("- `docs/results/walk_forward_results.csv`\n")
        f.write("- `docs/results/walk_forward_summary.md`\n")
        f.write("- `logs/champion_cycle_last_report.json`\n")
        f.write("- `models/registry/candidates/*/scorecard.json`\n")
        f.write("- `logs/audit_events.jsonl`\n")
        f.write("- `logs/trade_events.jsonl`\n")
        f.write("\n## Chart Inputs\n\n")
        f.write("- Use `walk_forward_results.csv` for charts in external BI tools or notebooks.\n")
        f.write("- Add dashboard screenshots in `docs/screenshots/`.\n")

    print(f"Evidence pack written to: {OUT}")


if __name__ == "__main__":
    build()
