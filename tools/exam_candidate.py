"""Give one staged candidate the normal exam (same gates as the champion cycle) without trialling it.
    PYTHONPATH=. .venv/bin/python tools/exam_candidate.py <candidate_dir> <SYMBOL>"""
import json
import sys

from Python.config_utils import load_project_config
from Python.model_evaluator import evaluate_candidate_vs_champion

cand, sym = sys.argv[1], sys.argv[2]
cfg = load_project_config(live_mode=False)
gates = (cfg.get("evaluation") or {}).get("gates") if isinstance(cfg.get("evaluation"), dict) else None
r = evaluate_candidate_vs_champion(cand, None, [sym], period=str((cfg.get("drl") or {}).get("eval_period", "120d")), gates=gates)
print("EXAM", sym, cand.rsplit("/", 1)[-1], json.dumps({k: r.get(k) for k in ("passes_thresholds", "wins", "error")}),
      json.dumps(r.get("candidate", r.get("per_symbol", r)))[:600], flush=True)
