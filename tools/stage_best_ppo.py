import os
import json
import shutil
import datetime
from Python.model_registry import ModelRegistry

def find_one(root, filename):
    matches = []
    for dirpath, _, files in os.walk(root):
        if filename in files:
            matches.append(os.path.join(dirpath, filename))
    # Sort by modification time, newest first
    matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return matches[0] if matches else None

def main():
    # 1. Find SB3 best model
    # We look for best_model.zip specifically from EvalCallback
    best_model = find_one(".", "best_model.zip")
    if not best_model:
        # Fallback to ppo_trading.zip if best_model.zip isn't there
        best_model = find_one(".", "ppo_trading.zip")
        
    if not best_model:
        print("❌ Could not find best_model.zip or ppo_trading.zip anywhere under repo root.")
        return

    # 2. Find the vec stats nearest to it (same folder priority)
    best_dir = os.path.dirname(best_model)
    vec_path = os.path.join(best_dir, "vec_normalize.pkl")
    
    if not os.path.exists(vec_path):
        # Fallback: newest vec_normalize.pkl in repo
        print("⚠️  No vec_normalize.pkl in best model folder. Searching repo for newest stats...")
        vec_path = find_one(".", "vec_normalize.pkl")

    if not vec_path:
        print("❌ Could not find vec_normalize.pkl. Staging aborted to avoid normalization mismatch.")
        return

    # 3. Stage into ModelRegistry candidates
    registry = ModelRegistry()
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    cand_dir = os.path.join(registry.candidates_dir, ts)
    os.makedirs(cand_dir, exist_ok=True)

    # Standardize names for the registry
    shutil.copy2(best_model, os.path.join(cand_dir, "ppo_trading.zip"))
    shutil.copy2(vec_path, os.path.join(cand_dir, "vec_normalize.pkl"))

    scorecard = {
        "type": "ppo",
        "source_best_model": os.path.abspath(best_model),
        "source_vecnormalize": os.path.abspath(vec_path),
        "created_at": datetime.datetime.now().isoformat(),
        "status": "manually_staged"
    }
    
    with open(os.path.join(cand_dir, "scorecard.json"), "w", encoding="utf-8") as f:
        json.dump(scorecard, f, indent=2)

    print(f"✅ STAGED_CANDIDATE_DIR={cand_dir}")
    print(f"   Model:   {best_model}")
    print(f"   VecNorm: {vec_path}")

if __name__ == "__main__":
    main()
