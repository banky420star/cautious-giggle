import json
import os

from Python.model_registry import ModelRegistry


def _candidate(root: str, name: str) -> str:
    path = os.path.join(root, "candidates", name)
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "ppo_trading.zip"), "wb") as f:
        f.write(b"model")
    with open(os.path.join(path, "vec_normalize.pkl"), "wb") as f:
        f.write(b"vec")
    with open(os.path.join(path, "scorecard.json"), "w", encoding="utf-8") as f:
        json.dump({"timesteps": 1000}, f)
    return path


def test_registry_tracks_global_champion_history(tmp_path):
    registry = ModelRegistry(root=str(tmp_path), registry_config={"ensemble": {"history_limit": 3}, "canary_policy": {}})
    cand1 = _candidate(str(tmp_path), "c1")
    cand2 = _candidate(str(tmp_path), "c2")

    registry.set_canary(cand1)
    registry.update_canary_metrics(99, 10.0, 0.01, 60.0)
    registry.promote_canary_to_champion()

    registry.set_canary(cand2)
    registry.update_canary_metrics(99, 12.0, 0.01, 60.0)
    registry.promote_canary_to_champion()

    history = registry.get_recent_champions()
    assert len(history) == 2
    assert history[0]["path"] == cand2
    assert history[1]["path"] == cand1
