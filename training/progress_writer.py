"""Atomic training progress writer for API consumption."""
import json
import os
import time

LOGS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")


def update_training_progress(trainer_key, data):
    """Write progress for a single trainer to its own JSON file.

    Args:
        trainer_key: "lstm", "ppo", or "dreamer"
        data: dict with progress fields (running, symbol, epoch, loss, etc.)
    """
    os.makedirs(LOGS_DIR, exist_ok=True)
    path = os.path.join(LOGS_DIR, f"{trainer_key}_progress.json")
    payload = {**data, "updated_at": time.time()}
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)  # atomic on Windows
