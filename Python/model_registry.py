import os
import json
import shutil
from datetime import datetime
from loguru import logger


class ModelRegistry:
    """
    File-based model registry.
    Layout:
      models/
        registry/
          active.json
          champion/<version>/
          canary/<version>/
          candidates/<version>/
    """

    def __init__(self, root=None):
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.root = root or os.path.join(base, "models", "registry")
        os.makedirs(self.root, exist_ok=True)

        self.active_path = os.path.join(self.root, "active.json")
        self.champion_dir = os.path.join(self.root, "champion")
        self.canary_dir = os.path.join(self.root, "canary")
        self.candidates_dir = os.path.join(self.root, "candidates")

        for d in (self.champion_dir, self.canary_dir, self.candidates_dir):
            os.makedirs(d, exist_ok=True)

        if not os.path.exists(self.active_path):
            self._write_active({"champion": None, "canary": None})

    def _read_active(self):
        with open(self.active_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _write_active(self, payload: dict):
        with open(self.active_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def _timestamp_version(self):
        return datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    def new_candidate_dir(self, tag: str = "candidate") -> str:
        ver = f"{tag}_{self._timestamp_version()}"
        path = os.path.join(self.candidates_dir, ver)
        os.makedirs(path, exist_ok=True)
        return path

    def load_active_model(self, prefer_canary: bool = True) -> str | None:
        active = self._read_active()
        if prefer_canary and active.get("canary"):
            return active["canary"]
        if active.get("champion"):
            return active["champion"]
        return None

    def set_canary(self, version_dir: str):
        active = self._read_active()
        active["canary"] = version_dir
        self._write_active(active)
        logger.warning(f"🟡 Canary set: {version_dir}")

    def promote_canary_to_champion(self):
        active = self._read_active()
        if not active.get("canary"):
            raise RuntimeError("No canary to promote.")
        active["champion"] = active["canary"]
        active["canary"] = None
        self._write_active(active)
        logger.success(f"🟢 Promoted to champion: {active['champion']}")

    def clear_canary(self):
        active = self._read_active()
        active["canary"] = None
        self._write_active(active)
        logger.warning("🟠 Canary cleared")

    def rollback_to_champion(self):
        self.clear_canary()

    def register_candidate(self, candidate_dir: str, metadata: dict):
        meta_path = os.path.join(candidate_dir, "metadata.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Candidate registered: {candidate_dir}")

    def read_metadata(self, version_dir: str) -> dict:
        meta_path = os.path.join(version_dir, "metadata.json")
        if not os.path.exists(meta_path):
            return {}
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)

    # Compatibility wrappers used by trainers
    def save_candidate(self, model_state, metrics: dict, model_type: str = "lstm") -> str:
        import torch

        candidate_dir = self.new_candidate_dir(tag=model_type)

        if model_type.lower() == "lstm":
            model_path = os.path.join(candidate_dir, "lstm_model.pth")
            torch.save(model_state, model_path)
        elif model_type.lower() == "ppo":
            # for PPO, training pipeline usually copies zip + vec directly
            pass

        self.register_candidate(candidate_dir, {
            "model_type": model_type,
            "metrics": metrics,
            "created_at": datetime.utcnow().isoformat(),
        })
        return candidate_dir

    def evaluate_and_stage_canary(self, candidate_dir: str) -> bool:
        """Simple fallback: stage as canary if no champion exists yet."""
        active = self._read_active()
        if not active.get("champion"):
            self.set_canary(candidate_dir)
            return True
        return False
