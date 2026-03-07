import json
import os
from datetime import datetime

from loguru import logger


class ModelRegistry:
    """
    File-based model registry.
    Layout:
      models/
        registry/
          active.json
          candidates/<version>/

    active.json structure:
      {
        "champion": <path or null>,
        "canary": <path or null>,
        "symbols": {
          "EURUSDm": {"champion": <path or null>, "canary": <path or null>},
          ...
        }
      }
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
            self._write_active({"champion": None, "canary": None, "symbols": {}})

    def _normalize_active(self, payload: dict) -> dict:
        out = payload if isinstance(payload, dict) else {}
        if "champion" not in out:
            out["champion"] = None
        if "canary" not in out:
            out["canary"] = None
        if "symbols" not in out or not isinstance(out.get("symbols"), dict):
            out["symbols"] = {}
        return out

    def _read_active(self):
        with open(self.active_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return self._normalize_active(payload)

    def _write_active(self, payload: dict):
        with open(self.active_path, "w", encoding="utf-8") as f:
            json.dump(self._normalize_active(payload), f, indent=2)

    def _timestamp_version(self):
        return datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    def new_candidate_dir(self, tag: str = "candidate") -> str:
        ver = f"{tag}_{self._timestamp_version()}"
        path = os.path.join(self.candidates_dir, ver)
        os.makedirs(path, exist_ok=True)
        return path

    def get_symbol_active(self, symbol: str) -> dict:
        active = self._read_active()
        return dict(active.get("symbols", {}).get(symbol, {"champion": None, "canary": None}))

    def load_active_model(self, prefer_canary: bool = True, symbol: str | None = None) -> str | None:
        active = self._read_active()

        if symbol:
            s = active.get("symbols", {}).get(symbol, {})
            if prefer_canary and s.get("canary"):
                return s["canary"]
            if s.get("champion"):
                return s["champion"]

        if prefer_canary and active.get("canary"):
            return active["canary"]
        if active.get("champion"):
            return active["champion"]
        return None

    def set_canary(self, version_dir: str, symbol: str | None = None):
        active = self._read_active()
        if symbol:
            symbols = active.setdefault("symbols", {})
            cur = dict(symbols.get(symbol, {"champion": None, "canary": None}))
            cur["canary"] = version_dir
            symbols[symbol] = cur
            self._write_active(active)
            logger.warning(f"Canary set for {symbol}: {version_dir}")
            return

        active["canary"] = version_dir
        self._write_active(active)
        logger.warning(f"Canary set: {version_dir}")

    def promote_canary_to_champion(self, symbol: str | None = None):
        active = self._read_active()
        if symbol:
            symbols = active.setdefault("symbols", {})
            cur = dict(symbols.get(symbol, {"champion": None, "canary": None}))
            if not cur.get("canary"):
                raise RuntimeError(f"No canary to promote for {symbol}.")
            cur["champion"] = cur["canary"]
            cur["canary"] = None
            symbols[symbol] = cur
            self._write_active(active)
            logger.success(f"Promoted {symbol} champion: {cur['champion']}")
            return

        if not active.get("canary"):
            raise RuntimeError("No canary to promote.")
        active["champion"] = active["canary"]
        active["canary"] = None
        self._write_active(active)
        logger.success(f"Promoted champion: {active['champion']}")

    def clear_canary(self, symbol: str | None = None):
        active = self._read_active()
        if symbol:
            symbols = active.setdefault("symbols", {})
            cur = dict(symbols.get(symbol, {"champion": None, "canary": None}))
            cur["canary"] = None
            symbols[symbol] = cur
            self._write_active(active)
            logger.warning(f"Canary cleared for {symbol}")
            return

        active["canary"] = None
        self._write_active(active)
        logger.warning("Canary cleared")

    def rollback_to_champion(self, symbol: str | None = None):
        self.clear_canary(symbol=symbol)

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
