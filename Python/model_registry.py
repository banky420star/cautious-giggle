import json
import os
from datetime import datetime, timezone

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
        if "canary_policy" not in out or not isinstance(out.get("canary_policy"), dict):
            out["canary_policy"] = {}
        if "canary_state" not in out or not isinstance(out.get("canary_state"), dict):
            out["canary_state"] = {}
        if "symbols" not in out or not isinstance(out.get("symbols"), dict):
            out["symbols"] = {}
        for sym, cfg in list(out["symbols"].items()):
            if not isinstance(cfg, dict):
                out["symbols"][sym] = {"champion": None, "canary": None, "canary_policy": {}, "canary_state": {}}
                continue
            if "champion" not in cfg:
                cfg["champion"] = None
            if "canary" not in cfg:
                cfg["canary"] = None
            if "canary_policy" not in cfg or not isinstance(cfg.get("canary_policy"), dict):
                cfg["canary_policy"] = {}
            if "canary_state" not in cfg or not isinstance(cfg.get("canary_state"), dict):
                cfg["canary_state"] = {}
            out["symbols"][sym] = cfg
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
        return dict(
            active.get(
                "symbols",
                {},
            ).get(symbol, {"champion": None, "canary": None, "canary_policy": {}, "canary_state": {}})
        )

    def _default_canary_policy(self) -> dict:
        return {
            "min_trades": 10,
            "min_realized_pnl": 0.0,
            "max_drawdown": 0.12,
            "min_runtime_minutes": 30,
        }

    def _merge_canary_policy(self, policy: dict | None) -> dict:
        out = self._default_canary_policy()
        if isinstance(policy, dict):
            for k in out.keys():
                if k in policy:
                    out[k] = policy[k]
        return out

    def _canary_passes(self, policy: dict, state: dict) -> tuple[bool, str]:
        trades = int(state.get("trades", 0))
        realized = float(state.get("realized_pnl", 0.0))
        dd = float(state.get("drawdown", 0.0))
        runtime = float(state.get("runtime_minutes", 0.0))

        if trades < int(policy.get("min_trades", 10)):
            return False, f"trades {trades} < min_trades {int(policy.get('min_trades', 10))}"
        if realized < float(policy.get("min_realized_pnl", 0.0)):
            return False, f"realized_pnl {realized:.2f} < min_realized_pnl {float(policy.get('min_realized_pnl', 0.0)):.2f}"
        if dd > float(policy.get("max_drawdown", 0.12)):
            return False, f"drawdown {dd:.4f} > max_drawdown {float(policy.get('max_drawdown', 0.12)):.4f}"
        if runtime < float(policy.get("min_runtime_minutes", 30)):
            return False, f"runtime_minutes {runtime:.1f} < min_runtime_minutes {float(policy.get('min_runtime_minutes', 30)):.1f}"
        return True, "ok"

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

    def set_canary(self, version_dir: str, symbol: str | None = None, policy: dict | None = None):
        active = self._read_active()
        merged = self._merge_canary_policy(policy)
        if symbol:
            symbols = active.setdefault("symbols", {})
            cur = dict(symbols.get(symbol, {"champion": None, "canary": None, "canary_policy": {}, "canary_state": {}}))
            cur["canary"] = version_dir
            cur["canary_policy"] = merged
            cur["canary_state"] = {"passed": False, "reason": "no_metrics"}
            symbols[symbol] = cur
            self._write_active(active)
            logger.warning(f"Canary set for {symbol}: {version_dir}")
            return

        active["canary"] = version_dir
        active["canary_policy"] = merged
        active["canary_state"] = {"passed": False, "reason": "no_metrics"}
        self._write_active(active)
        logger.warning(f"Canary set: {version_dir}")

    def update_canary_metrics(
        self,
        trades: int,
        realized_pnl: float,
        drawdown: float,
        runtime_minutes: float,
        symbol: str | None = None,
    ) -> dict:
        active = self._read_active()
        state = {
            "trades": int(trades),
            "realized_pnl": float(realized_pnl),
            "drawdown": float(drawdown),
            "runtime_minutes": float(runtime_minutes),
            "checked_at": datetime.now(timezone.utc).isoformat(),
        }

        if symbol:
            symbols = active.setdefault("symbols", {})
            cur = dict(symbols.get(symbol, {"champion": None, "canary": None, "canary_policy": {}, "canary_state": {}}))
            policy = self._merge_canary_policy(cur.get("canary_policy"))
            passed, reason = self._canary_passes(policy, state)
            state["passed"] = bool(passed)
            state["reason"] = reason
            cur["canary_policy"] = policy
            cur["canary_state"] = state
            symbols[symbol] = cur
            self._write_active(active)
            return state

        policy = self._merge_canary_policy(active.get("canary_policy"))
        passed, reason = self._canary_passes(policy, state)
        state["passed"] = bool(passed)
        state["reason"] = reason
        active["canary_policy"] = policy
        active["canary_state"] = state
        self._write_active(active)
        return state

    def can_promote_canary(self, symbol: str | None = None) -> tuple[bool, str]:
        active = self._read_active()
        if symbol:
            cur = dict(active.get("symbols", {}).get(symbol, {"champion": None, "canary": None, "canary_state": {}}))
            if not cur.get("canary"):
                return False, f"No canary to promote for {symbol}."
            state = cur.get("canary_state", {}) if isinstance(cur.get("canary_state"), dict) else {}
            if bool(state.get("passed", False)):
                return True, "ok"
            return False, str(state.get("reason", "canary survival checks not satisfied"))

        if not active.get("canary"):
            return False, "No canary to promote."
        state = active.get("canary_state", {}) if isinstance(active.get("canary_state"), dict) else {}
        if bool(state.get("passed", False)):
            return True, "ok"
        return False, str(state.get("reason", "canary survival checks not satisfied"))

    def promote_canary_to_champion(self, symbol: str | None = None, force: bool = False):
        active = self._read_active()
        if symbol:
            symbols = active.setdefault("symbols", {})
            cur = dict(symbols.get(symbol, {"champion": None, "canary": None, "canary_policy": {}, "canary_state": {}}))
            if not cur.get("canary"):
                raise RuntimeError(f"No canary to promote for {symbol}.")
            if not force:
                ok, reason = self.can_promote_canary(symbol=symbol)
                if not ok:
                    raise RuntimeError(f"Canary promotion blocked for {symbol}: {reason}")
            cur["champion"] = cur["canary"]
            cur["canary"] = None
            cur["canary_state"] = {}
            symbols[symbol] = cur
            self._write_active(active)
            logger.success(f"Promoted {symbol} champion: {cur['champion']}")
            return

        if not active.get("canary"):
            raise RuntimeError("No canary to promote.")
        if not force:
            ok, reason = self.can_promote_canary()
            if not ok:
                raise RuntimeError(f"Canary promotion blocked: {reason}")
        active["champion"] = active["canary"]
        active["canary"] = None
        active["canary_state"] = {}
        self._write_active(active)
        logger.success(f"Promoted champion: {active['champion']}")

    def clear_canary(self, symbol: str | None = None):
        active = self._read_active()
        if symbol:
            symbols = active.setdefault("symbols", {})
            cur = dict(symbols.get(symbol, {"champion": None, "canary": None, "canary_policy": {}, "canary_state": {}}))
            cur["canary"] = None
            cur["canary_state"] = {}
            symbols[symbol] = cur
            self._write_active(active)
            logger.warning(f"Canary cleared for {symbol}")
            return

        active["canary"] = None
        active["canary_state"] = {}
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
