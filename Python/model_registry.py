import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone

from loguru import logger

from Python.config_utils import load_project_config

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
INTEGRITY_TARGETS = {"model": "ppo_trading.zip", "vec_normalize": "vec_normalize.pkl"}


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
          "EURUSDm": {"champion": <path or null>, "canary": <path or null>, "canary_policy": {...}, "canary_state": {...}},
          ...
        },
        "registry_metadata": {
          "git_commit": "...",
          "champion_metadata": {...},
          "canary_metadata": {...}
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

        self.registry_config = self._load_registry_config()
        self._canary_policy_cfg = self.registry_config.get("canary_policy", {}) or {}
        self._canary_default_overrides = self._canary_policy_cfg.get("default", {}) or {}
        self._canary_symbol_overrides = self._canary_policy_cfg.get("per_symbol", {}) or {}

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

    def _load_registry_config(self) -> dict:
        try:
            cfg = load_project_config(PROJECT_ROOT, live_mode=False)
            return cfg.get("registry", {}) or {}
        except Exception:
            return {}

    def _write_active(self, payload: dict):
        normalized = self._normalize_active(payload)
        normalized["registry_metadata"] = self._build_registry_metadata(normalized)

        tmp = tempfile.NamedTemporaryFile(mode="w", delete=False, dir=self.root, suffix=".json", encoding="utf-8")
        try:
            json.dump(normalized, tmp, indent=2)
        finally:
            tmp.close()

        if os.path.exists(self.active_path):
            backup_path = f"{self.active_path}.bak"
            try:
                shutil.copy2(self.active_path, backup_path)
            except Exception:
                logger.warning("Unable to backup active registry file.")

        shutil.move(tmp.name, self.active_path)

    def _build_registry_metadata(self, active: dict) -> dict:
        meta = {
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        commit = self._current_git_commit_hash()
        if commit:
            meta["git_commit"] = commit

        for role in ("champion", "canary"):
            path = active.get(role)
            meta[f"{role}_metadata"] = self._gather_candidate_metadata(path)
        return meta

    def _gather_candidate_metadata(self, candidate_dir: str | None) -> dict:
        if not candidate_dir or not os.path.isdir(candidate_dir):
            return {}
        meta = dict(self.read_metadata(candidate_dir) or {})
        scorecard = self._read_scorecard(candidate_dir)
        if scorecard:
            meta["scorecard"] = scorecard
            if "timesteps" in scorecard:
                meta.setdefault("training_timesteps", int(scorecard.get("timesteps") or 0))
        integrity = self._integrity_snapshot(candidate_dir)
        if integrity:
            meta["integrity"] = integrity
        return meta

    def _read_scorecard(self, candidate_dir: str | None) -> dict:
        if not candidate_dir:
            return {}
        scorecard_path = os.path.join(candidate_dir, "scorecard.json")
        if not os.path.exists(scorecard_path):
            return {}
        try:
            with open(scorecard_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
                return payload if isinstance(payload, dict) else {}
        except Exception:
            return {}

    def _integrity_snapshot(self, candidate_dir: str | None) -> dict:
        targets = INTEGRITY_TARGETS
        snapshot = {}
        if not candidate_dir:
            return snapshot
        for label, fname in targets.items():
            path = os.path.join(candidate_dir, fname)
            if os.path.exists(path):
                snapshot[label] = self._file_hash(path)
        return snapshot

    def _clear_active_entry(self, active: dict, role_key: str, symbol: str | None = None):
        normalized_role = "canary" if "canary" in role_key else "champion"
        if symbol:
            symbols = active.setdefault("symbols", {})
            cur = dict(symbols.get(symbol, {"champion": None, "canary": None, "canary_policy": {}, "canary_state": {}}))
            cur[normalized_role] = None
            if normalized_role == "canary":
                cur["canary_state"] = {}
            symbols[symbol] = cur
            return
        active[normalized_role] = None

    def _validate_candidate_integrity(self, candidate_dir: str | None) -> bool:
        if not candidate_dir or not os.path.isdir(candidate_dir):
            return False
        recorded = self.read_metadata(candidate_dir).get("integrity")
        if not isinstance(recorded, dict) or not recorded:
            return True
        for label, expected in recorded.items():
            target = INTEGRITY_TARGETS.get(label)
            if not target or not expected:
                continue
            path = os.path.join(candidate_dir, target)
            if not os.path.exists(path):
                return False
            actual = self._file_hash(path)
            if not actual or str(actual) != str(expected):
                return False
        return True

    def _file_hash(self, path: str) -> str:
        hash_obj = hashlib.sha256()
        try:
            with open(path, "rb") as f:
                while chunk := f.read(8192):
                    hash_obj.update(chunk)
        except Exception:
            return ""
        return hash_obj.hexdigest()

    def _current_git_commit_hash(self) -> str | None:
        try:
            output = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True)
            return output.strip()
        except Exception:
            return None

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
        base = {
            "min_trades": 10,
            "min_realized_pnl": 0.0,
            "max_drawdown": 0.12,
            "min_runtime_minutes": 30,
        }
        base.update(self._canary_default_overrides)
        return base

    def _policy_for_symbol(self, symbol: str | None) -> dict:
        policy = dict(self._default_canary_policy())
        if symbol and symbol in self._canary_symbol_overrides:
            overrides = self._canary_symbol_overrides.get(symbol) or {}
            policy.update(overrides)
        return policy

    def _merge_canary_policy(self, policy: dict | None, symbol: str | None = None) -> dict:
        out = self._policy_for_symbol(symbol)
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
        updated = False

        def resolve_path(path: str | None, role: str, sym: str | None = None) -> str | None:
            nonlocal updated
            if not path:
                return None
            if self._validate_candidate_integrity(path):
                return path
            logger.error("Candidate integrity mismatch for %s (%s). Clearing entry.", role, path)
            self._clear_active_entry(active, role, sym)
            updated = True
            return None

        symbols = active.get("symbols", {})
        symbol_entry = symbols.get(symbol, {}) if symbol else {}

        if symbol:
            canary_path = symbol_entry.get("canary")
            champion_path = symbol_entry.get("champion")
            if prefer_canary:
                resolved = resolve_path(canary_path, "symbol_canary", symbol)
                if resolved:
                    if updated:
                        self._write_active(active)
                        updated = False
                    return resolved
            resolved = resolve_path(champion_path, "symbol_champion", symbol)
            if resolved:
                if updated:
                    self._write_active(active)
                    updated = False
                return resolved

        if prefer_canary:
            resolved = resolve_path(active.get("canary"), "canary")
            if resolved:
                if updated:
                    self._write_active(active)
                    updated = False
                return resolved
        resolved = resolve_path(active.get("champion"), "champion")
        if resolved:
            if updated:
                self._write_active(active)
                updated = False
            return resolved

        if updated:
            self._write_active(active)
        return None

    def set_canary(self, version_dir: str, symbol: str | None = None, policy: dict | None = None):
        active = self._read_active()
        merged = self._merge_canary_policy(policy, symbol)
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
            policy = self._merge_canary_policy(cur.get("canary_policy"), symbol)
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

    def register_candidate(self, candidate_dir: str, metadata: dict | None = None):
        meta = dict(self.read_metadata(candidate_dir) or {})
        meta.update(metadata or {})
        integrity = self._integrity_snapshot(candidate_dir)
        if integrity:
            meta["integrity"] = integrity
        scorecard = self._read_scorecard(candidate_dir)
        if scorecard:
            meta["scorecard"] = scorecard
            meta.setdefault("training_timesteps", int(scorecard.get("timesteps", 0) or 0))
        meta["registered_at"] = datetime.now(timezone.utc).isoformat()
        meta_path = os.path.join(candidate_dir, "metadata.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        logger.info(f"Candidate registered: {candidate_dir}")

    def update_metadata(self, candidate_dir: str, patch: dict):
        meta = self.read_metadata(candidate_dir)
        if not isinstance(meta, dict):
            meta = {}
        meta.update(patch or {})
        self.register_candidate(candidate_dir, meta)

    def read_metadata(self, version_dir: str) -> dict:
        meta_path = os.path.join(version_dir, "metadata.json")
        if not os.path.exists(meta_path):
            return {}
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
