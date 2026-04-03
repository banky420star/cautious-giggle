"""
Backtester using the same TradingEnv profile as training.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from loguru import logger

# Compatibility shim for PPO models pickled with numpy 2.x
try:
    import numpy._core.numeric  # noqa: F401
except (ModuleNotFoundError, ImportError):
    import types as _types
    import numpy.core.numeric as _ncn
    _core_mod = _types.ModuleType("numpy._core")
    _core_mod.numeric = _ncn
    sys.modules.setdefault("numpy._core", _core_mod)
    sys.modules.setdefault("numpy._core.numeric", _ncn)


class _PPOProxy:
    @staticmethod
    def load(*args, **kwargs):
        _require_rl_stack()
        return globals()["PPO"].load(*args, **kwargs)


PPO = _PPOProxy
DummyVecEnv = None


class _VecNormalizeProxy:
    @staticmethod
    def load(*args, **kwargs):
        _require_rl_stack()
        return globals()["VecNormalize"].load(*args, **kwargs)


VecNormalize = _VecNormalizeProxy
_RL_IMPORT_ERROR = None

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Python.config_utils import DEFAULT_TRADING_SYMBOLS
from Python.data_feed import fetch_training_data
from Python.feature_pipeline import ENGINEERED_V2, feature_count_for_version
from drl.trading_env import TradingEnv

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
try:
    logger.add(
        os.path.join(LOG_DIR, "backtester.log"),
        level="INFO",
        enqueue=True,
        mode="a",
        backtrace=False,
        diagnose=False,
    )
except Exception:
    pass


def _require_rl_stack() -> None:
    global PPO, DummyVecEnv, VecNormalize, _RL_IMPORT_ERROR
    if PPO is not _PPOProxy and DummyVecEnv is not None and VecNormalize is not _VecNormalizeProxy:
        return
    try:
        from stable_baselines3 import PPO as _PPO
        from stable_baselines3.common.vec_env import DummyVecEnv as _DummyVecEnv, VecNormalize as _VecNormalize
    except Exception as exc:
        _RL_IMPORT_ERROR = exc
        raise RuntimeError(
            "stable-baselines3 with a working torch runtime is required for PPO backtests."
        ) from exc
    PPO = _PPO
    DummyVecEnv = _DummyVecEnv
    VecNormalize = _VecNormalize


def _normalize_interval(interval: str | None) -> str:
    if not interval:
        return "5m"
    m = str(interval).strip().lower()
    if m.startswith("m") and m[1:].isdigit():
        return f"{m[1:]}m"
    if m.startswith("h") and m[1:].isdigit():
        return f"{m[1:]}h"
    return m


def _scalarize(value, default: float = np.nan) -> float:
    if value is None:
        return float(default)
    arr = np.asarray(value)
    if arr.size == 0:
        return float(default)
    return float(arr.reshape(-1)[0])


def _done_to_bool(done) -> bool:
    if isinstance(done, (list, tuple)):
        return any(_done_to_bool(item) for item in done)
    arr = np.asarray(done)
    if arr.size == 0:
        return False
    return bool(arr.reshape(-1)[0])


def _make_env(
    df_pd: pd.DataFrame,
    initial_balance: float = 10000.0,
    reward_weights: dict | None = None,
    portfolio_feature_count: int | None = None,
    feature_version: str = ENGINEERED_V2,
):
    _require_rl_stack()

    def _init():
        return TradingEnv(
            df_pd,
            initial_balance=initial_balance,
            reward_weights=reward_weights,
            portfolio_feature_count=portfolio_feature_count,
            feature_version=feature_version,
        )

    return DummyVecEnv([_init])



def run_ppo_backtest(
    symbol: str,
    model_path: str,
    vecnorm_path: str,
    period: str = "120d",
    interval: str = "5m",
    initial_balance: float = 10000.0,
    max_steps: int | None = None,
    reward_weights: dict | None = None,
) -> dict | None:
    metadata_path = os.path.join(model_dir := os.path.dirname(model_path), "metadata.json")
    feature_version = ENGINEERED_V2
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                meta = json.load(f) or {}
            feature_version = str(meta.get("feature_set_version", ENGINEERED_V2) or ENGINEERED_V2)
        except Exception:
            feature_version = ENGINEERED_V2
    df = fetch_training_data(symbol, period=period, interval=interval, strict=True, require_fresh=True)
    if df is None or df.empty or len(df) < 400:
        raise RuntimeError(f"Insufficient data for {symbol} (len={0 if df is None else len(df)})")

    if not os.path.exists(model_path):
        raise RuntimeError(f"Missing model file: {model_path}")
    model = PPO.load(model_path, device="cpu")
    expected_obs_dim = None
    obs_space = getattr(model, "observation_space", None)
    if obs_space is not None and getattr(obs_space, "shape", None):
        expected_obs_dim = int(np.prod(obs_space.shape))
    portfolio_feature_count = (
        TradingEnv.infer_portfolio_feature_count(
            expected_obs_dim,
            n_features=feature_count_for_version(feature_version),
        )
        if expected_obs_dim is not None
        else None
    )
    env = _make_env(
        df,
        initial_balance=initial_balance,
        reward_weights=reward_weights,
        portfolio_feature_count=portfolio_feature_count,
        feature_version=feature_version,
    )
    if not os.path.exists(vecnorm_path):
        raise RuntimeError(f"Missing vecnorm file: {vecnorm_path}")

    env = VecNormalize.load(vecnorm_path, env)
    env.training = False
    env.norm_reward = False

    obs = env.reset()
    equities, costs, positions, rewards, step_rets = [], [], [], [], []
    reward_component_sums = {
        "growth": 0.0,
        "payoff": 0.0,
        "sharpe_bonus": 0.0,
        "drawdown_penalty": 0.0,
        "cost_penalty": 0.0,
        "churn_penalty": 0.0,
    }

    steps = 0
    prev_eq = None

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        info0 = info[0] if isinstance(info, (list, tuple)) else info
        if not isinstance(info0, dict):
            raise RuntimeError(f"Invalid vec env info payload for {symbol}: {type(info0).__name__}")
        eq = _scalarize(info0.get("equity"), np.nan)
        cost = _scalarize(info0.get("cost"), 0.0)
        pos = _scalarize(info0.get("position"), 0.0)

        rc = info0.get("reward_components", {})
        for k in reward_component_sums:
            reward_component_sums[k] += _scalarize(rc.get(k), 0.0)

        equities.append(eq)
        costs.append(cost)
        positions.append(pos)
        rewards.append(_scalarize(reward, 0.0))

        if prev_eq is not None and prev_eq > 0:
            step_rets.append((eq - prev_eq) / prev_eq)
        prev_eq = eq

        steps += 1
        if max_steps and steps >= max_steps:
            break
        if _done_to_bool(done):
            break

    equity = np.array(equities, dtype=np.float64)
    if len(equity) < 3:
        raise RuntimeError(f"Insufficient backtest steps for {symbol}: {len(equity)}")

    rets = np.array(step_rets, dtype=np.float64) if step_rets else np.zeros(1)
    vol = float(np.std(rets) + 1e-12)
    sharpe = float(np.mean(rets) / vol) if vol > 0 else 0.0

    peak = np.maximum.accumulate(equity)
    dd = (peak - equity) / (peak + 1e-12)
    max_dd = float(np.max(dd))

    total_return = float((equity[-1] / (equity[0] + 1e-12)) - 1.0)
    avg_cost = float(np.mean(costs)) if costs else 0.0
    pos_arr = np.array(positions, dtype=np.float64)
    turnover = float(np.mean(np.abs(np.diff(pos_arr)))) if len(pos_arr) > 2 else 0.0

    score = (total_return * 100.0) - (max_dd * 100.0 * 1.8) + (sharpe * 6.0) - (turnover * 2.0)

    n = max(1, steps)
    result = {
        "symbol": symbol,
        "period": period,
        "interval": interval,
        "candles": int(len(df)),
        "total_return": float(total_return),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_dd),
        "avg_cost": float(avg_cost),
        "turnover": float(turnover),
        "steps": int(steps),
        "final_equity": float(equity[-1]),
        "score": float(score),
        "reward_component_avg": {k: float(v / n) for k, v in reward_component_sums.items()},
    }

    logger.info(
        f"BACKTEST {symbol} | tf={interval} ret={total_return:.2%} sharpe={sharpe:.2f} "
        f"maxDD={max_dd:.2%} score={score:.2f} steps={steps}"
    )
    return result


def run_multi(
    symbols: list[str],
    model_dir: str,
    period: str = "120d",
    interval: str = "5m",
    reward_weights: dict | None = None,
) -> dict:
    model_path = os.path.join(model_dir, "ppo_trading.zip")
    vec_path = os.path.join(model_dir, "vec_normalize.pkl")

    per_symbol = []
    errors = []
    tf = _normalize_interval(interval)
    for sym in symbols:
        try:
            r = run_ppo_backtest(
                sym,
                model_path,
                vec_path,
                period=period,
                interval=tf,
                reward_weights=reward_weights,
            )
            if r:
                per_symbol.append(r)
        except Exception as exc:
            msg = f"{sym}: {exc}"
            errors.append(msg)
            logger.error(f"BACKTEST_FAIL {msg}")

    if not per_symbol:
        return {"error": "No valid backtests", "errors": errors}

    scores = [x["score"] for x in per_symbol]
    rets = [x["total_return"] for x in per_symbol]
    dds = [x["max_drawdown"] for x in per_symbol]
    sharpes = [x["sharpe"] for x in per_symbol]

    agg = {
        "symbols": [x["symbol"] for x in per_symbol],
        "period": period,
        "interval": tf,
        "avg_score": float(np.mean(scores)),
        "avg_return": float(np.mean(rets)),
        "worst_drawdown": float(np.max(dds)),
        "avg_sharpe": float(np.mean(sharpes)),
        "per_symbol": per_symbol,
        "errors": errors,
    }
    return agg


if __name__ == "__main__":
    symbols = list(DEFAULT_TRADING_SYMBOLS)
    md = sys.argv[1] if len(sys.argv) > 1 else os.path.join("models", "registry", "champion")
    report = run_multi(symbols, md, period="120d", interval="5m")
    print(json.dumps(report, indent=2))
