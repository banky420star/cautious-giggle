"""
Backtester using the same TradingEnv profile as training.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from loguru import logger

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
    # Avoid Windows rotation file-lock contention when multiple backtests run in parallel.
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


def _make_env(
    df_pd: pd.DataFrame,
    initial_balance: float = 10000.0,
    reward_weights: dict | None = None,
    portfolio_feature_count: int | None = None,
    feature_version: str = ENGINEERED_V2,
    env_kwargs: dict | None = None,
    action_config: dict | None = None,
):
    _require_rl_stack()

    def _init():
        return TradingEnv(
            df_pd,
            initial_balance=initial_balance,
            reward_weights=reward_weights,
            portfolio_feature_count=portfolio_feature_count,
            feature_version=feature_version,
            action_config=action_config,
            **(env_kwargs or {}),
        )

    return DummyVecEnv([_init])


# M5 bars per year for annualising Sharpe (BTC trades 24/7, gold ~24/5; use 24/5).
_BARS_PER_YEAR = {"1m": 252 * 1440, "5m": 252 * 288, "15m": 252 * 96, "30m": 252 * 48, "1h": 252 * 24, "4h": 252 * 6}


def _annualized_sharpe(rets: np.ndarray, interval: str) -> float:
    rets = np.asarray(rets, dtype=np.float64)
    if rets.size < 2:
        return 0.0
    sd = float(np.std(rets))
    if sd <= 1e-12:
        return 0.0
    return float(np.mean(rets) / sd * np.sqrt(_BARS_PER_YEAR.get(str(interval), 252 * 288)))


def _equity_stats(equity: np.ndarray, interval: str) -> dict:
    equity = np.asarray(equity, dtype=np.float64)
    rets = np.diff(equity) / (equity[:-1] + 1e-12) if equity.size > 1 else np.zeros(1)
    peak = np.maximum.accumulate(equity)
    return {
        "total_return": float(equity[-1] / (equity[0] + 1e-12) - 1.0),
        "sharpe": _annualized_sharpe(rets, interval),
        "max_drawdown": float(np.max((peak - equity) / (peak + 1e-12))),
    }


def run_baselines(df: pd.DataFrame, interval: str, cost_bps: float, fast: int = 20, slow: int = 100) -> dict:
    """Simple strategies on the same bars and costs. A model must beat these to matter.

    ma_cross: long when EMA(fast) > EMA(slow), short otherwise, decided on the closed
    bar and held over the next bar. buy_hold: long the whole window.
    """
    close = pd.to_numeric(df["close"], errors="coerce").to_numpy(dtype=np.float64)
    if close.size < slow + 10:
        return {}
    bar_ret = np.diff(close) / (close[:-1] + 1e-12)
    ema_f = pd.Series(close).ewm(span=fast, adjust=False).mean().to_numpy()
    ema_s = pd.Series(close).ewm(span=slow, adjust=False).mean().to_numpy()
    pos = np.where(ema_f > ema_s, 1.0, -1.0)[:-1]
    pos[:slow] = 0.0
    turnover = np.abs(np.diff(np.concatenate([[0.0], pos])))
    net = pos * bar_ret - turnover * cost_bps / 10000.0
    ma_eq = 10000.0 * np.cumprod(1.0 + net)
    bh_eq = 10000.0 * np.cumprod(1.0 + bar_ret)
    bh_eq[0] -= 10000.0 * cost_bps / 10000.0
    return {
        "ma_cross": _equity_stats(np.concatenate([[10000.0], ma_eq]), interval),
        "buy_hold": _equity_stats(np.concatenate([[10000.0], bh_eq]), interval),
    }


def _load_model_meta(model_dir: str) -> dict:
    for name in ("metadata.json", "scorecard.json"):
        path = os.path.join(model_dir, name)
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f) or {}
            except Exception:
                return {}
    return {}


def _holdout_slice(df: pd.DataFrame, holdout_start: str | None) -> pd.DataFrame:
    """Keep only bars at/after the model's training holdout boundary (out-of-sample)."""
    if not holdout_start:
        return df
    try:
        if "time" in df.columns:
            t = pd.to_datetime(df["time"], utc=True, errors="coerce")
        elif isinstance(df.index, pd.DatetimeIndex):
            # broker data arrives time-indexed; without this the holdout cut was skipped and the
            # exam replayed the whole history, training data included
            idx = df.index if df.index.tz is not None else df.index.tz_localize("UTC")
            t = pd.Series(idx, index=df.index)
        else:
            return df
        cut = pd.to_datetime(holdout_start, utc=True)
        out = df.loc[t >= cut].reset_index(drop=True)
        return out
    except Exception:
        return df



def run_ppo_backtest(
    symbol: str,
    model_path: str,
    vecnorm_path: str,
    period: str = "120d",
    interval: str = "5m",
    initial_balance: float = 10000.0,
    max_steps: int | None = None,
    reward_weights: dict | None = None,
    override_holdout_start: str | None = None,
) -> dict | None:
    model_dir = os.path.dirname(model_path)
    meta = _load_model_meta(model_dir)
    feature_version = str(meta.get("feature_set_version", ENGINEERED_V2) or ENGINEERED_V2)
    env_kwargs = dict(meta.get("env_kwargs") or {})
    env_kwargs.pop("random_start", None)
    env_kwargs.pop("episode_length", None)
    action_config = meta.get("action_config") if isinstance(meta.get("action_config"), dict) else None
    df = fetch_training_data(symbol, period=period, interval=interval, strict=True, require_fresh=True)
    if df is None or df.empty or len(df) < 400:
        raise RuntimeError(f"Insufficient data for {symbol} (len={0 if df is None else len(df)})")
    holdout_start = override_holdout_start or meta.get("holdout_start")
    full_len = len(df)
    oos = _holdout_slice(df, holdout_start)
    if len(oos) < 400:
        raise RuntimeError(
            f"Insufficient out-of-sample data for {symbol}: {len(oos)} bars after holdout_start={holdout_start} "
            f"(fetched {full_len}); fetch a longer period"
        )
    if holdout_start and len(oos) == full_len:
        logger.warning(
            f"{symbol}: fetched data starts after holdout_start={holdout_start}; evaluating a partial holdout "
            "with no feature warm-up. Increase the evaluation period."
        )
    # Features are computed on the full fetch (warm-up history), the episode starts at
    # the first out-of-sample bar.
    start_index = full_len - len(oos)
    if start_index > 0:
        env_kwargs["start_index"] = int(start_index)

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
        env_kwargs=env_kwargs,
        action_config=action_config,
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
        eq = float(info0.get("equity", np.nan))
        cost = float(info0.get("cost", 0.0))
        pos = float(info0.get("position", 0.0))

        rc = info0.get("reward_components", {}) if isinstance(info0, dict) else {}
        for k in reward_component_sums:
            reward_component_sums[k] += float(rc.get(k, 0.0))

        equities.append(eq)
        costs.append(cost)
        positions.append(pos)
        rewards.append(float(np.asarray(reward, dtype=np.float64).reshape(-1)[0]))  # VecEnv returns shape (1,)

        if prev_eq is not None and prev_eq > 0:
            step_rets.append((eq - prev_eq) / prev_eq)
        prev_eq = eq

        steps += 1
        if max_steps and steps >= max_steps:
            break
        if bool(np.asarray(done).reshape(-1)[0]):
            break

    equity = np.array(equities, dtype=np.float64)
    if len(equity) < 3:
        raise RuntimeError(f"Insufficient backtest steps for {symbol}: {len(equity)}")

    rets = np.array(step_rets, dtype=np.float64) if step_rets else np.zeros(1)
    sharpe = _annualized_sharpe(rets, interval)

    peak = np.maximum.accumulate(equity)
    dd = (peak - equity) / (peak + 1e-12)
    max_dd = float(np.max(dd))

    total_return = float((equity[-1] / (equity[0] + 1e-12)) - 1.0)
    avg_cost = float(np.mean(costs)) if costs else 0.0
    pos_arr = np.array(positions, dtype=np.float64)
    turnover = float(np.mean(np.abs(np.diff(pos_arr)))) if len(pos_arr) > 2 else 0.0

    score = (total_return * 100.0) - (max_dd * 100.0 * 1.8) + (sharpe * 6.0) - (turnover * 2.0)
    cost_bps = float(env_kwargs.get("spread_bps", 2.0)) + float(env_kwargs.get("slippage_bps", 0.0)) + float(
        env_kwargs.get("commission_rate", 0.0002)
    ) * 10000.0
    baselines = run_baselines(oos, interval, cost_bps)

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
        "holdout_start": holdout_start,
        "out_of_sample": bool(holdout_start),
        "baselines": baselines,
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
    holdout_start: str | None = None,
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
                override_holdout_start=holdout_start,
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

    def _avg_baseline(name, key):
        vals = [float(x.get("baselines", {}).get(name, {}).get(key)) for x in per_symbol if x.get("baselines", {}).get(name)]
        return float(np.mean(vals)) if vals else None

    agg = {
        "baseline_avg_return": _avg_baseline("ma_cross", "total_return"),
        "baseline_avg_sharpe": _avg_baseline("ma_cross", "sharpe"),
        "buy_hold_avg_return": _avg_baseline("buy_hold", "total_return"),
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
