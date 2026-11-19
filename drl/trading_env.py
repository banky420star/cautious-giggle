import datetime
import json
import os

import gymnasium as gym
import numpy as np
import polars as pl
import pandas as pd
from gymnasium import spaces
from Python.feature_pipeline import ENGINEERED_V2, build_env_feature_matrix, feature_count_for_version

ENGINEERED_FEATURE_COUNT = 21
DEFAULT_PORTFOLIO_FEATURE_COUNT = 3
MAX_PORTFOLIO_FEATURE_COUNT = 6


class TradingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df=None,
        initial_balance: float = 10000.0,
        commission_rate: float = 0.0002,
        spread_bps: float = 2.0,
        max_drawdown: float = 0.15,
        window_size: int = 100,
        max_leverage: float = 1.0,
        reward_weights: dict | None = None,
        trade_memory: dict | None = None,
        portfolio_feature_count: int | None = None,
        feature_version: str = ENGINEERED_V2,
        action_config: dict | None = None,
        symbol: str | None = None,
        execution_mode: str = "trade",
        slippage_bps: float = 1.0,
        random_start: bool = False,
        episode_length: int | None = None,
        sl_atr_mult: float = 1.4,
        tp_atr_mult: float = 2.2,
        min_rebalance: float = 0.05,
        start_index: int | None = None,
        mf_symbol: str | None = None,
        mf_last: float | None = None,
        trend_gate: bool = False,
        compact_ablate: bool = False,
    ):
        super().__init__()
        from Python.compact_features import COMPACT_V3, TRADE_STATE_COUNT
        self._compact = str(feature_version or "").lower() == COMPACT_V3
        if self._compact:
            window_size = 1                       # compact_v3 rows already carry their own short history
            if portfolio_feature_count is None:
                portfolio_feature_count = TRADE_STATE_COUNT
        self.mf_symbol = mf_symbol
        self.mf_last = mf_last
        # live trend gate inside training (compact_v3): no new risk against BOTH the 4h and 1h trend
        # comparison run: compact inputs only (filter, trend and trade-state inputs blanked, no trend gate)
        self.compact_ablate = bool(compact_ablate) and self._compact
        self.trend_gate = bool(trend_gate) and self._compact and not self.compact_ablate
        # "target": live-parity mode. Each step the action is a target position that is
        # held until changed, exactly like Server_AGI -> MT5Executor.reconcile_exposure,
        # with executor-style ATR stop-loss / take-profit checked against bar high/low.
        # "trade": legacy fire-and-wait mode (kept so old checkpoints can be replayed).
        self.execution_mode = str(execution_mode or "trade").lower()
        self.slippage_bps = float(slippage_bps)
        self.random_start = bool(random_start)
        self.episode_length = int(episode_length) if episode_length else None
        self.sl_atr_mult = float(sl_atr_mult)
        self.tp_atr_mult = float(tp_atr_mult)
        self.min_rebalance = float(min_rebalance)
        # Backtests compute features on a longer history (warm-up) and start here.
        self.start_index = int(start_index) if start_index else None
        self.initial_balance = float(initial_balance)
        self.commission_rate = float(commission_rate)
        self.spread_bps = float(spread_bps)
        self.max_drawdown = float(max_drawdown)
        self.window_size = int(window_size)
        self.max_leverage = float(max_leverage)
        self.feature_version = str(feature_version or ENGINEERED_V2)
        self.symbol = str(symbol or "")
        self.action_version = "multi_trade_v1"
        self.trade_memory = trade_memory or {}

        w = reward_weights or {}
        self.reward_weights = {
            "growth": float(w.get("growth", 8.0)),
            "payoff": float(w.get("payoff", 2.0)),
            "sharpe_bonus": float(w.get("sharpe_bonus", 1.0)),
            "drawdown_penalty": float(w.get("drawdown_penalty", 3.0)),
            "cost_penalty": float(w.get("cost_penalty", 5.0)),
            "churn_penalty": float(w.get("churn_penalty", 0.5)),
            "memory_expectancy_bonus": float(w.get("memory_expectancy_bonus", 0.5)),
            "loss_streak_penalty": float(w.get("loss_streak_penalty", 0.4)),
            "directional_followthrough": float(w.get("directional_followthrough", 0.0)),
            "actionable_target_bonus": float(w.get("actionable_target_bonus", 0.0)),
            "neutral_collapse_penalty": float(w.get("neutral_collapse_penalty", 0.0)),
        }
        ac = action_config or {}
        self.action_config = {
            "min_direction_abs": float(ac.get("min_direction_abs", 0.03)),
            "min_size_abs": float(ac.get("min_size_abs", 0.03)),
            "min_target_abs": float(ac.get("min_target_abs", 0.03)),
            "neutral_target_epsilon": float(ac.get("neutral_target_epsilon", ac.get("min_target_abs", 0.03))),
            "neutral_return_floor": float(ac.get("neutral_return_floor", 0.0006)),
        }

        os.makedirs(os.path.join(os.getcwd(), "logs"), exist_ok=True)
        self.profit_log_path = os.path.join(os.getcwd(), "logs", "profitability.jsonl")
        # Writing a JSON line on every step dominated training time; sample instead.
        # AGI_ENV_PROFIT_LOG_EVERY=0 disables the log entirely.
        self.profit_log_every = max(0, int(os.environ.get("AGI_ENV_PROFIT_LOG_EVERY", "500") or 0))
        self.breakeven_trigger_pct = float(os.environ.get("AGI_BREAKEVEN_TRIGGER_PCT", "0.002"))
        self.trailing_trigger_pct = float(os.environ.get("AGI_TRAILING_TRIGGER_PCT", "0.003"))
        self.trailing_distance_pct = float(os.environ.get("AGI_TRAILING_DISTANCE_PCT", "0.002"))
        self.trailing_step_pct = float(os.environ.get("AGI_TRAILING_STEP_PCT", "0.001"))
        self.equity_curve = []
        self._trade_metrics = {}
        self.memory_features = self._build_memory_features(self.trade_memory)
        self.max_portfolio_features = MAX_PORTFOLIO_FEATURE_COUNT
        if portfolio_feature_count is not None:
            self.portfolio_feature_count = min(int(portfolio_feature_count), self.max_portfolio_features)
        else:
            self.portfolio_feature_count = self._default_portfolio_feature_count()

        self.n_features = 0
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)

        if df is not None:
            self._set_data(df)
        else:
            self._use_synthetic()

    @staticmethod
    def _safe_div(a, b):
        return a / (b + 1e-12)

    @staticmethod
    def _shift(arr: np.ndarray, n: int) -> np.ndarray:
        if n <= 0:
            return arr.copy()
        out = np.empty_like(arr)
        out[:n] = arr[0]
        out[n:] = arr[:-n]
        return out

    @staticmethod
    def _rolling_mean(arr: np.ndarray, win: int) -> np.ndarray:
        return pd.Series(arr).rolling(win, min_periods=1).mean().to_numpy(dtype=np.float64)

    @staticmethod
    def _rolling_std(arr: np.ndarray, win: int) -> np.ndarray:
        return pd.Series(arr).rolling(win, min_periods=1).std().fillna(0.0).to_numpy(dtype=np.float64)

    def _build_memory_features(self, memory: dict) -> dict:
        m = memory if isinstance(memory, dict) else {}
        win_rate = float(m.get("win_rate", 50.0))
        expectancy = float(m.get("expectancy", 0.0))
        avg_loss = abs(float(m.get("avg_loss", 0.0)))
        if avg_loss < 1e-6:
            avg_loss = 1.0
        recent_loss_streak = int(m.get("recent_loss_streak", 0))
        trades = int(m.get("trades", 0))
        losses = int(m.get("losses", 0))
        loss_ratio = float(losses / max(1, trades))
        return {
            "win_rate_norm": float(np.clip((win_rate / 50.0) - 1.0, -1.0, 1.0)),
            "expectancy_norm": float(np.tanh(expectancy / avg_loss)),
            "loss_streak_norm": float(np.clip(recent_loss_streak / 10.0, 0.0, 1.0)),
            "loss_ratio_norm": float(np.clip((loss_ratio * 2.0) - 1.0, -1.0, 1.0)),
        }

    def _default_portfolio_feature_count(self) -> int:
        raw = os.environ.get("AGI_PORTFOLIO_FEATURE_COUNT", str(DEFAULT_PORTFOLIO_FEATURE_COUNT))
        try:
            count = int(raw)
        except Exception:
            count = DEFAULT_PORTFOLIO_FEATURE_COUNT
        return max(0, min(self.max_portfolio_features, count))

    @staticmethod
    def infer_portfolio_feature_count(
        obs_dim: int | None,
        window_size: int = 100,
        n_features: int = ENGINEERED_FEATURE_COUNT,
        default: int = DEFAULT_PORTFOLIO_FEATURE_COUNT,
        max_features: int = MAX_PORTFOLIO_FEATURE_COUNT,
    ) -> int:
        if obs_dim is None:
            return default
        residual = int(obs_dim) - int(window_size) * int(n_features)
        if 0 <= residual <= max_features:
            return residual
        return default

    @staticmethod
    def decode_action(
        action,
        max_leverage: float = 1.0,
        min_direction_abs: float = 0.03,
        min_size_abs: float = 0.03,
        min_target_abs: float = 0.03,
    ) -> dict:
        raw = np.asarray(action, dtype=np.float32).reshape(-1)
        if raw.size <= 1:
            target = float(np.clip(raw[0] if raw.size else 0.0, -1.0, 1.0)) * float(max_leverage)
            if abs(target) < float(min_target_abs):
                target = 0.0
            return {
                "direction": float(np.clip(target / max(float(max_leverage), 1e-12), -1.0, 1.0)),
                "size": float(min(1.0, abs(target) / max(float(max_leverage), 1e-12))),
                "risk": 1.0,
                "target": float(target),
                "legacy": True,
            }

        if raw.size == 3:
            direction_raw = float(np.clip(raw[0], -1.0, 1.0))
            size_raw = float(np.clip(raw[1], -1.0, 1.0))
            risk_raw = float(np.clip(raw[2], -1.0, 1.0))

            size = float(np.clip((size_raw + 1.0) * 0.5, 0.0, 1.0))
            risk = float(np.clip((risk_raw + 1.0) * 0.5, 0.0, 1.0))
            target = float(np.clip(direction_raw * size, -1.0, 1.0) * float(max_leverage))
            tp_sl_offset_pct = float(0.005 + risk * 0.015)

            if abs(direction_raw) < float(min_direction_abs) or size < float(min_size_abs) or abs(target) < float(min_target_abs):
                target = 0.0

            return {
                "direction": direction_raw,
                "size": size,
                "risk": risk,
                "target": float(target),
                "entry_mode": "market",
                "entry_offset_pct": 0.0,
                "tp_offset_pct": tp_sl_offset_pct,
                "sl_offset_pct": tp_sl_offset_pct,
                "legacy": True,
            }

        direction_raw = float(np.clip(raw[0], -1.0, 1.0))
        size_raw = float(np.clip(raw[1], -1.0, 1.0))
        entry_mode_raw = float(np.clip(raw[2], -1.0, 1.0))
        entry_offset_raw = float(np.clip(raw[3], -1.0, 1.0))
        tp_raw = float(np.clip(raw[4], -1.0, 1.0))
        sl_raw = float(np.clip(raw[5], -1.0, 1.0))

        size = float(np.clip((size_raw + 1.0) * 0.5, 0.0, 1.0))
        target = float(np.clip(direction_raw * size, -1.0, 1.0) * float(max_leverage))
        entry_mode = TradingEnv._entry_mode_from_raw(entry_mode_raw)
        entry_offset_pct = float(entry_offset_raw * 0.005)
        tp_offset_pct = float(0.005 + max(0.0, tp_raw) * 0.015)
        sl_offset_pct = float(0.005 + max(0.0, -sl_raw) * 0.015)

        if abs(direction_raw) < float(min_direction_abs) or size < float(min_size_abs) or abs(target) < float(min_target_abs):
            target = 0.0

        return {
            "direction": direction_raw,
            "size": size,
            "target": float(target),
            "entry_mode": entry_mode,
            "entry_offset_pct": entry_offset_pct,
            "tp_offset_pct": tp_offset_pct,
            "sl_offset_pct": sl_offset_pct,
            "legacy": False,
        }

    @staticmethod
    def _entry_mode_from_raw(value: float) -> str:
        if value <= -0.33:
            return "market"
        if value <= 0.33:
            return "limit"
        return "stop"

    def _extract_arrays(self, df):
        if isinstance(df, pl.DataFrame):
            pdf = df.to_pandas()
        elif isinstance(df, pd.DataFrame):
            pdf = df.copy()
        else:
            pdf = pl.DataFrame(df).to_pandas()

        pdf.columns = [str(c).lower() for c in pdf.columns]
        if "tick_volume" in pdf.columns and "volume" not in pdf.columns:
            pdf = pdf.rename(columns={"tick_volume": "volume"})

        required = ["open", "high", "low", "close"]
        for c in required:
            if c not in pdf.columns:
                raise ValueError(f"missing required column: {c}")

        if "volume" not in pdf.columns:
            pdf["volume"] = 0.0

        dates = None
        if "time" in pdf.columns:
            dates = pd.to_datetime(pdf["time"], utc=True, errors="coerce")
        elif isinstance(pdf.index, pd.DatetimeIndex):
            dates = pd.to_datetime(pdf.index, utc=True, errors="coerce")

        o = pdf["open"].to_numpy(dtype=np.float64)
        h = pdf["high"].to_numpy(dtype=np.float64)
        l = pdf["low"].to_numpy(dtype=np.float64)
        c = pdf["close"].to_numpy(dtype=np.float64)
        v = pdf["volume"].to_numpy(dtype=np.float64)
        return o, h, l, c, v, dates

    def _set_data(self, df):
        o, h, l, c, v, dates = self._extract_arrays(df)
        self.prices = c.astype(np.float64)
        self._set_bar_arrays(h, l, c)
        self.opens = np.array(o, dtype=np.float64)
        base = pd.DataFrame(
            {
                "time": dates if dates is not None else pd.RangeIndex(len(c)),
                "open": o,
                "high": h,
                "low": l,
                "close": c,
                "volume": v,
            }
        )
        self.feature_data = build_env_feature_matrix(base, feature_version=self.feature_version,
                                                     symbol=self.mf_symbol or self.symbol or None, mf_last=self.mf_last)
        if self.compact_ablate:
            self.feature_data[:, -5:] = 0.0
        self.n_features = int(self.feature_data.shape[1])

        self._update_observation_space()
        self.reset()

    def _use_synthetic(self):
        n = 2000
        price = 1.10 + np.cumsum(np.random.randn(n) * 0.001)
        o = price + np.random.randn(n) * 0.0001
        h = np.maximum(o, price) + np.abs(np.random.randn(n) * 0.0004)
        l = np.minimum(o, price) - np.abs(np.random.randn(n) * 0.0004)
        v = np.random.randint(100, 10000, n).astype(float)
        dates = pd.date_range("2026-01-01", periods=n, freq="5min", tz="UTC")

        self.prices = price.astype(np.float64)
        self._set_bar_arrays(h, l, price)
        self.opens = np.array(o, dtype=np.float64)
        base = pd.DataFrame(
            {
                "time": dates,
                "open": o,
                "high": h,
                "low": l,
                "close": price,
                "volume": v,
            }
        )
        self.feature_data = build_env_feature_matrix(base, feature_version=self.feature_version)
        self.n_features = int(self.feature_data.shape[1])
        self._update_observation_space()
        self.reset()

    def _set_bar_arrays(self, h, l, c):
        self.highs = np.array(h, dtype=np.float64)
        self.lows = np.array(l, dtype=np.float64)
        c = np.asarray(c, dtype=np.float64)
        prev_c = np.concatenate([[c[0]], c[:-1]])
        tr = np.maximum(self.highs - self.lows, np.maximum(np.abs(self.highs - prev_c), np.abs(self.lows - prev_c)))
        # ATR(14) known at the close of each bar (no future data).
        self.atr = pd.Series(tr).rolling(14, min_periods=1).mean().to_numpy(dtype=np.float64)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = max(self.window_size, self.start_index or 0)
        last_start = len(self.prices) - 2 - (self.episode_length or 0)
        if self.random_start and last_start > self.window_size:
            self.current_step = int(self.np_random.integers(self.window_size, last_start + 1))
        self.episode_start = int(self.current_step)
        self.target_trade = None  # target-mode position bookkeeping (entry/sl/tp)
        self.equity = self.initial_balance
        self.position = 0.0
        self.last_action = {"direction": 0.0, "size": 0.0, "target": 0.0, "legacy": False}
        self.peak_equity = self.initial_balance
        self.recent_returns = np.zeros(50, dtype=np.float32)
        self.pending_order = None
        self.open_trade = None
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        if self.execution_mode == "target":
            return self._step_target(action)
        action_meta = self.decode_action(
            action,
            max_leverage=self.max_leverage,
            min_direction_abs=float(self.action_config.get("min_direction_abs", 0.03)),
            min_size_abs=float(self.action_config.get("min_size_abs", 0.03)),
            min_target_abs=float(self.action_config.get("min_target_abs", 0.03)),
        )
        prev_equity = self.equity
        prev_position = float(self.position)

        current_price = float(self.prices[self.current_step])
        prev_price = float(self.prices[self.current_step - 1])

        price_ret = (current_price - prev_price) / (prev_price + 1e-12)
        pnl = self.position * prev_equity * price_ret
        self.equity += pnl

        self._process_action(action_meta, current_price)
        self._update_open_trade(current_price)

        delta = self.position - prev_position
        traded_notional = abs(delta) * self.equity
        commission_cost = traded_notional * self.commission_rate
        spread_cost = traded_notional * (self.spread_bps / 10000.0)
        total_cost = commission_cost + spread_cost

        self.equity -= total_cost
        self.last_action = action_meta

        self.peak_equity = max(self.peak_equity, self.equity)
        self.equity_curve.append(float(self.equity))
        drawdown = (self.peak_equity - self.equity) / (self.peak_equity + 1e-12)

        step_ret = (self.equity - prev_equity) / (prev_equity + 1e-12)
        self.recent_returns = np.roll(self.recent_returns, -1)
        self.recent_returns[-1] = step_ret

        vol = float(np.std(self.recent_returns) + 1e-8)
        sharpe = float(np.mean(self.recent_returns) / (vol + 1e-12))

        payoff = max(step_ret, 0.0) - 0.5 * abs(min(step_ret, 0.0))
        dd_penalty = max(0.0, drawdown - 0.06)
        cost_penalty = total_cost / (prev_equity + 1e-12)
        churn_penalty = abs(delta)
        sharpe_bonus = max(0.0, sharpe)
        target_val = float(action_meta.get("target", 0.0) or 0.0)
        target_mag = abs(target_val)
        directional_followthrough = target_val * float(price_ret)
        neutral_epsilon = float(self.action_config.get("neutral_target_epsilon", self.action_config.get("min_target_abs", 0.03)))
        neutral_return_floor = float(self.action_config.get("neutral_return_floor", 0.0006))
        actionable_target_bonus = target_mag if directional_followthrough > 0.0 and target_mag >= neutral_epsilon else 0.0
        neutral_collapse_penalty = 0.0
        if target_mag < neutral_epsilon and abs(float(price_ret)) >= neutral_return_floor:
            neutral_collapse_penalty = float(min(1.0, abs(float(price_ret)) / max(neutral_return_floor, 1e-12)))
        rw = self.reward_weights
        mem_expectancy = float(self.memory_features.get("expectancy_norm", 0.0))
        mem_loss_streak = float(self.memory_features.get("loss_streak_norm", 0.0))
        memory_growth_scale = float(np.clip(1.0 + rw["memory_expectancy_bonus"] * mem_expectancy, 0.5, 1.5))
        growth_term = memory_growth_scale * step_ret
        loss_streak_penalty = rw["loss_streak_penalty"] * mem_loss_streak * churn_penalty

        reward = (
            rw["growth"] * growth_term
            + rw["payoff"] * payoff
            + rw["sharpe_bonus"] * sharpe_bonus
            + rw["directional_followthrough"] * directional_followthrough
            + rw["actionable_target_bonus"] * actionable_target_bonus
            - rw["drawdown_penalty"] * dd_penalty
            - rw["cost_penalty"] * cost_penalty
            - rw["churn_penalty"] * churn_penalty
            - loss_streak_penalty
            - rw["neutral_collapse_penalty"] * neutral_collapse_penalty
        )
        reward = float(np.clip(reward, -5.0, 5.0))

        terminated = bool(drawdown > self.max_drawdown or self.equity <= 0)
        truncated = bool(self.current_step >= len(self.prices) - 1)

        info = {
            "equity": float(self.equity),
            "position": float(self.position),
            "drawdown": float(drawdown),
            "vol": float(vol),
            "sharpe": float(sharpe),
            "cost": float(total_cost),
            "feature_version": self.feature_version,
            "action_version": self.action_version,
            "action_components": {
                "direction": float(action_meta.get("direction", 0.0)),
                "size": float(action_meta.get("size", 0.0)),
                "target": float(action_meta.get("target", 0.0)),
                "entry_mode": action_meta.get("entry_mode", "market"),
                "entry_offset_pct": float(action_meta.get("entry_offset_pct", 0.0)),
                "tp_offset_pct": float(action_meta.get("tp_offset_pct", 0.0)),
                "sl_offset_pct": float(action_meta.get("sl_offset_pct", 0.0)),
                "legacy": bool(action_meta.get("legacy", True)),
            },
            "reward_components": {
                "growth": float(growth_term),
                "payoff": float(payoff),
                "sharpe_bonus": float(sharpe_bonus),
                "directional_followthrough": float(directional_followthrough),
                "actionable_target_bonus": float(actionable_target_bonus),
                "drawdown_penalty": float(dd_penalty),
                "cost_penalty": float(cost_penalty),
                "churn_penalty": float(churn_penalty),
                "loss_streak_penalty": float(loss_streak_penalty),
                "neutral_collapse_penalty": float(neutral_collapse_penalty),
                "memory_expectancy_norm": float(mem_expectancy),
                "weights": rw,
            },
            "trade_state": self._trade_state_snapshot(current_price),
            "profitability": {
                "equity_curve": list(self.equity_curve[-5:]),
                "trade_metrics": dict(self._trade_metrics),
            },
        }

        self._log_profit_snapshot(current_price, info)

        self.current_step += 1
        return self._get_obs(), reward, terminated, truncated, info

    # ------------------------------------------------------------ target mode
    def _cost_rate(self) -> float:
        return self.commission_rate + (self.spread_bps + self.slippage_bps) / 10000.0

    def _manage_target_stops(self, close_price: float):
        """Mirror MT5Executor.manage_open_positions: breakeven at 1/3 of the stop
        distance in profit, then trail at 1/8 of the stop distance after 1/2."""
        tt = self.target_trade
        d = 1.0 if self.position > 0 else -1.0
        sl_dist = abs(tt["entry"] - tt.get("initial_sl", tt["sl"]))
        tt.setdefault("initial_sl", tt["sl"])
        if sl_dist <= 0:
            return
        profit = (close_price - tt["entry"]) * d
        new_sl = tt["sl"]
        if profit >= sl_dist / 3.0:
            be = tt["entry"] + d * 0.0001 * tt["entry"]
            new_sl = max(new_sl, be) if d > 0 else min(new_sl, be)
        if profit >= sl_dist / 2.0:
            trail = close_price - d * sl_dist / 8.0
            new_sl = max(new_sl, trail) if d > 0 else min(new_sl, trail)
        tt["sl"] = new_sl

    def _step_target(self, action):
        """Live-parity step.

        At the close of bar t-1 (the last bar in the observation) the agent picks a
        target. It is filled at that close (+costs). During bar t the position earns the
        bar's return unless the ATR stop/TP is touched by the bar's high/low, in which
        case it exits at the stop/TP price. This is how MT5Executor behaves live.
        """
        action_meta = self.decode_action(
            action,
            max_leverage=self.max_leverage,
            min_direction_abs=float(self.action_config.get("min_direction_abs", 0.03)),
            min_size_abs=float(self.action_config.get("min_size_abs", 0.03)),
            min_target_abs=float(self.action_config.get("min_target_abs", 0.03)),
        )
        t = self.current_step
        fill_price = float(self.prices[t - 1])
        prev_equity = float(self.equity)
        prev_position = float(self.position)
        target = float(np.clip(action_meta.get("target", 0.0) or 0.0, -self.max_leverage, self.max_leverage))
        if self.trend_gate and target != 0.0:
            from Python.compact_features import trend_columns
            c4, c1 = trend_columns(self.n_features)
            row = self.feature_data[t - 1]
            d_ = 1.0 if target > 0 else -1.0
            if row[c4] == -d_ and row[c1] == -d_:
                # same as the live gate: no opening or adding; an opposite position may only be closed
                if prev_position == 0.0 or np.sign(prev_position) != d_:
                    target = 0.0
                elif abs(target) > abs(prev_position):
                    target = prev_position

        # Rebalance only on a meaningful change (live: volume step + min threshold).
        new_position = prev_position
        if target == 0.0 or np.sign(target) != np.sign(prev_position) or abs(target - prev_position) >= self.min_rebalance:
            new_position = target
        delta = new_position - prev_position
        total_cost = abs(delta) * prev_equity * self._cost_rate()
        self.equity -= total_cost
        if delta != 0.0:
            if new_position == 0.0:
                self.target_trade = None
            elif self.target_trade is None or np.sign(new_position) != np.sign(prev_position):
                atr = float(self.atr[t - 1])
                d = float(np.sign(new_position))
                self.target_trade = {
                    "entry": fill_price,
                    "sl": fill_price - d * self.sl_atr_mult * atr,
                    "tp": fill_price + d * self.tp_atr_mult * atr,
                    "step": int(t),
                }
        self.position = new_position

        # Mark to market over bar t, with intrabar stop / take-profit.
        exit_reason = None
        exit_price = float(self.prices[t])
        if self.position != 0.0 and self.target_trade is not None:
            hi, lo = float(self.highs[t]), float(self.lows[t])
            op = float(self.opens[t]) if getattr(self, "opens", None) is not None else fill_price
            sl, tp = self.target_trade["sl"], self.target_trade["tp"]
            # A gap through the stop fills at the open, not at the stop price.
            if self.position > 0:
                if lo <= sl:
                    exit_reason, exit_price = "sl", min(op, sl)
                elif hi >= tp:
                    exit_reason, exit_price = "tp", max(op, tp) if op >= tp else tp
            else:
                if hi >= sl:
                    exit_reason, exit_price = "sl", max(op, sl)
                elif lo <= tp:
                    exit_reason, exit_price = "tp", min(op, tp) if op <= tp else tp
        bar_ret = (exit_price - fill_price) / (fill_price + 1e-12)
        self.equity += self.position * self.equity * bar_ret
        if exit_reason is None and self.position != 0.0 and self.target_trade is not None:
            self._manage_target_stops(float(self.prices[t]))
        if exit_reason is not None:
            exit_cost = abs(self.position) * self.equity * self._cost_rate()
            self.equity -= exit_cost
            total_cost += exit_cost
            self.position = 0.0
            self.target_trade = None

        self.last_action = action_meta
        self.peak_equity = max(self.peak_equity, self.equity)
        self.equity_curve.append(float(self.equity))
        drawdown = (self.peak_equity - self.equity) / (self.peak_equity + 1e-12)
        step_ret = (self.equity - prev_equity) / (prev_equity + 1e-12)
        self.recent_returns = np.roll(self.recent_returns, -1)
        self.recent_returns[-1] = step_ret

        # Reward = log growth of equity (costs included) minus a penalty for drawdown
        # beyond 5%. No windowed-Sharpe bonus: it dominated PnL and rewarded variance.
        log_ret = float(np.log(max(self.equity, 1e-9) / max(prev_equity, 1e-9)))
        dd_pen = max(0.0, drawdown - 0.05)
        rw = self.reward_weights
        reward = 100.0 * log_ret - rw["drawdown_penalty"] * 0.1 * dd_pen
        reward = float(np.clip(reward, -5.0, 5.0))

        self.current_step += 1
        terminated = bool(drawdown > self.max_drawdown or self.equity <= 0)
        end = len(self.prices) - 1
        if self.episode_length:
            end = min(end, self.episode_start + self.episode_length)
        truncated = bool(self.current_step >= end)

        info = {
            "equity": float(self.equity),
            "position": float(self.position),
            "drawdown": float(drawdown),
            "cost": float(total_cost),
            "exit_reason": exit_reason,
            "feature_version": self.feature_version,
            "execution_mode": self.execution_mode,
            "reward_components": {"growth": log_ret, "drawdown_penalty": dd_pen, "cost_penalty": total_cost / (prev_equity + 1e-12)},
        }
        if self.profit_log_every > 0 and self.current_step % self.profit_log_every == 0:
            self._log_profit_snapshot(exit_price, {"trade_state": {"target_trade": self.target_trade}, "profitability": {"equity_curve": list(self.equity_curve[-5:])}})
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        window = self.feature_data[self.current_step - self.window_size : self.current_step].copy()
        obs_window = window.flatten().astype(np.float32)
        portfolio_state = np.array(self._build_portfolio_state(), dtype=np.float32)
        return np.concatenate([obs_window, portfolio_state]).astype(np.float32)

    def _trade_state_compact(self) -> list[float]:
        """position, open profit / time in trade / distance to stop / to target (in stop units = sl_atr_mult x ATR)."""
        t = max(1, int(self.current_step))
        tt = self.target_trade
        if self.position == 0.0 or tt is None:
            return [0.0, 0.0, 0.0, 0.0, 0.0]
        d = 1.0 if self.position > 0 else -1.0
        px = float(self.prices[min(t - 1, len(self.prices) - 1)])
        unit = max(1e-12, self.sl_atr_mult * float(self.atr[min(t - 1, len(self.atr) - 1)]))
        clip = lambda v: float(np.clip(v, -5.0, 5.0))  # noqa: E731
        return [float(self.position), clip((px - tt["entry"]) * d / unit),
                float(np.clip((t - int(tt.get("step", t))) / 48.0, 0.0, 3.0)),
                clip((px - tt["sl"]) * d / unit), clip((tt["tp"] - px) * d / unit)]

    def set_live_trade(self, entry: float, sl: float, tp: float, bars_in_trade: float):
        """Live inference: describe the bot's real open trade the way training saw its own."""
        self.target_trade = {"entry": float(entry), "sl": float(sl), "tp": float(tp), "initial_sl": float(sl),
                             "step": int(self.current_step - max(0.0, float(bars_in_trade)))}

    def _build_portfolio_state(self) -> list[float]:
        if self._compact:
            if self.compact_ablate:
                return [float(self.position), 0.0, 0.0, 0.0, 0.0][: self.portfolio_feature_count]
            return self._trade_state_compact()[: self.portfolio_feature_count]
        if self.execution_mode == "target":
            # Only the current position is observable identically in training, backtest
            # and live; equity ratio / recent returns / trade memory are held neutral.
            base = [1.0, self.position, 0.0, 0.0, 0.0, 0.0]
            if self.portfolio_feature_count <= 0:
                return []
            return base[: min(self.portfolio_feature_count, len(base))]
        base = [
            self.equity / self.initial_balance,
            self.position,
            float(np.mean(self.recent_returns)),
            float(self.memory_features.get("win_rate_norm", 0.0)),
            float(self.memory_features.get("expectancy_norm", 0.0)),
            float(self.memory_features.get("loss_ratio_norm", 0.0)),
        ]
        if self.portfolio_feature_count <= 0:
            return []
        return base[: min(self.portfolio_feature_count, len(base))]

    def set_portfolio_feature_count(self, count: int):
        self.portfolio_feature_count = max(0, min(self.max_portfolio_features, int(count)))
        self._update_observation_space()

    def _update_observation_space(self):
        shape = self.window_size * self.n_features + self.portfolio_feature_count
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(shape,),
            dtype=np.float32,
        )

    def _trade_state_snapshot(self, current_price: float) -> dict:
        return {
            "open_trade": None if not self.open_trade else dict(self.open_trade),
            "pending_order": None if not self.pending_order else dict(self.pending_order),
            "current_price": float(current_price),
        }

    def _log_profit_snapshot(self, current_price: float, info: dict):
        if self.profit_log_every <= 0 or int(self.current_step) % self.profit_log_every != 0:
            return
        payload = {
            "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "equity": float(self.equity),
            "position": float(self.position),
            "current_price": float(current_price),
            "trade_state": info.get("trade_state"),
            "profitability": info.get("profitability"),
        }
        try:
            _PROFIT_LOG_MAX = 50 * 1024 * 1024  # 50 MB cap
            if os.path.exists(self.profit_log_path) and os.path.getsize(self.profit_log_path) >= _PROFIT_LOG_MAX:
                backup = self.profit_log_path + ".1"
                if os.path.exists(backup):
                    os.remove(backup)
                os.rename(self.profit_log_path, backup)
            with open(self.profit_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload) + "\n")
        except Exception:
            pass

    def _process_action(self, action_meta: dict, current_price: float):
        if self.open_trade and self.open_trade.get("is_open"):
            return

        # A dead-zoned action (target 0) means "stay flat". Previously sign(0) or 1.0
        # turned every flat action into a long, so the agent could never sit out.
        if abs(float(action_meta.get("target", 0.0) or 0.0)) <= 0.0:
            self.position = 0.0
            return

        direction = float(np.sign(action_meta.get("direction", 0.0)) or 1.0)
        entry_mode = action_meta.get("entry_mode", "market")
        entry_offset = float(action_meta.get("entry_offset_pct", 0.0))
        if entry_mode == "market":
            entry_price = self._compute_entry_price(direction, current_price, entry_offset)
            self._open_trade(action_meta, entry_price)
            return

        entry_price = self._compute_entry_price(direction, current_price, entry_offset)
        self.pending_order = {
            "mode": action_meta["entry_mode"],
            "entry_price": float(entry_price),
            "direction": direction,
            "size": float(action_meta["size"]),
            "tp_offset_pct": float(action_meta["tp_offset_pct"]),
            "sl_offset_pct": float(action_meta["sl_offset_pct"]),
            "action_meta": action_meta,
        }
        self.position = 0.0

    def _update_open_trade(self, current_price: float):
        if self.pending_order and self.pending_order.get("mode"):
            if self._check_pending_fill(current_price):
                entry_price = float(self.pending_order["entry_price"])
                self._open_trade(self.pending_order["action_meta"], entry_price)
                self.pending_order = None

        if self.open_trade and self.open_trade.get("is_open"):
            direction = float(self.open_trade["direction"])
            tp = float(self.open_trade["tp_price"])
            sl = float(self.open_trade["sl_price"])
            hit_tp = (direction > 0 and current_price >= tp) or (direction < 0 and current_price <= tp)
            hit_sl = (direction > 0 and current_price <= sl) or (direction < 0 and current_price >= sl)
            entry_price = float(self.open_trade["entry_price"])
            if direction > 0:
                fav = max(0.0, (current_price - entry_price) / (entry_price + 1e-12))
                adv = max(0.0, (entry_price - current_price) / (entry_price + 1e-12))
            else:
                fav = max(0.0, (entry_price - current_price) / (entry_price + 1e-12))
                adv = max(0.0, (current_price - entry_price) / (entry_price + 1e-12))
            self.open_trade["max_fav"] = max(self.open_trade.get("max_fav", 0.0), fav)
            self.open_trade["max_adv"] = max(self.open_trade.get("max_adv", 0.0), adv)

            if not self.open_trade.get("breakeven_triggered") and self.open_trade["max_fav"] >= self.breakeven_trigger_pct:
                self.open_trade["breakeven_triggered"] = True
                self.open_trade["sl_price"] = float(entry_price)

            if not self.open_trade.get("trailing_active") and self.open_trade["max_fav"] >= self.trailing_trigger_pct:
                self.open_trade["trailing_active"] = True

            if self.open_trade.get("trailing_active"):
                trail_distance = self.trailing_distance_pct * entry_price
                new_sl = current_price - trail_distance if direction > 0 else current_price + trail_distance
                step = self.trailing_step_pct * entry_price
                last_trail = self.open_trade.get("last_trailing_price", entry_price)
                moved = abs(new_sl - float(self.open_trade["sl_price"]))
                if moved >= step:
                    self.open_trade["sl_price"] = float(new_sl)
                    self.open_trade["last_trailing_price"] = float(new_sl)
                    self.open_trade["trailing_moves"] = int(self.open_trade.get("trailing_moves", 0) + 1)

            if hit_tp:
                self._close_trade(current_price, "tp")
            elif hit_sl:
                self._close_trade(current_price, "sl")

    def _compute_entry_price(self, direction: float, current_price: float, offset_pct: float) -> float:
        if direction >= 0:
            return float(current_price * (1.0 + offset_pct))
        return float(current_price * (1.0 - offset_pct))

    def _open_trade(self, action_meta: dict, entry_price: float):
        direction = float(np.sign(action_meta.get("direction", 0.0)) or 1.0)
        size = float(action_meta.get("size", 0.0))
        tp_pct = float(action_meta.get("tp_offset_pct", 0.0))
        sl_pct = float(action_meta.get("sl_offset_pct", 0.0))

        if direction >= 0:
            tp_price = float(entry_price * (1.0 + tp_pct))
            sl_price = float(entry_price * (1.0 - sl_pct))
        else:
            tp_price = float(entry_price * (1.0 - tp_pct))
            sl_price = float(entry_price * (1.0 + sl_pct))

        self.open_trade = {
            "direction": direction,
            "size": size,
            "entry_price": float(entry_price),
            "tp_price": float(tp_price),
            "sl_price": float(sl_price),
            "entry_mode": action_meta.get("entry_mode", "market"),
            "is_open": True,
            "max_fav": 0.0,
            "max_adv": 0.0,
            "breakeven_triggered": False,
            "trailing_active": False,
            "trailing_moves": 0,
            "last_entry": float(entry_price),
        }
        self.position = direction * size

    def _close_trade(self, exit_price: float, exit_type: str):
        if not self.open_trade:
            return
        entry_price = float(self.open_trade["entry_price"])
        direction = float(self.open_trade["direction"])
        profit = (exit_price - entry_price) if direction > 0 else (entry_price - exit_price)
        exit_quality_reward = 1.0 if exit_type == "tp" else -1.0
        max_fav = float(self.open_trade.get("max_fav", 0.0))
        base = max(max_fav * entry_price, 1e-4)
        trailing_efficiency = min(1.0, abs(profit) / base) if max_fav > 0 else 0.0
        breakeven_reward = 1.0 if self.open_trade.get("breakeven_triggered") else 0.0
        self._trade_metrics = {
            "exit_type": exit_type,
            "profit": float(profit),
            "exit_quality_reward": exit_quality_reward,
            "trailing_efficiency": float(trailing_efficiency),
            "breakeven_reward": float(breakeven_reward),
            "max_favorable": max_fav,
            "max_adverse": float(self.open_trade.get("max_adv", 0.0)),
            "trailing_moves": int(self.open_trade.get("trailing_moves", 0)),
        }
        self.position = 0.0
        self.open_trade = None

    def _check_pending_fill(self, current_price: float) -> bool:
        pending = self.pending_order
        if not pending:
            return False

        direction = float(pending["direction"])
        mode = pending.get("mode", "limit")
        entry_price = float(pending["entry_price"])

        if mode == "limit":
            return (direction > 0 and current_price <= entry_price) or (direction < 0 and current_price >= entry_price)
        if mode == "stop":
            return (direction > 0 and current_price >= entry_price) or (direction < 0 and current_price <= entry_price)
        return True
