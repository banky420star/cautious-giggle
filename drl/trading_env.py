import datetime
import json
import os

import gymnasium as gym
import numpy as np
import polars as pl
import pandas as pd
from gymnasium import spaces

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
    ):
        super().__init__()
        self.initial_balance = float(initial_balance)
        self.commission_rate = float(commission_rate)
        self.spread_bps = float(spread_bps)
        self.max_drawdown = float(max_drawdown)
        self.window_size = int(window_size)
        self.max_leverage = float(max_leverage)
        self.feature_version = "engineered_v2"
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
        }

        os.makedirs(os.path.join(os.getcwd(), "logs"), exist_ok=True)
        self.profit_log_path = os.path.join(os.getcwd(), "logs", "profitability.jsonl")
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
    def decode_action(action, max_leverage: float = 1.0) -> dict:
        raw = np.asarray(action, dtype=np.float32).reshape(-1)
        if raw.size <= 1:
            target = float(np.clip(raw[0] if raw.size else 0.0, -1.0, 1.0)) * float(max_leverage)
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

            if abs(direction_raw) < 0.03 or size < 0.03:
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

        if abs(direction_raw) < 0.03 or size < 0.03:
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

    def _build_feature_matrix(self, o, h, l, c, v, dates):
        eps = 1e-12
        range_ = np.maximum(h - l, eps)

        close_shift1 = self._shift(c, 1)
        close_shift5 = self._shift(c, 5)
        close_shift20 = self._shift(c, 20)

        log_ret1 = np.log(np.maximum(c, eps) / np.maximum(close_shift1, eps))
        log_ret5 = np.log(np.maximum(c, eps) / np.maximum(close_shift5, eps))
        log_ret20 = np.log(np.maximum(c, eps) / np.maximum(close_shift20, eps))

        body_ratio = (c - o) / range_
        upper_wick = (h - np.maximum(o, c)) / range_
        lower_wick = (np.minimum(o, c) - l) / range_
        range_ratio = self._safe_div(h - l, c)

        rv_20 = self._rolling_std(log_ret1, 20)
        vol_ma20 = self._rolling_mean(np.maximum(v, 0.0), 20)
        rel_volume = self._safe_div(np.maximum(v, 0.0), vol_ma20)
        spread_est_bps = self._safe_div(h - l, c) * 10000.0

        ma50 = self._rolling_mean(c, 50)
        htf_trend = self._safe_div(c, ma50) - 1.0

        hour_sin = np.zeros_like(c)
        hour_cos = np.zeros_like(c)
        dow_sin = np.zeros_like(c)
        dow_cos = np.zeros_like(c)
        if dates is not None:
            dt = pd.to_datetime(dates, utc=True, errors="coerce")
            if isinstance(dt, pd.DatetimeIndex):
                hour = dt.hour.astype(np.float64)
                dow = dt.dayofweek.astype(np.float64)
            else:
                hour = dt.dt.hour.to_numpy(dtype=np.float64)
                dow = dt.dt.dayofweek.to_numpy(dtype=np.float64)
            hour_sin = np.sin(2.0 * np.pi * hour / 24.0)
            hour_cos = np.cos(2.0 * np.pi * hour / 24.0)
            dow_sin = np.sin(2.0 * np.pi * dow / 7.0)
            dow_cos = np.cos(2.0 * np.pi * dow / 7.0)

        valid_rv = rv_20[np.isfinite(rv_20)]
        if len(valid_rv) > 10:
            q1 = np.quantile(valid_rv, 0.33)
            q2 = np.quantile(valid_rv, 0.66)
            vol_bucket = np.where(rv_20 <= q1, 0.0, np.where(rv_20 <= q2, 0.5, 1.0))
        else:
            vol_bucket = np.zeros_like(c)

        close_rel = self._safe_div(c, close_shift1) - 1.0
        open_rel = self._safe_div(o, c) - 1.0
        high_rel = self._safe_div(h, c) - 1.0
        low_rel = self._safe_div(l, c) - 1.0
        log_vol = np.log1p(np.maximum(v, 0.0))

        mat = np.column_stack(
            [
                open_rel,
                high_rel,
                low_rel,
                close_rel,
                log_vol,
                log_ret1,
                log_ret5,
                log_ret20,
                body_ratio,
                upper_wick,
                lower_wick,
                range_ratio,
                rv_20,
                rel_volume,
                spread_est_bps,
                hour_sin,
                hour_cos,
                dow_sin,
                dow_cos,
                htf_trend,
                vol_bucket,
            ]
        )

        mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        return mat

    def _set_data(self, df):
        o, h, l, c, v, dates = self._extract_arrays(df)
        self.prices = c.astype(np.float64)
        self.feature_data = self._build_feature_matrix(o, h, l, c, v, dates)
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
        self.feature_data = self._build_feature_matrix(o, h, l, price, v, dates)
        self.n_features = int(self.feature_data.shape[1])
        self._update_observation_space()
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.equity = self.initial_balance
        self.position = 0.0
        self.last_action = {"direction": 0.0, "size": 0.0, "target": 0.0, "legacy": False}
        self.peak_equity = self.initial_balance
        self.recent_returns = np.zeros(50, dtype=np.float32)
        self.pending_order = None
        self.open_trade = None
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        action_meta = self.decode_action(action, max_leverage=self.max_leverage)
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
            - rw["drawdown_penalty"] * dd_penalty
            - rw["cost_penalty"] * cost_penalty
            - rw["churn_penalty"] * churn_penalty
            - loss_streak_penalty
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
                "drawdown_penalty": float(dd_penalty),
                "cost_penalty": float(cost_penalty),
                "churn_penalty": float(churn_penalty),
                "loss_streak_penalty": float(loss_streak_penalty),
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

    def _get_obs(self):
        window = self.feature_data[self.current_step - self.window_size : self.current_step].copy()
        obs_window = window.flatten().astype(np.float32)
        portfolio_state = np.array(self._build_portfolio_state(), dtype=np.float32)
        return np.concatenate([obs_window, portfolio_state]).astype(np.float32)

    def _build_portfolio_state(self) -> list[float]:
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
        payload = {
            "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "equity": float(self.equity),
            "position": float(self.position),
            "current_price": float(current_price),
            "trade_state": info.get("trade_state"),
            "profitability": info.get("profitability"),
        }
        try:
            with open(self.profit_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload) + "\n")
        except Exception:
            pass

    def _process_action(self, action_meta: dict, current_price: float):
        if self.open_trade and self.open_trade.get("is_open"):
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
