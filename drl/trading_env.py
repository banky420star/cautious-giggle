import gymnasium as gym
import numpy as np
import polars as pl
import pandas as pd
from gymnasium import spaces


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
    ):
        super().__init__()
        self.initial_balance = float(initial_balance)
        self.commission_rate = float(commission_rate)
        self.spread_bps = float(spread_bps)
        self.max_drawdown = float(max_drawdown)
        self.window_size = int(window_size)
        self.max_leverage = float(max_leverage)
        self.feature_version = "engineered_v2"

        w = reward_weights or {}
        self.reward_weights = {
            "growth": float(w.get("growth", 8.0)),
            "payoff": float(w.get("payoff", 2.0)),
            "sharpe_bonus": float(w.get("sharpe_bonus", 1.0)),
            "drawdown_penalty": float(w.get("drawdown_penalty", 3.0)),
            "cost_penalty": float(w.get("cost_penalty", 5.0)),
            "churn_penalty": float(w.get("churn_penalty", 0.5)),
        }

        self.n_features = 0
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

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
                hour = dt.hour.to_numpy(dtype=np.float64)
                dow = dt.dayofweek.to_numpy(dtype=np.float64)
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

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.window_size * self.n_features + 3,),
            dtype=np.float32,
        )
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
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.window_size * self.n_features + 3,),
            dtype=np.float32,
        )
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.equity = self.initial_balance
        self.position = 0.0
        self.peak_equity = self.initial_balance
        self.recent_returns = np.zeros(50, dtype=np.float32)
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        target = float(np.clip(action[0], -1.0, 1.0)) * self.max_leverage
        prev_equity = self.equity

        current_price = float(self.prices[self.current_step])
        prev_price = float(self.prices[self.current_step - 1])

        price_ret = (current_price - prev_price) / (prev_price + 1e-12)
        pnl = self.position * prev_equity * price_ret
        self.equity += pnl

        delta = target - self.position
        traded_notional = abs(delta) * self.equity
        commission_cost = traded_notional * self.commission_rate
        spread_cost = traded_notional * (self.spread_bps / 10000.0)
        total_cost = commission_cost + spread_cost

        self.equity -= total_cost
        self.position = target

        self.peak_equity = max(self.peak_equity, self.equity)
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
        reward = (
            rw["growth"] * step_ret
            + rw["payoff"] * payoff
            + rw["sharpe_bonus"] * sharpe_bonus
            - rw["drawdown_penalty"] * dd_penalty
            - rw["cost_penalty"] * cost_penalty
            - rw["churn_penalty"] * churn_penalty
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
            "reward_components": {
                "growth": float(step_ret),
                "payoff": float(payoff),
                "sharpe_bonus": float(sharpe_bonus),
                "drawdown_penalty": float(dd_penalty),
                "cost_penalty": float(cost_penalty),
                "churn_penalty": float(churn_penalty),
                "weights": rw,
            },
        }

        self.current_step += 1
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        window = self.feature_data[self.current_step - self.window_size : self.current_step].copy()
        obs_window = window.flatten().astype(np.float32)
        portfolio_state = np.array(
            [
                self.equity / self.initial_balance,
                self.position,
                float(np.mean(self.recent_returns)),
            ],
            dtype=np.float32,
        )
        return np.concatenate([obs_window, portfolio_state]).astype(np.float32)
