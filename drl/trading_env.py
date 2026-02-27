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
        commission_rate: float = 0.0002,   # 2 bps per notional traded (approx)
        spread_bps: float = 2.0,            # spread in basis points applied as slippage on fills
        max_drawdown: float = 0.15,
        window_size: int = 100,
        max_leverage: float = 1.0,
    ):
        super().__init__()
        self.initial_balance = float(initial_balance)
        self.commission_rate = float(commission_rate)
        self.spread_bps = float(spread_bps)
        self.max_drawdown = float(max_drawdown)
        self.window_size = int(window_size)
        self.max_leverage = float(max_leverage)

        # Observation: window of 5 features normalized + 3 portfolio stats
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(window_size * 5 + 3,), dtype=np.float32
        )

        # Action: target position fraction in [-1, 1] (scaled by max_leverage)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        if df is not None:
            self._set_data(df)
        else:
            self._use_synthetic()

    def _set_data(self, df):
        if isinstance(df, pd.DataFrame):
            self.df = pl.from_pandas(df)
        else:
            self.df = df

        cols = ["open", "high", "low", "close", "volume"]
        self.prices = self.df["close"].to_numpy().astype(np.float64)
        self.raw_data = self.df.select(cols).to_numpy().astype(np.float32)
        self.reset()

    def _use_synthetic(self):
        n = 2000
        price = 1.10 + np.cumsum(np.random.randn(n) * 0.001)
        self.prices = price.astype(np.float64)
        self.raw_data = np.column_stack(
            [
                price,
                price + np.abs(np.random.randn(n) * 0.0005),
                price - np.abs(np.random.randn(n) * 0.0005),
                price + np.random.randn(n) * 0.0002,
                np.random.randint(100, 10000, n).astype(float),
            ]
        ).astype(np.float32)
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.equity = self.initial_balance
        self.position = 0.0  # target exposure in [-max_leverage, +max_leverage]
        self.peak_equity = self.initial_balance
        self.recent_returns = np.zeros(50, dtype=np.float32)  # smoother than 20
        self._last_price = float(self.prices[self.current_step - 1])
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        target = float(np.clip(action[0], -1.0, 1.0)) * self.max_leverage
        prev_equity = self.equity

        # Market prices
        current_price = float(self.prices[self.current_step])
        prev_price = float(self.prices[self.current_step - 1])

        # Mark-to-market PnL on held position (equity-based notional)
        # notional = equity at previous step
        price_ret = (current_price - prev_price) / (prev_price + 1e-12)
        pnl = self.position * prev_equity * price_ret
        self.equity += pnl

        # Trading cost if position changes (commission + spread slippage)
        delta = target - self.position
        traded_notional = abs(delta) * self.equity  # scale with current equity
        commission_cost = traded_notional * self.commission_rate

        # Spread modeled as slippage on entry/exit proportional to traded notional
        spread_cost = traded_notional * (self.spread_bps / 10000.0)

        total_cost = commission_cost + spread_cost
        self.equity -= total_cost
        self.position = target

        # Drawdown tracking
        self.peak_equity = max(self.peak_equity, self.equity)
        drawdown = (self.peak_equity - self.equity) / (self.peak_equity + 1e-12)

        # Return for reward + risk stats
        step_ret = (self.equity - prev_equity) / (prev_equity + 1e-12)
        self.recent_returns = np.roll(self.recent_returns, -1)
        self.recent_returns[-1] = step_ret

        vol = float(np.std(self.recent_returns) + 1e-8)
        sharpe = float(np.mean(self.recent_returns) / (vol + 1e-12))

        # Stable reward: return - costs - smooth drawdown penalty
        # Drawdown penalty only bites after 8% DD (soft)
        dd_penalty = max(0.0, drawdown - 0.08) * 2.5
        cost_penalty = total_cost / (prev_equity + 1e-12) * 5.0

        reward = float(step_ret * 10.0 - cost_penalty - dd_penalty)

        # Optional clipping (gentler)
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
        }

        self.current_step += 1
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        window = self.raw_data[self.current_step - self.window_size : self.current_step].copy()

        # Normalize OHLC by last close to reduce scale issues; volume log-scaled
        last_close = float(window[-1, 3]) + 1e-12
        window[:, 0:4] = (window[:, 0:4] / last_close) - 1.0
        window[:, 4] = np.log1p(np.maximum(window[:, 4], 0.0))

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
