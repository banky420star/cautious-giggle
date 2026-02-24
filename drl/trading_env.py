"""
Trading Environment for DRL PPO Agent.
Uses real market data features (OHLCV) as observations.
Supports BUY/SELL/HOLD actions with realistic PnL simulation.
"""
import gymnasium as gym
import numpy as np
import pandas as pd
from loguru import logger


class TradingEnv(gym.Env):
    """OpenAI Gymnasium environment for forex/commodity trading."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, df: pd.DataFrame = None, initial_balance: float = 10000.0,
                 commission: float = 0.0001, seq_len: int = 60):
        super().__init__()

        self.initial_balance = initial_balance
        self.commission = commission
        self.seq_len = seq_len

        # Action space: 0=HOLD, 1=BUY, 2=SELL
        self.action_space = gym.spaces.Discrete(3)
        # Observation: seq_len x 5 (OHLCV normalised window)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(seq_len, 5), dtype=np.float32
        )

        # Store market data
        if df is not None:
            self._set_data(df)
        else:
            # Synthetic fallback so the env can be instantiated without data
            self._use_synthetic()

        logger.info(f"DRL Trading Environment initialized | balance=${initial_balance:,.0f} | "
                    f"commission={commission} | seq_len={seq_len}")

    # ── Data management ──────────────────────────────────────────────
    def _set_data(self, df: pd.DataFrame):
        """Ingest a real OHLCV DataFrame."""
        cols = ['open', 'high', 'low', 'close', 'volume']
        self.raw = df[cols].values.astype(np.float64)
        # Min-max normalise per column
        self.data_min = self.raw.min(axis=0)
        self.data_max = self.raw.max(axis=0)
        denom = (self.data_max - self.data_min)
        denom[denom == 0] = 1.0
        self.data = ((self.raw - self.data_min) / denom).astype(np.float32)
        self.prices = df['close'].values.astype(np.float64)
        self.n_steps = len(self.data)
        self._synthetic = False

    def _use_synthetic(self):
        """Generate synthetic random walk data for unit-testing / dry runs."""
        n = 500
        price = 1.10 + np.cumsum(np.random.randn(n) * 0.001)
        self.raw = np.column_stack([
            price,                            # open
            price + np.abs(np.random.randn(n) * 0.0005),  # high
            price - np.abs(np.random.randn(n) * 0.0005),  # low
            price + np.random.randn(n) * 0.0002,          # close
            np.random.randint(100, 10000, n).astype(float) # volume
        ])
        self.prices = self.raw[:, 3]
        self.data_min = self.raw.min(axis=0)
        self.data_max = self.raw.max(axis=0)
        denom = (self.data_max - self.data_min)
        denom[denom == 0] = 1.0
        self.data = ((self.raw - self.data_min) / denom).astype(np.float32)
        self.n_steps = n
        self._synthetic = True

    # ── Gym interface ────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.position = 0        # +1 = long, -1 = short, 0 = flat
        self.entry_price = 0.0
        self.current_step = self.seq_len
        self.total_pnl = 0.0
        self.trades = 0
        self.wins = 0
        return self._get_obs(), {}

    def _get_obs(self):
        """Return the last seq_len candles as a normalised 2D array."""
        start = self.current_step - self.seq_len
        return self.data[start:self.current_step].copy()

    def step(self, action):
        """Execute one time-step.  action: 0=HOLD, 1=BUY, 2=SELL."""
        price = self.prices[self.current_step]
        reward = 0.0

        # ── Close existing position if signal flips ──────────────────
        if self.position != 0:
            if (self.position == 1 and action == 2) or \
               (self.position == -1 and action == 1) or \
               action == 0:
                # Close position
                if self.position == 1:
                    pnl = (price - self.entry_price) / self.entry_price
                else:
                    pnl = (self.entry_price - price) / self.entry_price
                pnl -= self.commission  # round-trip commission
                dollar_pnl = pnl * self.initial_balance
                self.balance += dollar_pnl
                self.total_pnl += dollar_pnl
                self.trades += 1
                if dollar_pnl > 0:
                    self.wins += 1
                reward = pnl * 100  # scale for RL
                self.position = 0
                self.entry_price = 0.0

        # ── Open new position ────────────────────────────────────────
        if action == 1 and self.position == 0:
            self.position = 1
            self.entry_price = price
        elif action == 2 and self.position == 0:
            self.position = -1
            self.entry_price = price

        # ── Unrealised PnL penalty/reward for holding ────────────────
        if self.position != 0:
            if self.position == 1:
                unrealised = (price - self.entry_price) / self.entry_price
            else:
                unrealised = (self.entry_price - price) / self.entry_price
            reward += unrealised * 10  # small shaping signal

        self.current_step += 1

        # ── Termination conditions ───────────────────────────────────
        terminated = False
        if self.current_step >= self.n_steps - 1:
            terminated = True
        if self.balance < self.initial_balance * 0.7:  # 30% drawdown → stop
            terminated = True
            reward -= 5.0  # heavy penalty for blowing up

        truncated = False
        info = {
            "balance": self.balance,
            "total_pnl": self.total_pnl,
            "trades": self.trades,
            "win_rate": self.wins / max(self.trades, 1),
            "position": self.position,
        }

        return self._get_obs(), reward, terminated, truncated, info

    def render(self, mode="human"):
        wr = self.wins / max(self.trades, 1) * 100
        print(f"Step {self.current_step}/{self.n_steps} | "
              f"Balance: ${self.balance:,.2f} | "
              f"PnL: ${self.total_pnl:,.2f} | "
              f"Trades: {self.trades} | WR: {wr:.0f}%")
