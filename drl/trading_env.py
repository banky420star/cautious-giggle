import gymnasium as gym
import numpy as np
import polars as pl
import pandas as pd
from gymnasium import spaces
from loguru import logger

class TradingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, df=None, initial_balance: float = 10000.0,
                 commission_rate: float = 0.0005, spread: float = 0.0002,
                 max_drawdown: float = 0.15, window_size: int = 100):
        super().__init__()
        
        self.initial_balance = initial_balance
        self.commission_rate = commission_rate      
        self.spread = spread
        self.max_drawdown = max_drawdown
        self.window_size = window_size
        
        # Observation space: price features (5 cols) * window_size + portfolio state (3)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(window_size * 5 + 3,), dtype=np.float32)
        
        # Action space: continuous position [-1.0, 1.0] (leverage)
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
            
        cols = ['open', 'high', 'low', 'close', 'volume']
        # Extract features for state (normalize locally or rely on VecNormalize)
        self.prices = self.df["close"].to_numpy().astype(np.float64)
        
        # Extract the OCHLV data natively to np for maximum speed in get_obs
        self.raw_data = self.df.select(cols).to_numpy().astype(np.float32)
        self.reset()
        
    def _use_synthetic(self):
        n = 500
        price = 1.10 + np.cumsum(np.random.randn(n) * 0.001)
        self.prices = price.astype(np.float64)
        self.raw_data = np.column_stack([
            price,                            # open
            price + np.abs(np.random.randn(n) * 0.0005),  # high
            price - np.abs(np.random.randn(n) * 0.0005),  # low
            price + np.random.randn(n) * 0.0002,          # close
            np.random.randint(100, 10000, n).astype(float) # volume
        ]).astype(np.float32)
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.position = 0.0
        self.entry_price = 0.0
        self.peak_balance = self.initial_balance
        self.recent_returns = np.zeros(20)  # for volatility & Sharpe
        self.episode_rewards = []
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        action_val = np.clip(action[0], -1.0, 1.0)
        
        current_price = self.prices[self.current_step]
        prev_balance = self.balance
        
        # Execute trade with realistic costs
        delta_position = action_val - self.position
        transaction_cost = abs(delta_position) * self.initial_balance * (self.commission_rate + self.spread)
        
        # Unrealized PnL is only the difference from the LAST step
        if self.position != 0:
            unrealized_pnl = self.position * self.initial_balance * ((current_price - self.prices[self.current_step - 1]) / self.prices[self.current_step - 1])
            self.balance += unrealized_pnl
        
        self.position = action_val
        if delta_position != 0:
            self.entry_price = current_price
            
        self.balance -= transaction_cost
        
        # Update drawdown & volatility
        self.peak_balance = max(self.peak_balance, self.balance)
        drawdown = (self.peak_balance - self.balance) / self.peak_balance
        
        # New return for Sharpe/vol
        ret = (self.balance - prev_balance) / prev_balance if prev_balance > 0 else 0
        self.recent_returns = np.roll(self.recent_returns, -1)
        self.recent_returns[-1] = ret
        # === REVISED 2026 RISK-ADJUSTED REWARD (Grok-3 + our optimizations) ===
        pct_return = (self.balance - prev_balance) / prev_balance if prev_balance > 0 else 0.0
        
        volatility = self.recent_returns.std() + 1e-8
        drawdown = (self.peak_balance - self.balance) / self.peak_balance if self.peak_balance > 0 else 0.0
        
        # Core components
        return_signal = pct_return * 15.0
        cost_penalty = abs(transaction_cost) * 80.0
        volatility_penalty = volatility * 12.0
        drawdown_penalty = max(0.0, drawdown - 0.08) * 30.0
        sharpe_bonus = (pct_return / volatility) * 10.0
        
        reward = return_signal - cost_penalty - volatility_penalty - drawdown_penalty + sharpe_bonus
        
        # Safety clipping
        reward = float(np.clip(reward, -10.0, 10.0))
        
        # RiskEngine hard stop
        terminated = bool(drawdown > self.max_drawdown or self.balance <= 0)
        truncated = bool(self.current_step >= len(self.prices) - 1)
        
        self.current_step += 1
        self.episode_rewards.append(reward)
        
        info = {
            "balance": self.balance,
            "position": self.position,
            "drawdown": drawdown,
            "sharpe": np.mean(self.episode_rewards) / (np.std(self.episode_rewards) + 1e-8)
        }
        
        return self._get_obs(), float(reward), terminated, truncated, info

    def _get_obs(self):
        # Feature window + portfolio state
        window = self.raw_data[self.current_step - self.window_size:self.current_step].flatten()
        portfolio_state = np.array([self.balance / self.initial_balance, self.position, self.recent_returns.mean()])
        return np.concatenate([window, portfolio_state]).astype(np.float32)
