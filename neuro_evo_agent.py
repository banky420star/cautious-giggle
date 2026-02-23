import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
import torch as th

class TradingEnv(gym.Env):
    """Custom Environment for MT5 Trading that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df, initial_balance=10000):
        super(TradingEnv, self).__init__()
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        
        # Action space: 0=Hold, 1=Buy, 2=Sell
        self.action_space = spaces.Discrete(3)
        
        # Observation space: OHLCV + Indicators (last 50 candles as window)
        # Assuming 10 features per candle
        self.window_size = 50
        self.features_count = 10 
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, 
                                            shape=(self.window_size, self.features_count), 
                                            dtype=np.float32)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.current_step = self.window_size
        self.position = 0 # 0=None, 1=Long, -1=Short
        self.entry_price = 0
        self.equity_history = [self.initial_balance]
        
        return self._next_observation(), {}

    def _next_observation(self):
        # Slice window and normalize (mock normalization)
        obs = self.df.iloc[self.current_step - self.window_size : self.current_step]
        # Drop non-numeric for obs
        obs_numeric = obs.select_dtypes(include=[np.number]).values
        # Pad if features mismatch
        if obs_numeric.shape[1] < self.features_count:
            padding = np.zeros((self.window_size, self.features_count - obs_numeric.shape[1]))
            obs_numeric = np.hstack((obs_numeric, padding))
        elif obs_numeric.shape[1] > self.features_count:
            obs_numeric = obs_numeric[:, :self.features_count]
            
        return obs_numeric.astype(np.float32)

    def step(self, action):
        self.current_step += 1
        
        if self.current_step >= len(self.df) - 1:
            return self._next_observation(), 0, True, False, {}

        current_price = self.df.loc[self.current_step, 'close']
        reward = 0
        done = False

        # Simple reward: Change in equity
        if self.position == 1: # Long
            reward = current_price - self.df.loc[self.current_step - 1, 'close']
        elif self.position == -1: # Short
            reward = self.df.loc[self.current_step - 1, 'close'] - current_price

        # Action Logic
        if action == 1 and self.position != 1: # Buy
            if self.position == -1: reward += (self.entry_price - current_price) # Close short
            self.position = 1
            self.entry_price = current_price
        elif action == 2 and self.position != -1: # Sell
            if self.position == 1: reward += (current_price - self.entry_price) # Close long
            self.position = -1
            self.entry_price = current_price
        elif action == 0 and self.position != 0: # Close position
            if self.position == 1: reward += (current_price - self.entry_price)
            else: reward += (self.entry_price - current_price)
            self.position = 0
            self.entry_price = 0

        self.balance += reward
        self.equity_history.append(self.balance)
        
        if self.balance <= self.initial_balance * 0.5: # 50% drawdown = Done
            done = True

        return self._next_observation(), reward, done, False, {}

def create_agent(env):
    policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=dict(pi=[128, 128], qf=[128, 128]))
    model = PPO("MlpPolicy", env, verbose=1, policy_kwargs=None)
    return model
