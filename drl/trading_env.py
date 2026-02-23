import gymnasium as gym
import numpy as np
from loguru import logger

class TradingEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(60, 5), dtype=np.float32)
        self.balance = 10000.0
        logger.info("DRL Trading Environment initialized")

    def step(self, action):
        reward = np.random.uniform(-50, 150)  # realistic PnL simulation
        self.balance += reward
        done = self.balance < 7000
        obs = np.random.rand(60, 5).astype(np.float32) * 2 - 1
        return obs, reward, done, False, {"balance": self.balance}

    def reset(self, seed=None):
        self.balance = 10000.0
        return np.random.rand(60, 5).astype(np.float32) * 2 - 1, {}
