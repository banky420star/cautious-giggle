from stable_baselines3 import PPO
from drl.trading_env import TradingEnv
from loguru import logger

env = TradingEnv()
model = PPO("MlpPolicy", env, verbose=1, device="mps")

def train(steps=100000):
    model.learn(total_timesteps=steps)
    model.save("models/ppo_trading.zip")
    logger.success(f"DRL PPO trained for {steps} steps — Self-learning complete!")
