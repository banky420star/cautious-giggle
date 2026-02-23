import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drl.ppo_agent import train
from loguru import logger

logger.info("Starting DRL self-learning training...")
train(steps=300000)
