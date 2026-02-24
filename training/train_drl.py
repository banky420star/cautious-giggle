"""
DRL PPO Training Script — trains on real market data from Yahoo Finance.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from loguru import logger
from Python.data_feed import fetch_training_data
from drl.ppo_agent import train

# ── Logging ─────────────────────────────────────────────────────────
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
logger.add(os.path.join(LOG_DIR, "drl_training.log"), rotation="10 MB", level="DEBUG")

# ── Config ──────────────────────────────────────────────────────────
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

symbols = cfg.get("trading", {}).get("symbols", ["EURUSD"])
total_timesteps = cfg.get("drl", {}).get("total_timesteps", 300_000)


def main():
    logger.info(f"DRL Training — symbols: {symbols} | timesteps: {total_timesteps:,}")

    for sym in symbols:
        logger.info(f"Fetching training data for {sym}...")
        df = fetch_training_data(sym, period="60d")
        if df.empty or len(df) < 100:
            logger.warning(f"Insufficient data for {sym}, skipping")
            continue

        logger.info(f"Training PPO on {sym} ({len(df)} candles)...")
        train(steps=total_timesteps // len(symbols), df=df)

    logger.success("DRL training complete for all symbols!")


if __name__ == "__main__":
    main()
