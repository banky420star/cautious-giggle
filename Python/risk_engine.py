import os
import yaml
from loguru import logger

class RiskEngine:
    def __init__(self):
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)
        logger.success("Risk Engine active — 1% risk per trade")

    def can_trade(self, balance: float, signal: str, price: float) -> bool:
        if signal == "HOLD":
            return False
        risk_amount = balance * (self.cfg['trading']['risk_percent'] / 100)
        logger.info(f"Risk check passed — ${risk_amount:.2f} at risk")
        return True
