import yaml
from datetime import datetime

class RiskEngine:
    def __init__(self):
        with open("config.yaml", "r") as f:
            cfg = yaml.safe_load(f)

        self.max_daily_loss = cfg["risk"]["max_daily_loss"]
        self.max_daily_trades = cfg["risk"]["max_daily_trades"]
        self.max_lots = cfg["risk"]["max_lots"]

        self.realized_pnl_today = 0.0
        self.daily_trades = 0
        self.halt = False
        self.error_count = 0

    def reset_daily(self):
        self.realized_pnl_today = 0.0
        self.daily_trades = 0
        self.error_count = 0

    def record_trade(self):
        self.daily_trades += 1

    def record_pnl(self, pnl):
        self.realized_pnl_today += pnl
        if self.realized_pnl_today <= -abs(self.max_daily_loss):
            self.halt = True

    def record_error(self):
        self.error_count += 1
        if self.error_count >= 3:
            self.halt = True

    def can_trade(self):
        if self.halt:
            return False
        if self.daily_trades >= self.max_daily_trades:
            return False
        return True
