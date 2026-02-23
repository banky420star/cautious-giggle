import numpy as np
import pandas as pd
import os
import json

class RiskEngine:
    def __init__(self, config_path="config.json"):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
            
        self.risk_per_trade = self.config.get("risk_per_trade", 0.01)
        self.global_dd_limit = self.config.get("global_drawdown_killswitch", 0.15)
        self.leverage = self.config.get("leverage", 100)

    def calculate_position_size(self, balance, entry_price, sl_price, symbol):
        """
        Calculate position size based on balance, risk per trade, and ATR-based SL.
        Uses Kelly Criterion adjustment if provided.
        """
        if sl_price == 0 or entry_price == sl_price:
            return 0.01 # Minimum default
            
        risk_amount = balance * self.risk_per_trade
        stop_loss_pips = abs(entry_price - sl_price)
        
        # Simplified lot size calculation (standard for Forex/CFD)
        # In MT5, 1 lot usually = 100,000 units for Forex.
        # We need to adjust based on symbol properties (point, tick_value)
        # For simplicity in this shell, we'll return a scaled lot size.
        
        lot_size = risk_amount / (stop_loss_pips * 100000) # Proxy for 1 lot = 100k
        return round(max(0.01, min(lot_size, 5.0)), 2) 

    def check_drawdown(self, equity, initial_balance):
        """Killswitch if global drawdown exceeded."""
        drawdown = (initial_balance - equity) / initial_balance
        if drawdown >= self.global_dd_limit:
            return True # TRANCE: Stop all trading
        return False

    def get_atr_sl_tp(self, df, window=14, multiplier_sl=2.0, multiplier_tp=4.0):
        """Calculate dynamic SL/TP based on ATR."""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=window).mean().iloc[-1]
        
        return atr * multiplier_sl, atr * multiplier_tp

    def apply_kelly_criterion(self, win_rate, profit_factor):
        """Adjust risk per trade using simplified Kelly."""
        # Kelly % = W - (1-W)/R
        # W = Win Rate, R = Profit Factor (Average Win / Average Loss)
        if profit_factor <= 0: return self.risk_per_trade
        
        kelly = win_rate - (1 - win_rate) / profit_factor
        # We use a fractional Kelly (e.g., 20% of suggested Kelly) for safety
        adjusted_risk = max(0.005, min(kelly * 0.2, self.risk_per_trade * 2))
        return adjusted_risk
