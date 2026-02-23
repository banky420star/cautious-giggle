import time
import json
import os
import pandas as pd
from mt5_connector import MT5Connector
from risk_engine import RiskEngine
from neuro_evo_agent import TradingEnv
from stable_baselines3 import PPO

class TradeServer:
    def __init__(self, config_path="config.json"):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.connector = MT5Connector()
        self.risk_engine = RiskEngine()
        self.symbols = self.config.get("symbols", ["BTCUSD"])
        self.models = {}
        self._load_models()

    def _load_models(self):
        """Load PPO models for each active symbol."""
        for symbol in self.symbols:
            model_path = f"models/PPO_{symbol}_Latest.zip"
            if os.path.exists(model_path):
                self.models[symbol] = PPO.load(model_path)
            else:
                print(f"⚠️ Model for {symbol} not found. Using generic if available.")

    def run_cycle(self):
        """Single iteration of the trading loop."""
        if not self.connector.connected:
            self.connector.connect()

        acct_info = self.connector.get_account_info()
        if not acct_info: return

        equity = acct_info['equity']
        balance = acct_info['balance']

        # Check Global Drawdown Killswitch
        if self.risk_engine.check_drawdown(equity, balance):
            print("🚨 GLOBAL DRAWDOWN KILLSWITCH ACTIVATED! Closing all trades.")
            # Implementation for closing all trades would go here
            return

        for symbol in self.symbols:
            # 1. Get Market State
            df = self.connector.get_rates(symbol, "M15", 100)
            if df is None: continue

            # 2. Get Signal from Model
            if symbol in self.models:
                # Prepare observation (simplified)
                obs = df.select_dtypes(include=['number']).tail(50).values.astype('float32')
                # Padding or reshaping as needed by TradingEnv
                if obs.shape[1] < 10:
                    import numpy as np
                    padding = np.zeros((50, 10 - obs.shape[1]))
                    obs = np.hstack((obs, padding))
                
                action, _ = self.models[symbol].predict(obs, deterministic=True)
                
                # 3. Execution Logic
                self._execute_action(symbol, action, df, balance)

    def _execute_action(self, symbol, action, df, balance):
        """Translate model actions (0,1,2) into MT5 orders."""
        # 0=Hold, 1=Buy, 2=Sell
        if action == 1: # Buy Signal
            atr_sl, atr_tp = self.risk_engine.get_atr_sl_tp(df)
            curr_price = df['close'].iloc[-1]
            sl = curr_price - atr_sl
            tp = curr_price + atr_tp
            vol = self.risk_engine.calculate_position_size(balance, curr_price, sl, symbol)
            
            print(f"🔵 BUY {symbol} | Vol: {vol} | SL: {sl:.2f} | TP: {tp:.2f}")
            self.connector.place_order(symbol, "BUY", vol, sl=sl, tp=tp)
            
        elif action == 2: # Sell Signal
            atr_sl, atr_tp = self.risk_engine.get_atr_sl_tp(df)
            curr_price = df['close'].iloc[-1]
            sl = curr_price + atr_sl
            tp = curr_price - atr_tp
            vol = self.risk_engine.calculate_position_size(balance, curr_price, sl, symbol)
            
            print(f"🔴 SELL {symbol} | Vol: {vol} | SL: {sl:.2f} | TP: {tp:.2f}")
            self.connector.place_order(symbol, "SELL", vol, sl=sl, tp=tp)

    def start(self):
        print("🚀 Trade Server Loop Started...")
        while True:
            try:
                self.run_cycle()
                time.sleep(60) # Scan every minute
            except Exception as e:
                print(f"Error in trade loop: {e}")
                time.sleep(10)

if __name__ == "__main__":
    server = TradeServer()
    server.start()
