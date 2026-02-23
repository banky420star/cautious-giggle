import rpyc
import pandas as pd
import numpy as np
import os
import time
from dotenv import load_dotenv

load_dotenv()

class MT5Connector:
    def __init__(self, host="localhost", port=18812):
        self.host = host
        self.port = port
        self.conn = None
        self.mt5 = None # This will be the remote service
        self.connected = False

    def connect(self):
        try:
            print(f"Connecting to MT5 Bridge at {self.host}:{self.port}...")
            self.conn = rpyc.connect(self.host, self.port)
            self.mt5 = self.conn.root
            
            login = int(os.getenv("MT5_LOGIN", 0))
            password = os.getenv("MT5_PASSWORD", "")
            server = os.getenv("MT5_SERVER", "")
            
            if not self.mt5.initialize(login, password, server):
                print(f"MT5 Bridge Remote initialization failed.")
                self.connected = False
                return False
            
            print("✅ Successfully connected to MT5 via Bridge")
            self.connected = True
            return True
        except Exception as e:
            print(f"❌ Failed to connect to bridge: {e}")
            self.connected = False
            return False

    def get_rates(self, symbol, timeframe, count=1000):
        """Fetch rates via bridge."""
        if not self.connected: self.connect()
        
        # Timeframe mapping (MT5 constants are ints)
        tf_map = {
            'M1': 1, 'M5': 5, 'M15': 15, 'H1': 16385, 'H4': 16388, 'D1': 16408
        }
        
        mt5_tf = tf_map.get(timeframe, 16385)
        
        try:
            rates_list = self.mt5.copy_rates_from_pos(symbol, mt5_tf, 0, count)
            if rates_list is None:
                return None
                
            df = pd.DataFrame(rates_list)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            return df
        except Exception as e:
            print(f"Error getting rates: {e}")
            return None

    def place_order(self, symbol, order_type, volume, price=0, sl=0, tp=0, magic=123456):
        """Place order via bridge."""
        if not self.connected: self.connect()

        # MT5 Constants
        ORDER_TYPE_BUY = 0
        ORDER_TYPE_SELL = 1
        TRADE_ACTION_DEAL = 1
        
        tick = self.mt5.symbol_info_tick(symbol)
        if not tick: return None
        
        request = {
            "action": TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(volume),
            "type": ORDER_TYPE_BUY if order_type.upper() == 'BUY' else ORDER_TYPE_SELL,
            "price": float(price if price > 0 else (tick['ask'] if order_type.upper() == 'BUY' else tick['bid'])),
            "sl": float(sl),
            "tp": float(tp),
            "magic": magic,
            "comment": "NeuroTrader v3.1 (Bridge)",
            "type_time": 0, # GTC
            "type_filling": 1, # IOC
        }
        
        try:
            result = self.mt5.order_send(request)
            return result
        except Exception as e:
            print(f"Bridge Order Error: {e}")
            return None

    def get_account_info(self):
        if not self.connected: self.connect()
        return self.mt5.account_info() if self.connected else None

    def close_connection(self):
        if self.mt5:
            self.mt5.shutdown()
        if self.conn:
            self.conn.close()
        self.connected = False
