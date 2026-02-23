import json
import os

class SymbolManager:
    def __init__(self, config_path="config.json"):
        self.config_path = config_path
        with open(config_path, 'r') as f:
            self.config = json.load(f)
            
    def get_active_symbols(self, current_equity):
        """Scale number of symbols based on equity."""
        pool = self.config.get("symbol_pool", ["BTCUSD", "EURUSD"])
        min_equity = self.config.get("min_equity_per_symbol", 2000)
        
        # Calculate how many symbols we can afford
        count = int(current_equity // min_equity)
        count = max(1, min(count, len(pool)))
        
        active_symbols = pool[:count]
        
        # Update config if changed (persistency)
        if active_symbols != self.config.get("symbols"):
            self.config["symbols"] = active_symbols
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=4)
                
        return active_symbols

    def add_symbol_manually(self, symbol):
        if symbol not in self.config["symbol_pool"]:
            self.config["symbol_pool"].append(symbol)
        
        if symbol not in self.config["symbols"]:
            self.config["symbols"].append(symbol)
            
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=4)
        return True
