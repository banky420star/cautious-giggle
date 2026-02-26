from loguru import logger
import redis
import json
import asyncio
import os
from Python.risk_engine import RiskEngine

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None
    logger.warning("MetaTrader5 package not found. Paper mode only.")

class MT5Executor:
    def __init__(self, paper_mode: bool = True):
        self.paper_mode = paper_mode
        self.risk = RiskEngine()
        
        # Use simple os.environ or fallback to localhost for Redis if not in Docker yet 
        redis_host = os.environ.get('REDIS_HOST', 'localhost')
        self.redis = redis.Redis(host=redis_host, port=6379, decode_responses=True)

        if not paper_mode:
            if mt5 is None:
                logger.error("❌ MT5 package not installed. Cannot run in LIVE mode.")
                raise Exception("MT5 package missing")
                
            if not mt5.initialize():
                logger.error("❌ MT5 initialization failed. Is MT5 terminal running?")
                raise Exception("MT5 connection failed")
            account_info = mt5.account_info()
            logger.success(f"✅ MT5 Connected | Account: {account_info.login} | LIVE MODE")
            logger.info(f"💰 Balance: ${account_info.balance:.2f} | Equity: ${account_info.equity:.2f} | PnL: ${account_info.profit:.2f}")
    async def execute_order(self, symbol: str, action: str, lot: float, sl: float = None, tp: float = None):
        """Main execution — ALWAYS gated by RiskEngine with live metrics"""
        current_balance = 10000.0
        open_positions = 0
        
        # Inject live numbers into the risk guardrails if connected
        if not self.paper_mode and mt5 is not None:
            account_info = mt5.account_info()
            if account_info:
                current_balance = account_info.balance
            open_positions = mt5.positions_total() or 0
        
        if not self.risk.can_trade(current_balance, action.upper(), 1.0, 1.0, current_open_positions=open_positions):
            logger.warning(f"🚫 RiskEngine BLOCKED {action} {lot} {symbol}")
            return {"status": "risk_blocked"}

        if self.paper_mode:
            logger.info(f"📄 [PAPER MODE] {action} {lot:.2f} {symbol} | SL:{sl} TP:{tp}")
            await self._log_to_redis(symbol, action, lot, sl, tp)
            return {"status": "paper_executed", "paper": True}

        # === LIVE EXECUTION ===
        price = mt5.symbol_info_tick(symbol).ask if action == "BUY" else mt5.symbol_info_tick(symbol).bid

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot,
            "type": mt5.ORDER_TYPE_BUY if action.upper() == "BUY" else mt5.ORDER_TYPE_SELL,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": 30,
            "magic": 424242,
            "comment": "Grok-AGI Live",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.success(f"✅ LIVE FILLED → {action} {lot:.2f} {symbol} | Ticket: {result.order}")
            await self._log_to_redis(symbol, action, lot, sl, tp)
            return {"status": "live_executed", "ticket": result.order}
        else:
            logger.error(f"❌ MT5 failed: {result.comment}")
            return {"status": "failed", "error": result.comment}

    async def _log_to_redis(self, symbol, action, lot, sl, tp):
        state = {
            "symbol": symbol,
            "action": action,
            "lot": lot,
            "sl": sl,
            "tp": tp,
            "timestamp": asyncio.get_running_loop().time()
        }
        try:
            await asyncio.to_thread(self.redis.hset, "last_trade", mapping=state)
            logger.debug(f"Redis state updated: {state}")
        except Exception as e:
            logger.warning(f"Redis not available for state saving: {e}")
