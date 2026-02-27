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
        realized_pnl = 0.0
        
        # Inject live numbers into the risk guardrails if connected
        if not self.paper_mode and mt5 is not None:
            account_info = mt5.account_info()
            if account_info:
                current_balance = account_info.balance
                
            open_positions = mt5.positions_total() or 0
            
            # Extract actual daily Realized PnL natively from MT5 History (Required for real kill switch)
            import datetime
            now = datetime.datetime.now()
            midnight = datetime.datetime(now.year, now.month, now.day)
            deals = mt5.history_deals_get(midnight, now)
            if deals:
                # MT5 deals record profit on the OUT entries (closings)
                realized_pnl = sum(deal.profit for deal in deals if deal.entry == mt5.DEAL_ENTRY_OUT)
        
        if not self.risk.can_trade(current_balance, action.upper(), 1.0, 1.0, current_open_positions=open_positions, realized_pnl=realized_pnl):
            logger.warning(f"🚫 RiskEngine BLOCKED {action} {lot} {symbol} | Live PnL Today: ${realized_pnl:.2f}")
            return {"status": "risk_blocked"}

        if self.paper_mode:
            logger.info(f"📄 [PAPER MODE] {action} {lot:.2f} {symbol} | SL:{sl} TP:{tp}")
            await self._log_to_redis(symbol, action, lot, sl, tp)
            return {"status": "paper_executed", "paper": True}

        # === LIVE EXECUTION ===
        # Ensure symbol is in Market Watch
        if not mt5.symbol_select(symbol, True):
            logger.error(f"❌ MT5 failed to select symbol {symbol}. Does your broker use a suffix (like EURUSD.pro)?")
            return {"status": "failed", "error": f"Symbol {symbol} not found in MT5 Market Watch"}
            
        tick = mt5.symbol_info_tick(symbol)
        symbol_info = mt5.symbol_info(symbol)
        if tick is None or symbol_info is None:
            logger.error(f"❌ MT5 returned no tick data for {symbol}. Market closed or invalid symbol.")
            return {"status": "failed", "error": f"No tick data for {symbol}"}
            
        price = tick.ask if action.upper() == "BUY" else tick.bid
        
        # Calculate dynamic Lot size securely using LIVE MT5 Balance and Symbol Steps
        lot = self.risk.lot_size(balance=current_balance, price=price)
        lot_step = symbol_info.volume_step
        lot = round(lot / lot_step) * lot_step
        lot = max(symbol_info.volume_min, min(lot, symbol_info.volume_max))
        lot = round(lot, 2)  # Avoid Python float glitches
        
        # Calculate accurate stops natively using MT5 points to prevent Invalid Stops error
        point = symbol_info.point
        sl_points = 50 * (10 if symbol_info.digits in [3, 5] else 1) * point
        rr_ratio = 2.0
        tp_points = sl_points * rr_ratio
        
        if action.upper() == "BUY":
            sl = round(price - sl_points, symbol_info.digits)
            tp = round(price + tp_points, symbol_info.digits)
        else:
            sl = round(price + sl_points, symbol_info.digits)
            tp = round(price - tp_points, symbol_info.digits)

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
