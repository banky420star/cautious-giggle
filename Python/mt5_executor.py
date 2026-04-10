"""
MT5 Executor — Trade execution with conditional MT5 import and dry-run mode.
Automatically falls back to DryRunExecutor on Mac/Linux.
"""
import sys
import os
from loguru import logger

# ── Conditional MT5 import ──────────────────────────────────────────
_mt5 = None
if sys.platform == "win32":
    try:
        import MetaTrader5 as mt5
        _mt5 = mt5
    except ImportError:
        logger.warning("MetaTrader5 not installed on Windows — using dry-run mode")


class MT5Executor:
    """Live MT5 execution (Windows only)."""

    def __init__(self, risk):
        self.risk = risk
        self._is_live = _mt5 is not None and sys.platform == "win32"

        if self._is_live:
            try:
                if not _mt5.initialize():
                    logger.error("MT5 initialize() failed — falling back to dry-run")
                    self._is_live = False
            except Exception as e:
                logger.error(f"MT5 init error: {e}")
                self._is_live = False

        if self._is_live:
            logger.success("MT5Executor: LIVE mode — connected to MetaTrader 5")
        else:
            logger.info("MT5Executor: DRY-RUN mode — trades will be logged only")

    def get_positions(self, symbol):
        longs = []
        shorts = []

        if not self._is_live:
            return longs, shorts

        positions = _mt5.positions_get(symbol=symbol)
        if positions:
            for p in positions:
                if p.type == 0:
                    longs.append(p)
                else:
                    shorts.append(p)
        return longs, shorts

    def reconcile_exposure(self, symbol, target_exposure, max_lots):
        if not self.risk.can_trade():
            return

        if not self._is_live:
            # Dry-run: just log the intended trade
            target_lots = round(target_exposure * max_lots, 2)
            direction = "BUY" if target_exposure > 0 else "SELL" if target_exposure < 0 else "FLAT"
            logger.info(
                f"📋 DRY-RUN TRADE: {symbol} | {direction} | "
                f"exposure={target_exposure:.4f} | lots={abs(target_lots):.2f}"
            )
            self.risk.record_trade()
            return

        # ── Live MT5 execution ──
        longs, shorts = self.get_positions(symbol)

        long_lots = sum(p.volume for p in longs)
        short_lots = sum(p.volume for p in shorts)

        net_lots = long_lots - short_lots
        target_lots = round(target_exposure * max_lots, 2)
        delta = target_lots - net_lots

        if abs(delta) < 0.01:
            return

        if delta > 0:
            if short_lots > 0:
                self.close_positions(shorts)
            self.open_position(symbol, _mt5.ORDER_TYPE_BUY, abs(delta))
        else:
            if long_lots > 0:
                self.close_positions(longs)
            self.open_position(symbol, _mt5.ORDER_TYPE_SELL, abs(delta))

        self.risk.record_trade()

    def close_positions(self, positions):
        if not self._is_live:
            return

        for p in positions:
            request = {
                "action": _mt5.TRADE_ACTION_DEAL,
                "symbol": p.symbol,
                "volume": p.volume,
                "type": _mt5.ORDER_TYPE_SELL if p.type == 0 else _mt5.ORDER_TYPE_BUY,
                "position": p.ticket
            }
            result = _mt5.order_send(request)
            if result.retcode != _mt5.TRADE_RETCODE_DONE:
                self.risk.record_error()

    def open_position(self, symbol, order_type, volume):
        if not self._is_live:
            return

        tick = _mt5.symbol_info_tick(symbol)
        if tick is None:
            logger.error(f"Cannot get tick for {symbol}")
            self.risk.record_error()
            return

        request = {
            "action": _mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "price": tick.ask if order_type == _mt5.ORDER_TYPE_BUY else tick.bid,
        }
        result = _mt5.order_send(request)
        if result.retcode != _mt5.TRADE_RETCODE_DONE:
            self.risk.record_error()
