"""
Telegram alerts via simple HTTP API — sends MT5-style execution messages.
Format: Executed! (One Way)
        buy EURUSD Q=0.1 SL=2% TP=4% M=505
"""
import os
import requests
import yaml
from datetime import datetime
from loguru import logger

# Load config (env vars override config.yaml)
_config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")
try:
    with open(_config_path) as f:
        _cfg = yaml.safe_load(f)
except Exception:
    _cfg = {"telegram": {}, "trading": {}}

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN") or _cfg.get("telegram", {}).get("token", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID") or _cfg.get("telegram", {}).get("chat_id", "")

# MT5 magic number for identifying AGI orders
MAGIC_NUMBER = _cfg.get("trading", {}).get("magic_number", 505)


def tg_send(text: str) -> bool:
    """Send a message via Telegram HTTP API. Returns True on success."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID or "YOUR_" in TELEGRAM_TOKEN:
        logger.warning("Telegram disabled: missing or placeholder token/chat_id")
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        r = requests.post(url, json={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": text,
            "parse_mode": "HTML",
        }, timeout=10)
        if r.status_code != 200:
            logger.error(f"Telegram send failed {r.status_code}: {r.text}")
            return False
        logger.success(f"Telegram sent: {text[:80]}...")
        return True
    except Exception as e:
        logger.error(f"Telegram exception: {e}")
        return False


# ── MT5-style trade execution alerts ────────────────────────────────

def send_trade_alert(signal: str, symbol: str, price: float, confidence: float,
                     lots: float = 0.01, sl_pct: float = 2.0, tp_pct: float = 4.0,
                     magic: int = None):
    """
    Send an MT5-style execution alert.

    Output format:
        Executed! (One Way)
        buy EURUSD Q=0.1 SL=2% TP=4% M=505
    """
    if magic is None:
        magic = MAGIC_NUMBER

    action = signal.lower()  # buy / sell
    ts = datetime.now().strftime("%H:%M:%S")

    msg = (
        f"<b>Executed!</b> (One Way)\n"
        f"\n"
        f"<code>{action} {symbol} Q={lots} SL={sl_pct}% TP={tp_pct}% M={magic}</code>\n"
        f"\n"
        f"📊 Price: {price:.5f}\n"
        f"🎯 Confidence: {confidence:.2%}\n"
        f"🕐 {ts}"
    )
    return tg_send(msg)


def send_trade_rejected(symbol: str, reason: str = "Invalid Params",
                        details: str = ""):
    """
    Send a rejection / invalid params alert.

    Output format:
        Invalid Params
        EURUSD — reason
    """
    msg = (
        f"<b>{reason}</b>\n"
        f"\n"
        f"<code>{symbol}</code>"
    )
    if details:
        msg += f"\n{details}"
    return tg_send(msg)


def send_hold_alert(symbol: str, price: float, confidence: float):
    """Send a HOLD notification (low confidence or risk blocked)."""
    msg = (
        f"⏸ <b>HOLD</b> {symbol}\n"
        f"Price: {price:.5f} | Conf: {confidence:.2%}"
    )
    return tg_send(msg)


def send_startup_alert():
    """Send server startup notification."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return tg_send(
        f"✅ <b>AGI Trading Server LIVE</b>\n"
        f"Real data mode active\n"
        f"Started: {ts}"
    )


def send_daily_summary(trades: int, holds: int, pnl: float = 0.0):
    """Send end-of-day summary."""
    msg = (
        f"📋 <b>Daily Summary</b>\n"
        f"\n"
        f"Trades: {trades}\n"
        f"Holds: {holds}\n"
        f"PnL: ${pnl:+.2f}"
    )
    return tg_send(msg)


# ── Backward compat aliases ─────────────────────────────────────────
def send_alert(signal: str, confidence: float, pnl: float = 0.0):
    return tg_send(f"🚨 {signal} | conf={confidence:.2%} | PnL=${pnl:.2f}")
