"""Telegram alerts via simple HTTP API — no async, no library issues."""
import os
import requests
import yaml
from loguru import logger

# Load config (env vars override config.yaml)
_config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")
try:
    with open(_config_path) as f:
        _cfg = yaml.safe_load(f)
except Exception:
    _cfg = {"telegram": {}}

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN") or _cfg.get("telegram", {}).get("token", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID") or _cfg.get("telegram", {}).get("chat_id", "")


def tg_send(text: str) -> bool:
    """Send a message via Telegram HTTP API. Returns True on success."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID or "YOUR_" in TELEGRAM_TOKEN:
        logger.warning("Telegram disabled: missing or placeholder token/chat_id")
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        r = requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": text}, timeout=10)
        if r.status_code != 200:
            logger.error(f"Telegram send failed {r.status_code}: {r.text}")
            return False
        logger.success(f"Telegram sent: {text[:60]}...")
        return True
    except Exception as e:
        logger.error(f"Telegram exception: {e}")
        return False


def send_trade_alert(signal: str, symbol: str, price: float, confidence: float):
    """Send a formatted trade alert."""
    emoji = "📈" if signal == "BUY" else "📉"
    msg = (
        f"{emoji} AGI TRADE EXECUTED\n"
        f"Signal: {signal}\n"
        f"Symbol: {symbol}\n"
        f"Price: {price:.5f}\n"
        f"Confidence: {confidence:.2%}"
    )
    return tg_send(msg)


def send_startup_alert():
    """Send server startup notification."""
    return tg_send("✅ AGI Trading Server started — real data mode active")


# Backward compat alias
def send_alert(signal: str, confidence: float, pnl: float = 0.0):
    return tg_send(f"🚨 {signal} | conf={confidence:.2%} | PnL=${pnl:.2f}")
