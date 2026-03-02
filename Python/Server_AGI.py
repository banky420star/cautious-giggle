import os
import time
import MetaTrader5 as mt5
from datetime import datetime
from Python.risk_engine import RiskEngine
from Python.mt5_executor import MT5Executor
from Python.hybrid_brain import HybridBrain
from alerts.telegram_alerts import TelegramAlerter

def main(live=False):
    if live:
        os.environ["AGI_IS_LIVE"] = "1"

    mt5.initialize()

    risk = RiskEngine()
    executor = MT5Executor(risk)
    brain = HybridBrain(risk, executor)

    alerter = TelegramAlerter(
        os.environ.get("TELEGRAM_TOKEN"),
        os.environ.get("TELEGRAM_CHAT_ID")
    )

    start_time = time.time()

    while True:
        uptime = int(time.time() - start_time)

        alerter.heartbeat(
            uptime=str(uptime) + " sec",
            mt5_connected=mt5.initialize(),
            trading_enabled=not risk.halt
        )

        time.sleep(120)

if __name__ == "__main__":
    import sys
    live_flag = "--live" in sys.argv
    main(live=live_flag)
