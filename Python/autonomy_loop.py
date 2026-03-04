import asyncio
import datetime
import os
import subprocess
import sys
import time

from loguru import logger

from Python.model_evaluator import evaluate_candidate_vs_champion
from Python.model_registry import ModelRegistry
from alerts.telegram_alerts import TelegramAlerter


class AutonomyLoop:
    def __init__(self, brain, interval_sec: int = 6 * 60 * 60):
        self.brain = brain
        self.registry = ModelRegistry()

        self.interval_sec = int(os.environ.get("AGI_AUTONOMY_INTERVAL_SEC", str(3600)))
        self.enable_train = os.environ.get("AGI_AUTONOMY_TRAIN", "true").lower() == "true"
        self.enable_auto_canary = os.environ.get("AGI_AUTONOMY_AUTO_CANARY", "true").lower() == "true"

        self.canary_min_trades = int(os.environ.get("CANARY_MIN_TRADES", "10"))
        self.canary_max_loss = float(os.environ.get("CANARY_MAX_LOSS", "75"))
        self.canary_max_dd = float(os.environ.get("CANARY_MAX_DD", "0.12"))

        self._canary_start_trade_count = None
        self._canary_set_time = None
        self._last_evaluated_candidate = None

        self.alerter = self._init_alerter()

    def _init_alerter(self):
        token = os.environ.get("TELEGRAM_TOKEN")
        chat_id = os.environ.get("TELEGRAM_CHAT_ID")

        cfg_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")
        if os.path.exists(cfg_path):
            try:
                import yaml

                with open(cfg_path, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
                tel = cfg.get("telegram", {}) or {}
                token = token or tel.get("token")
                chat_id = chat_id or tel.get("chat_id")
            except Exception:
                pass

        if token in (None, "", "YOUR_BOT_TOKEN_HERE") or chat_id in (None, "", "YOUR_CHAT_ID_HERE"):
            return TelegramAlerter(None, None)
        return TelegramAlerter(token, chat_id)

    def _notify(self, message: str):
        try:
            self.alerter.alert(message)
        except Exception:
            pass

    def _latest_candidate_dir(self):
        root = self.registry.candidates_dir
        if not os.path.exists(root):
            return None
        dirs = [os.path.join(root, d) for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
        if not dirs:
            return None
        dirs.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return dirs[0]

    def _get_champion_dir(self):
        return self.registry._read_active().get("champion")

    def _get_canary_dir(self):
        return self.registry._read_active().get("canary")

    async def _train_candidate(self):
        self._notify("Autonomy training started: LSTM + PPO")
        subprocess.check_call([sys.executable, "training/train_lstm.py"])
        subprocess.check_call([sys.executable, "training/train_drl.py"])
        self._notify("Autonomy training completed")

    def _maybe_reload_brain(self):
        if hasattr(self.brain, "_load_ppo_from_registry"):
            try:
                self.brain._load_ppo_from_registry()
            except Exception as exc:
                logger.warning(f"brain reload failed: {exc}")

    def _maybe_set_canary(self, candidate_dir: str):
        import yaml

        cfg_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")
        if not os.path.exists(cfg_path):
            symbols = ["EURUSDm", "GBPUSDm"]
            eval_period = "120d"
        else:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            symbols = cfg.get("trading", {}).get("symbols", ["EURUSDm", "GBPUSDm"])
            eval_period = cfg.get("drl", {}).get("eval_period", "120d")

        report = evaluate_candidate_vs_champion(
            candidate_dir=candidate_dir,
            champion_dir=self._get_champion_dir(),
            symbols=symbols,
            period=eval_period,
        )

        if report.get("error"):
            logger.warning(f"Autonomy evaluator error: {report['error']}")
            self._notify(f"Autonomy evaluator error: {report['error']}")
            return

        if self.enable_auto_canary and report["wins"] and report["passes_thresholds"]:
            self.registry.set_canary(candidate_dir)
            risk = getattr(self.brain, "risk_engine", None)
            self._canary_start_trade_count = int(getattr(risk, "daily_trades", 0))
            self._canary_set_time = time.time()
            self._notify(f"Canary enabled: {os.path.basename(candidate_dir)}")
        else:
            logger.info("candidate not promoted")

    def _canary_monitor(self):
        canary = self._get_canary_dir()
        if not canary:
            return

        risk = getattr(self.brain, "risk_engine", None)
        trades_now = int(getattr(risk, "daily_trades", 0))

        if self._canary_start_trade_count is None:
            self._canary_start_trade_count = trades_now
            self._canary_set_time = time.time()

        trades_since = trades_now - self._canary_start_trade_count

        realized = 0.0
        try:
            import MetaTrader5 as mt5
            import pytz

            if mt5 is not None and mt5.initialize():
                tz = pytz.timezone("Etc/UTC")
                now_utc = datetime.datetime.now(tz)
                lookback = now_utc - datetime.timedelta(days=7)
                deals = mt5.history_deals_get(lookback, now_utc)
                if deals:
                    realized = sum(deal.profit for deal in deals if deal.entry == mt5.DEAL_ENTRY_OUT)
        except Exception as exc:
            logger.warning(f"Autonomy MT5 PnL check failed: {exc}")

        dd_pct = float(getattr(risk, "current_dd", 0.0)) / 100.0

        if realized <= -self.canary_max_loss or dd_pct >= self.canary_max_dd:
            self.registry.rollback_to_champion()
            self._canary_start_trade_count = None
            self._canary_set_time = None
            self._maybe_reload_brain()
            self._notify(f"Canary rollback triggered. realized={realized:.2f}, dd={dd_pct:.3f}")
            return

        if trades_since >= self.canary_min_trades and realized >= 0:
            self.registry.promote_canary_to_champion()
            self._canary_start_trade_count = None
            self._canary_set_time = None
            self._maybe_reload_brain()
            self._notify(f"Canary promoted. trades={trades_since}, realized={realized:.2f}")

    async def nightly_training_loop(self):
        while True:
            now = datetime.datetime.now()
            next_midnight = datetime.datetime(now.year, now.month, now.day, 23, 59, 59)
            seconds_to_midnight = (next_midnight - now).total_seconds()
            await asyncio.sleep(max(60, seconds_to_midnight + 60))
            if self.enable_train:
                await self._train_candidate()

    async def start(self):
        logger.warning("AutonomyLoop started")
        self._notify("AutonomyLoop started")
        asyncio.create_task(self.nightly_training_loop())

        while True:
            try:
                self._canary_monitor()
                candidate = self._latest_candidate_dir()
                if candidate and candidate != self._last_evaluated_candidate:
                    self._last_evaluated_candidate = candidate
                    if not self._get_canary_dir():
                        self._maybe_set_canary(candidate)
            except Exception as exc:
                logger.warning(f"Autonomy loop error: {exc}")
                self._notify(f"Autonomy loop error: {exc}")

            await asyncio.sleep(self.interval_sec)
