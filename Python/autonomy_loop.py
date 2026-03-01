import os
import time
import asyncio
import subprocess
import sys
import datetime
from loguru import logger

from Python.model_registry import ModelRegistry
from Python.model_evaluator import evaluate_candidate_vs_champion

class AutonomyLoop:
    def __init__(self, brain, interval_sec: int = 6 * 60 * 60):
        self.brain = brain
        self.registry = ModelRegistry()

        self.interval_sec = int(os.environ.get("AGI_AUTONOMY_INTERVAL_SEC", str(3600))) # Hourly check
        self.enable_train = os.environ.get("AGI_AUTONOMY_TRAIN", "false").lower() == "true"
        self.enable_auto_canary = os.environ.get("AGI_AUTONOMY_AUTO_CANARY", "true").lower() == "true"

        # Canary rules
        self.canary_min_trades = int(os.environ.get("CANARY_MIN_TRADES", "10"))
        self.canary_max_loss = float(os.environ.get("CANARY_MAX_LOSS", "75"))  # realized PnL stop
        self.canary_max_dd = float(os.environ.get("CANARY_MAX_DD", "0.12"))

        # Internal canary tracking
        self._canary_start_trade_count = None
        self._canary_set_time = None
        self._last_evaluated_candidate = None

    def _latest_candidate_dir(self):
        root = self.registry.candidates_dir
        dirs = []
        if not os.path.exists(root):
            return None
        for d in os.listdir(root):
            p = os.path.join(root, d)
            if os.path.isdir(p):
                dirs.append(p)
        if not dirs:
            return None
        dirs.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return dirs[0] if dirs else None

    def _get_champion_dir(self):
        active = self.registry._read_active()
        return active.get("champion")

    def _get_canary_dir(self):
        active = self.registry._read_active()
        return active.get("canary")

    async def _train_candidate(self):
        logger.info("🌙 Autonomy: Nightly training candidate (train_drl.py)...")
        subprocess.check_call([sys.executable, "training/train_drl.py"])

    def _maybe_set_canary(self, candidate_dir: str):
        # Evaluate candidate vs champion
        import yaml
        cfg_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")
        if not os.path.exists(cfg_path):
            symbols = ["EURUSDm", "GBPUSDm"]
            eval_period = "120d"
        else:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            symbols = cfg.get("trading", {}).get("symbols", ["EURUSD", "GBPUSD"])
            eval_period = cfg.get("drl", {}).get("eval_period", "120d")
            
        champ_dir = self._get_champion_dir()

        logger.info("Executing Evaluator Simulation for Candidate against Champion...")
        report = evaluate_candidate_vs_champion(
            candidate_dir=candidate_dir,
            champion_dir=champ_dir,
            symbols=symbols,
            period=eval_period
        )

        if report.get("error"):
            logger.warning(f"Autonomy: evaluator error: {report['error']}")
            return

        logger.info(f"Evaluator: candidate score={report['candidate']['score']:.3f} "
                    f"dd={report['candidate']['max_drawdown']:.3f} ret={report['candidate']['total_return']:.3f} "
                    f"wins={report['wins']} passes={report['passes_thresholds']}")

        if self.enable_auto_canary and report["wins"] and report["passes_thresholds"]:
            self.registry.set_canary(candidate_dir)
            
            # Start tracking metrics for live staging
            self._canary_start_trade_count = self.brain.risk_engine.daily_trades
            self._canary_set_time = time.time()
            logger.warning("🟡 Canary enabled. Monitoring live performance for promotion/rollback.")
        else:
            logger.info("Autonomy: candidate not promoted to canary (didn't win or failed thresholds).")

    def _canary_monitor(self):
        canary = self._get_canary_dir()
        if not canary:
            return

        # initialize baseline
        if self._canary_start_trade_count is None:
            self._canary_start_trade_count = self.brain.risk_engine.daily_trades
            self._canary_set_time = time.time()

        trades_since = self.brain.risk_engine.daily_trades - self._canary_start_trade_count
        
        # Pull TRUE PnL natively from MT5 just like the Kill Switch
        realized = 0.0
        try:
            import MetaTrader5 as mt5
            import pytz
            if mt5 is not None and mt5.initialize():
                # MT5 History needs proper timestamps; if not provided timezone-aware, it assumes UTC
                tz = pytz.timezone("Etc/UTC")
                now_utc = datetime.datetime.now(tz)
                # Check realized PnL over the last 7 days as a proxy for the canary's lifetime
                lookback = now_utc - datetime.timedelta(days=7)
                deals = mt5.history_deals_get(lookback, now_utc)
                if deals:
                    realized = sum(deal.profit for deal in deals if deal.entry == mt5.DEAL_ENTRY_OUT)
        except Exception as e:
            logger.warning(f"Autonomy MT5 PnL check failed: {e}")

        dd = float(self.brain.risk_engine.current_dd) / 100.0

        # rollback conditions
        if realized <= -self.canary_max_loss or dd >= self.canary_max_dd:
            logger.error(f"🔴 Canary rollback: realized={realized:.2f} dd={dd:.3f}")
            self.registry.rollback_to_champion()
            self._canary_start_trade_count = None
            self._canary_set_time = None
            
            # Force Brain to reload champion over the failed canary
            self.brain._load_ppo_from_registry()
            return

        # promotion conditions
        if trades_since >= self.canary_min_trades and realized >= 0:
            logger.success(f"🟢 Canary promoted: trades_since={trades_since} realized={realized:.2f} dd={dd:.3f}")
            self.registry.promote_canary_to_champion()
            self._canary_start_trade_count = None
            self._canary_set_time = None
            
            # Force Brain to latch to active champion state
            self.brain._load_ppo_from_registry()

    async def nightly_training_loop(self):
        """Triggers the RL training engine every night at midnight."""
        while True:
            now = datetime.datetime.now()
            next_midnight = datetime.datetime(now.year, now.month, now.day, 23, 59, 59)
            seconds_to_midnight = (next_midnight - now).total_seconds()
            
            logger.debug(f"Next Nightly Retraining scheduled in {int(seconds_to_midnight/3600)} hours.")
            await asyncio.sleep(seconds_to_midnight + 60) # Wait until exactly midnight
            
            if self.enable_train:
                await self._train_candidate()

    async def start(self):
        logger.warning("🤖 AutonomyLoop started (train → evaluate → canary → promote/rollback).")
        
        # Bind the hourly monitoring loop and the nightly retrainer concurrently!
        asyncio.create_task(self.nightly_training_loop())
        
        while True:
            try:
                # always monitor canary actively for risk conditions
                self._canary_monitor()

                # Search exactly once an hour for new candidates and evaluate them
                candidate = self._latest_candidate_dir()
                if candidate and candidate != self._last_evaluated_candidate:
                    curr_canary = self._get_canary_dir()
                    
                    # Store as evaluated to prevent infinite rejection evaluation loops
                    self._last_evaluated_candidate = candidate
                    
                    if not curr_canary:
                        self._maybe_set_canary(candidate)

            except Exception as e:
                logger.warning(f"Autonomy loop error: {e}")

            await asyncio.sleep(self.interval_sec)
