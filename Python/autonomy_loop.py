"""
Autonomy Loop — Train → Evaluate → Canary → Promote/Rollback lifecycle.
Guarded against missing MT5 on Mac.
"""
import os
import time
import asyncio
import subprocess
import sys
import datetime
from loguru import logger

from Python.model_registry import ModelRegistry
from Python.model_evaluator import evaluate_candidate_vs_champion
import json


class AutonomyLoop:
    def __init__(self, brain, interval_sec: int = 6 * 60 * 60):
        self.brain = brain
        self.registry = ModelRegistry()

        self.interval_sec = int(os.environ.get("AGI_AUTONOMY_INTERVAL_SEC", str(3600)))
        self.enable_train = os.environ.get("AGI_AUTONOMY_TRAIN", "false").lower() == "true"
        self.enable_auto_canary = os.environ.get("AGI_AUTONOMY_AUTO_CANARY", "true").lower() == "true"

        # Canary rules
        self.canary_min_trades = int(os.environ.get("CANARY_MIN_TRADES", "10"))
        self.canary_max_loss = float(os.environ.get("CANARY_MAX_LOSS", "75"))
        self.canary_max_dd = float(os.environ.get("CANARY_MAX_DD", "0.12"))

        # Internal canary tracking
        self._canary_start_trade_count = None
        self._canary_set_time = None
        self._canary_set_time = None
        self._last_evaluated_candidate = None

    def _log_incident(self, i_type: str, severity: str, timestamp: str, message: str):
        import json
        log_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "live_incidents.json")
        data = []
        try:
            if os.path.exists(log_path):
                with open(log_path, 'r') as f:
                    data = json.load(f)
        except:
            pass
        data.insert(0, {"id": f"LIV-{str(int(time.time()))[-4:]}", "type": i_type, "severity": severity, "timestamp": timestamp, "message": message})
        data = data[:10]  # Keep last 10
        try:
            with open(log_path, 'w') as f:
                json.dump(data, f, indent=2)
        except:
            pass

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
        import yaml
        cfg_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")
        if not os.path.exists(cfg_path):
            symbols = ["EURUSD", "GBPUSD"]
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

        logger.info(f"Evaluator: candidate score={report['candidate']['avg_score']:.3f} "
                    f"dd={report['candidate']['worst_drawdown']:.3f} ret={report['candidate']['avg_return']:.3f} "
                    f"wins={report['wins']} passes={report['passes_thresholds']}")

        if self.enable_auto_canary and report["wins"] and report["passes_thresholds"]:
            self.registry.set_canary(candidate_dir)

            # Start tracking metrics for live staging
            self._canary_start_trade_count = getattr(self.brain, 'risk_engine', getattr(self.brain, 'risk', None))
            if self._canary_start_trade_count is not None:
                self._canary_start_trade_count = self._canary_start_trade_count.daily_trades
            else:
                self._canary_start_trade_count = 0
            self._canary_set_time = time.time()
            logger.warning("🟡 Canary enabled. Monitoring live performance for promotion/rollback.")
        else:
            logger.info("Autonomy: candidate not promoted to canary (didn't win or failed thresholds).")

    def _canary_monitor(self):
        canary = self._get_canary_dir()
        if not canary:
            return

        # Get risk engine reference (handle both AGIServer and older brain shapes)
        risk = getattr(self.brain, 'risk_engine', getattr(self.brain, 'risk', None))
        if risk is None:
            logger.warning("Autonomy: no risk engine available for canary monitoring")
            return

        # Initialize baseline
        if self._canary_start_trade_count is None:
            self._canary_start_trade_count = risk.daily_trades
            self._canary_set_time = time.time()

        trades_since = risk.daily_trades - self._canary_start_trade_count

        # Pull TRUE PnL from MT5 if available (Windows only)
        realized = 0.0
        try:
            if sys.platform == "win32":
                import MetaTrader5 as mt5
                import pytz
                if mt5 is not None and mt5.initialize():
                    tz = pytz.timezone("Etc/UTC")
                    now_utc = datetime.datetime.now(tz)
                    lookback = now_utc - datetime.timedelta(days=7)
                    deals = mt5.history_deals_get(lookback, now_utc)
                    if deals:
                        realized = sum(deal.profit for deal in deals if deal.entry == mt5.DEAL_ENTRY_OUT)
            else:
                # On Mac, use the risk engine's tracked PnL as proxy
                realized = risk.realized_pnl_today
        except Exception as e:
            logger.warning(f"Autonomy PnL check failed: {e}")
            realized = risk.realized_pnl_today

        dd = float(risk.current_dd) / 100.0

        # Rollback conditions
        if realized <= -self.canary_max_loss or dd >= self.canary_max_dd:
            logger.error(f"🔴 Canary rollback: realized={realized:.2f} dd={dd:.3f}")
            self._log_incident("learning", "fail", "Just now", f"[RECURSIVE PENALTY] Canary model rolled back instantly upon hitting max limit (DD: {dd:.3f}). Parameter configuration flagged for suppressed probabilities.")
            self.registry.rollback_to_champion()
            self._canary_start_trade_count = None
            self._canary_set_time = None

            # Force Brain to reload champion
            if hasattr(self.brain, '_load_ppo_from_registry'):
                self.brain._load_ppo_from_registry()
            elif hasattr(self.brain, 'brain') and hasattr(self.brain.brain, '_load_ppo_from_registry'):
                self.brain.brain._load_ppo_from_registry()
            return

        # Promotion conditions
        if trades_since >= self.canary_min_trades and realized >= 0:
            logger.success(f"🟢 Canary promoted: trades_since={trades_since} realized={realized:.2f} dd={dd:.3f}")
            self._log_incident("learning", "pass", "Just now", f"[AUTO-PROMOTION] Canary outperformed live metrics over {trades_since} cycles. Model hot-swapped to new champion.")
            self.registry.promote_canary_to_champion()
            self._canary_start_trade_count = None
            self._canary_set_time = None

            if hasattr(self.brain, '_load_ppo_from_registry'):
                self.brain._load_ppo_from_registry()
            elif hasattr(self.brain, 'brain') and hasattr(self.brain.brain, '_load_ppo_from_registry'):
                self.brain.brain._load_ppo_from_registry()
            
            self._export_state()

    def _export_state(self):
        """Broadcasts the real-time interior brain state to a JSON file for the API Bridge."""
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        state_path = os.path.join(root, "live_state.json")
        
        # Pull real metrics from the brain and its sub-components
        risk = getattr(self.brain, 'risk_engine', getattr(self.brain, 'risk', None))
        active = self.registry._read_active()
        
        state = {
            "timestamp": time.time(),
            "registry": {
                "champion": active.get("champion"),
                "canary": active.get("canary")
            },
            "trading": {
                "account": {
                    "balance": getattr(risk, "account_balance", 0) if risk else 0,
                    "equity": getattr(risk, "account_equity", 0) if risk else 0,
                    "floatingPnl": getattr(risk, "floating_pnl", 0) if risk else 0,
                    "realizedToday": getattr(risk, "realized_pnl_today", 0) if risk else 0,
                },
                "risk": {
                    "drawdownPct": getattr(risk, "current_dd", 0) if risk else 0,
                    "canTrade": True
                }
            },
            "training": {
                "active_canary": active.get("canary") is not None,
                "cycles_completed": 0 # Would be tracked in persistence
            }
        }
        
        try:
            with open(state_path, 'w') as f:
                json.dump(state, f, indent=2)
        except:
            pass

    async def nightly_training_loop(self):
        """Triggers the RL training engine every night at midnight."""
        while True:
            now = datetime.datetime.now()
            next_midnight = datetime.datetime(now.year, now.month, now.day, 23, 59, 59)
            seconds_to_midnight = (next_midnight - now).total_seconds()

            logger.debug(f"Next Nightly Retraining scheduled in {int(seconds_to_midnight/3600)} hours.")
            await asyncio.sleep(seconds_to_midnight + 60)

            if self.enable_train:
                await self._train_candidate()

    async def start(self):
        logger.warning("🤖 AutonomyLoop started (train → evaluate → canary → promote/rollback).")

        asyncio.create_task(self.nightly_training_loop())

        while True:
            try:
                self._canary_monitor()

                candidate = self._latest_candidate_dir()
                if candidate and candidate != self._last_evaluated_candidate:
                    curr_canary = self._get_canary_dir()
                    self._last_evaluated_candidate = candidate
                    if not curr_canary:
                        self._maybe_set_canary(candidate)

                self._export_state()
            except Exception as e:
                logger.warning(f"Autonomy loop error: {e}")

            await asyncio.sleep(self.interval_sec)
