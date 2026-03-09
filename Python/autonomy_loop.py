import asyncio
import datetime
import json
import os
import subprocess
import sys
import time

from loguru import logger

from Python.config_utils import load_project_config
from Python.model_evaluator import evaluate_candidate_vs_champion
from Python.model_registry import ModelRegistry
from alerts.telegram_alerts import TelegramAlerter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class AutonomyLoop:
    def __init__(self, brain, interval_sec: int = 6 * 60 * 60):
        self.brain = brain
        self.registry = ModelRegistry()
        self.interval_sec = int(os.environ.get("AGI_AUTONOMY_INTERVAL_SEC", str(3600)))
        self.train_every_sec = int(os.environ.get("AGI_AUTONOMY_TRAIN_EVERY_SEC", "0"))
        self.train_on_start = os.environ.get("AGI_AUTONOMY_TRAIN_ON_START", "false").lower() == "true"
        self.enable_train = os.environ.get("AGI_AUTONOMY_TRAIN", "true").lower() == "true"
        self.enable_auto_canary = os.environ.get("AGI_AUTONOMY_AUTO_CANARY", "true").lower() == "true"

        self.canary_min_trades = int(os.environ.get("CANARY_MIN_TRADES", "10"))
        self.canary_max_loss = float(os.environ.get("CANARY_MAX_LOSS", "75"))
        self.canary_max_dd = float(os.environ.get("CANARY_MAX_DD", "0.12"))
        self.min_score_delta = float(os.environ.get("AGI_GATE_MIN_SCORE_DELTA", "0.25"))
        self.max_eval_dd = float(os.environ.get("AGI_GATE_MAX_EVAL_DD", "0.20"))
        self.min_eval_sharpe = float(os.environ.get("AGI_GATE_MIN_EVAL_SHARPE", "-0.10"))
        self.min_eval_return = float(os.environ.get("AGI_GATE_MIN_EVAL_RETURN", "-0.02"))
        self.require_walkforward = os.environ.get("AGI_GATE_REQUIRE_WALKFORWARD", "false").lower() == "true"
        self.require_papertrade = os.environ.get("AGI_GATE_REQUIRE_PAPERTRADE", "false").lower() == "true"
        self.min_paper_trades = int(os.environ.get("AGI_GATE_MIN_PAPER_TRADES", "20"))
        self._canary_start_trade_count = None
        self._canary_set_time = None
        self._last_evaluated_candidate = None
        self._last_train_ts = 0.0

        self.alerter = self._init_alerter()
        self.eval_config = self._load_evaluation_config()

    def _init_alerter(self):
        token = os.environ.get("TELEGRAM_TOKEN")
        chat_id = os.environ.get("TELEGRAM_CHAT_ID")

        cfg_path = os.path.join(PROJECT_ROOT, "config.yaml")
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

    def _load_evaluation_config(self) -> dict:
        try:
            cfg = load_project_config(PROJECT_ROOT, live_mode=True)
            return cfg.get("evaluation", {}) or {}
        except Exception:
            return {}

    def _notify(self, message: str):
        try:
            self.alerter.alert(message)
        except Exception:
            pass

    def _update_candidate_metadata(self, candidate_dir: str, report: dict, gates_passed: bool, reasons: list[str]):
        payload = {
            "evaluation": {
                "candidate_score": float(report.get("candidate", {}).get("avg_score", 0.0)),
                "winner": bool(report.get("wins", False)),
                "gates_passed": bool(gates_passed),
                "gates_reasons": reasons,
                "forward_windows": report.get("forward_windows", []),
                "per_symbol": report.get("per_symbol_gates", []),
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }
        }
        self.registry.update_metadata(candidate_dir, payload)

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

    def _read_candidate_metadata(self, candidate_dir: str) -> dict:
        meta_path = os.path.join(candidate_dir, "metadata.json")
        if not os.path.exists(meta_path):
            return {}
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
                return payload if isinstance(payload, dict) else {}
        except Exception:
            return {}

    def _evaluate_release_gates(self, candidate_dir: str, report: dict) -> tuple[bool, list[str]]:
        reasons = []
        cand = report.get("candidate") or {}
        champ = report.get("champion") or {}

        cand_dd = float(cand.get("worst_drawdown", 1.0))
        cand_sharpe = float(cand.get("avg_sharpe", -999.0))
        cand_return = float(cand.get("avg_return", -999.0))
        cand_score = float(cand.get("avg_score", 0.0))
        champ_score = float(champ.get("avg_score", 0.0)) if champ else None

        if cand_dd > self.max_eval_dd:
            reasons.append(f"backtest_dd_fail:{cand_dd:.4f}>{self.max_eval_dd:.4f}")
        if cand_sharpe < self.min_eval_sharpe:
            reasons.append(f"backtest_sharpe_fail:{cand_sharpe:.4f}<{self.min_eval_sharpe:.4f}")
        if cand_return < self.min_eval_return:
            reasons.append(f"backtest_return_fail:{cand_return:.4f}<{self.min_eval_return:.4f}")

        if champ_score is not None:
            delta = cand_score - champ_score
            if delta < self.min_score_delta:
                reasons.append(f"score_delta_fail:{delta:.4f}<{self.min_score_delta:.4f}")

        if not report.get("wins"):
            reasons.append("candidate_win_checks_failed")

        for fw in report.get("forward_windows", []):
            if not bool(fw.get("wins")):
                reasons.append(f"forward_{fw.get('period', 'unknown')}_loss")

        meta = self._read_candidate_metadata(candidate_dir)
        if self.require_walkforward:
            wf = meta.get("walkforward", {}) if isinstance(meta, dict) else {}
            if not bool(wf.get("passed", False)):
                reasons.append("walkforward_fail:metadata.walkforward.passed!=true")

        if self.require_papertrade:
            paper = meta.get("paper_trade", {}) if isinstance(meta, dict) else {}
            paper_passed = bool(paper.get("passed", False))
            paper_trades = int(paper.get("trades", 0))
            if not paper_passed:
                reasons.append("paper_trade_fail:metadata.paper_trade.passed!=true")
            if paper_trades < self.min_paper_trades:
                reasons.append(f"paper_trade_min_trades_fail:{paper_trades}<{self.min_paper_trades}")

        return len(reasons) == 0, reasons

    async def _train_candidate(self):
        started = time.time()
        self._notify("Autonomy training started: LSTM + PPO")

        subprocess.check_call([sys.executable, "training/train_lstm.py"], cwd=PROJECT_ROOT)
        self._notify("LSTM training finished")

        subprocess.check_call([sys.executable, "training/train_drl.py"], cwd=PROJECT_ROOT)
        cand = self._latest_candidate_dir()
        elapsed = int(time.time() - started)
        self._notify(
            f"PPO training finished. candidate={os.path.basename(cand) if cand else 'none'} elapsed={elapsed}s"
        )

    def _maybe_reload_brain(self):
        if hasattr(self.brain, "_load_ppo_from_registry"):
            try:
                self.brain._load_ppo_from_registry()
            except Exception as exc:
                logger.warning(f"brain reload failed: {exc}")

    def _maybe_set_canary(self, candidate_dir: str):
        import yaml

        cfg_path = os.path.join(PROJECT_ROOT, "config.yaml")
        if not os.path.exists(cfg_path):
            symbols = ["EURUSDm", "GBPUSDm"]
            eval_period = "120d"
        else:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            symbols = cfg.get("trading", {}).get("symbols", ["EURUSDm", "GBPUSDm"])
            eval_period = cfg.get("drl", {}).get("eval_period", "120d")

            # keep evaluation window bounded for short-interval market data backtests
            try:
                if isinstance(eval_period, str) and eval_period.endswith("d"):
                    d = int(eval_period[:-1])
                    if d > 60:
                        eval_period = "60d"
            except Exception:
                eval_period = "60d"

        report = evaluate_candidate_vs_champion(
            candidate_dir=candidate_dir,
            champion_dir=self._get_champion_dir(),
            symbols=symbols,
            period=eval_period,
            gates=self.eval_config,
            interval="5m",
        )

        if report.get("error"):
            logger.warning(f"Autonomy evaluator error: {report['error']}")
            self._notify(f"Autonomy evaluator error: {report['error']}")
            return

        gates_passed, reasons = self._evaluate_release_gates(candidate_dir, report)
        cand = report.get("candidate") or {}
        summary = (
            f"score={float(cand.get('avg_score', 0.0)):.4f}, "
            f"ret={float(cand.get('avg_return', 0.0)):.4f}, "
            f"dd={float(cand.get('worst_drawdown', 1.0)):.4f}, "
            f"sharpe={float(cand.get('avg_sharpe', -999.0)):.4f}"
        )

        self._update_candidate_metadata(candidate_dir, report, gates_passed, reasons)

        if self.enable_auto_canary and report["wins"] and report["passes_thresholds"] and gates_passed:
            self.registry.set_canary(
                candidate_dir,
                policy={
                    "min_trades": self.canary_min_trades,
                    "min_realized_pnl": 0.0,
                    "max_drawdown": self.canary_max_dd,
                    "min_runtime_minutes": 30,
                },
            )
            risk = getattr(self.brain, "risk_engine", None)
            self._canary_start_trade_count = int(getattr(risk, "daily_trades", 0))
            self._canary_set_time = time.time()
            self._notify(f"Canary enabled: {os.path.basename(candidate_dir)} | {summary}")
        else:
            detail = ", ".join(reasons) if reasons else "wins_or_thresholds_not_met"
            logger.info(f"candidate not promoted: {detail}")
            self._notify(f"Candidate blocked by release gates: {detail} | {summary}")

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
        runtime_minutes = 0.0
        if self._canary_set_time is not None:
            runtime_minutes = max(0.0, (time.time() - self._canary_set_time) / 60.0)

        self.registry.update_canary_metrics(
            trades=trades_since,
            realized_pnl=realized,
            drawdown=dd_pct,
            runtime_minutes=runtime_minutes,
        )

        if realized <= -self.canary_max_loss or dd_pct >= self.canary_max_dd:
            self.registry.rollback_to_champion()
            self._canary_start_trade_count = None
            self._canary_set_time = None
            self._maybe_reload_brain()
            self._notify(f"Canary rollback triggered. realized={realized:.2f}, dd={dd_pct:.3f}")
            return

        if trades_since >= self.canary_min_trades and realized >= 0:
            try:
                self.registry.promote_canary_to_champion()
                self._canary_start_trade_count = None
                self._canary_set_time = None
                self._maybe_reload_brain()
                self._notify(f"Canary promoted. trades={trades_since}, realized={realized:.2f}")
            except Exception as exc:
                logger.warning(f"Canary promotion blocked: {exc}")

    async def nightly_training_loop(self):
        while True:
            now = datetime.datetime.now()
            next_midnight = datetime.datetime(now.year, now.month, now.day, 23, 59, 59)
            seconds_to_midnight = (next_midnight - now).total_seconds()
            await asyncio.sleep(max(60, seconds_to_midnight + 60))
            if self.enable_train:
                await self._train_candidate()
                self._last_train_ts = time.time()


    async def interval_training_loop(self):
        while True:
            await asyncio.sleep(max(60, self.train_every_sec))
            if self.enable_train:
                await self._train_candidate()
                self._last_train_ts = time.time()
                self._last_train_ts = time.time()

    async def start(self):
        logger.warning("AutonomyLoop started")
        self._notify("AutonomyLoop started")

        if self.enable_train and self.train_on_start:
            await self._train_candidate()
            self._last_train_ts = time.time()

        if self.train_every_sec > 0:
            asyncio.create_task(self.interval_training_loop())
        else:
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


