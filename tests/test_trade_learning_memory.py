import json
import shutil
import uuid
from pathlib import Path

from Python.trade_learning import load_trade_memory


def _tmp_learning_root() -> Path:
    root = Path(".tmp") / f"learning_test_{uuid.uuid4().hex}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def test_load_trade_memory_symbol_and_aggregate():
    root = _tmp_learning_root()
    try:
        payload = {
            "by_symbol": [
                {
                    "symbol": "EURUSDm",
                    "trades": 10,
                    "wins": 6,
                    "losses": 4,
                    "win_rate": 60.0,
                    "expectancy": 0.35,
                    "profit_factor": 1.4,
                    "avg_loss": -0.7,
                    "max_loss_streak": 2,
                    "recent_loss_streak": 1,
                },
                {
                    "symbol": "XAUUSDm",
                    "trades": 20,
                    "wins": 8,
                    "losses": 12,
                    "win_rate": 40.0,
                    "expectancy": -0.2,
                    "profit_factor": 0.8,
                    "avg_loss": -1.1,
                    "max_loss_streak": 5,
                    "recent_loss_streak": 3,
                },
            ]
        }
        with open(root / "trade_learning_latest.json", "w", encoding="utf-8") as f:
            json.dump(payload, f)

        eur = load_trade_memory(str(root), symbol="EURUSDm")
        assert eur["trades"] == 10
        assert eur["wins"] == 6
        assert eur["losses"] == 4
        assert eur["max_loss_streak"] == 2
        assert eur["recent_loss_streak"] == 1

        agg = load_trade_memory(str(root))
        assert agg["trades"] == 30
        assert agg["wins"] == 14
        assert agg["losses"] == 16
        assert agg["max_loss_streak"] == 5
        assert agg["recent_loss_streak"] == 3
    finally:
        shutil.rmtree(root, ignore_errors=True)
