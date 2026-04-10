import time

from Python.autonomy_loop import AutonomyLoop


def test_symbol_due_for_training_is_tracked_per_symbol():
    loop = AutonomyLoop.__new__(AutonomyLoop)
    loop.train_every_sec = 300
    now = time.time()
    loop._last_train_ts_by_symbol = {
        "BTCUSDm": now - 301,
        "XAUUSDm": now - 120,
    }

    assert loop._symbol_due_for_training("BTCUSDm") is True
    assert loop._symbol_due_for_training("XAUUSDm") is False


def test_registry_seed_drops_mismatched_global_champion():
    loop = AutonomyLoop.__new__(AutonomyLoop)
    active = {
        "champion": "global_xau",
        "symbols": {
            "BTCUSDm": {"champion": None, "canary": None, "canary_policy": {}, "canary_state": {}},
            "XAUUSDm": {"champion": None, "canary": None, "canary_policy": {}, "canary_state": {}},
        },
    }

    class _Registry:
        def _read_active(self):
            return active

        def _write_active(self, payload):
            snapshot = {
                "champion": payload.get("champion"),
                "symbols": dict(payload.get("symbols", {})),
            }
            active.clear()
            active.update(snapshot)

        def candidate_targets_symbol(self, candidate_dir, symbol):
            return candidate_dir == "global_xau" and symbol == "XAUUSDm"

    loop.registry = _Registry()
    loop._load_symbols_cfg = lambda: ["BTCUSDm", "XAUUSDm"]

    loop._ensure_symbol_registry_seeded()

    assert active["symbols"]["BTCUSDm"]["champion"] is None
    assert active["symbols"]["XAUUSDm"]["champion"] == "global_xau"
