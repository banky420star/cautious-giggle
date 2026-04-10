from pathlib import Path


def test_recent_live_runtime_decisions_are_btc_and_xau_only():
    log_path = Path("logs") / "server.log"
    assert log_path.exists()

    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    recent_decisions = [line for line in lines if "DECISION " in line][-20:]

    assert recent_decisions
    assert any("DECISION BTCUSDm" in line for line in recent_decisions)
    assert any("DECISION XAUUSDm" in line for line in recent_decisions)
    assert all("DECISION EURUSDm" not in line for line in recent_decisions)
    assert all("DECISION GBPUSDm" not in line for line in recent_decisions)
