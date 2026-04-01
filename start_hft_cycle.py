"""
Cautious Giggle — HFT Training Cycle
Runs champion_cycle.py with the HFT config (M1, aggressive reward weights).
Trains models optimized for high-frequency scalping.

Usage:
    python start_hft_cycle.py              # Run one cycle
    python start_hft_cycle.py --loop 30    # Loop every 30 minutes
"""
import os
import sys
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

CONFIG_PATH = os.path.join(BASE_DIR, "config_hft.yaml")
ENV_PATH = os.path.join(BASE_DIR, ".env")


def _load_env_file(path: str) -> None:
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if key and key not in os.environ:
                os.environ[key] = value

def main():
    _load_env_file(ENV_PATH)
    os.environ["AGI_CONFIG"] = CONFIG_PATH
    os.environ["AGI_MODE_TAG"] = "hft"

    from tools import champion_cycle

    loop_minutes = None
    if "--loop" in sys.argv:
        idx = sys.argv.index("--loop")
        if idx + 1 < len(sys.argv):
            loop_minutes = int(sys.argv[idx + 1])
        else:
            loop_minutes = 60

    if loop_minutes:
        if os.path.exists(ENV_PATH):
            print("[HFT-CYCLE] Loaded local .env overrides")
        print(f"[HFT-CYCLE] Loop mode: every {loop_minutes} minutes")
        while True:
            try:
                print(f"[HFT-CYCLE] Starting training cycle...")
                champion_cycle.main()
                print(f"[HFT-CYCLE] Cycle complete. Sleeping {loop_minutes} min...")
            except Exception as e:
                print(f"[HFT-CYCLE] Cycle failed: {e}")
            time.sleep(loop_minutes * 60)
    else:
        if os.path.exists(ENV_PATH):
            print("[HFT-CYCLE] Loaded local .env overrides")
        print(f"[HFT-CYCLE] Running single training cycle...")
        champion_cycle.main()
        print(f"[HFT-CYCLE] Done.")


if __name__ == "__main__":
    main()
