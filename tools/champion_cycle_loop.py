import argparse
import os
import sys
import time
from datetime import datetime, timezone

from loguru import logger

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools import champion_cycle


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run champion cycle continuously.")
    parser.add_argument(
        "--interval-minutes",
        type=int,
        default=30,
        help="Minutes to wait between cycle starts.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    interval_sec = max(1, int(args.interval_minutes) * 60)

    while True:
        started = datetime.now(timezone.utc).isoformat()
        try:
            logger.info("Champion loop tick start | utc={} | interval_min={}", started, args.interval_minutes)
            champion_cycle.main()
            logger.info("Champion loop tick success")
        except Exception as exc:
            logger.exception("Champion loop tick failed: {}", exc)
        time.sleep(interval_sec)


if __name__ == "__main__":
    main()
