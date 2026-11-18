import os
import sys
import tempfile

# Tests must never write into the bot's real logs (training logs, server log, trade/audit events, scorecards).
# Every module reads AGI_LOG_DIR at import time, so this has to be set before anything from Python/ or training/ loads.
_TEST_LOGS = tempfile.mkdtemp(prefix="agi_test_logs_")
os.environ["AGI_LOG_DIR"] = _TEST_LOGS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
