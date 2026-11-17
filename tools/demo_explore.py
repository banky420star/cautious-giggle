"""For each symbol with no approved model, run the newest candidate as a demo-exploration canary
(demo accounts only; see Python/demo_exploration.py)."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from loguru import logger

from Python.config_utils import DEFAULT_TRADING_SYMBOLS, load_project_config, resolve_trading_symbols
from Python.model_registry import ModelRegistry
from tools.champion_cycle import _latest_candidate, _maybe_demo_explore

cfg = load_project_config(ROOT)
reg = ModelRegistry()
for sym in resolve_trading_symbols(cfg, fallback=DEFAULT_TRADING_SYMBOLS):
    champ = reg.load_active_model(prefer_canary=False, symbol=sym)
    canary = reg.load_active_model(prefer_canary=True, symbol=sym)
    if champ or (canary and canary != champ):
        logger.info(f"{sym}: has an approved model ({os.path.basename(str(champ or canary))}); no exploration")
        continue
    cand = _latest_candidate(reg, symbol=sym)
    if not cand:
        logger.warning(f"{sym}: no trained candidate yet")
        continue
    _maybe_demo_explore(reg, cfg, sym, cand, None, None)
