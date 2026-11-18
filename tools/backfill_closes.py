"""One-off: record closed trades the bot missed (before the deal-id fix) from the broker's deal history.

Writes 'trade_closed' events (marked backfill) to logs/trade_events.jsonl for every close deal not already
recorded, and scores each one in 'Who called it' when its entry ballot exists. Does NOT feed the risk
supervisor (today's loss stop must not be distorted by old trades).
    source scripts/mac/account_env.sh; PYTHONPATH=. .venv/bin/python tools/backfill_closes.py [--days 3]
"""
import argparse, datetime as dt, json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from Python.mt5_compat import mt5  # noqa: E402
from Python import trade_scorecard  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--days", type=int, default=3); a = ap.parse_args()
mt5.initialize()
# resolve the bot's market names first, so deals come back as e.g. XAUUSDm (what the bot looks up), not XAUUSD
for s in ("XAUUSDm", "BTCUSDm", "EURUSDm", "USDJPYm", "XAGUSDm", "ETHUSDm", "USOILm"):
    mt5.instrument(s)
te = os.path.join(ROOT, "logs", "trade_events.jsonl")
seen = set()
for ln in open(te, encoding="utf-8"):
    if '"trade_closed"' in ln:
        seen.add(int(json.loads(ln)["payload"].get("deal_id", 0)))
now = dt.datetime.now(dt.timezone.utc)
deals = mt5.history_deals_get(now - dt.timedelta(days=a.days), now) or []
n = scored = 0
with open(te, "a", encoding="utf-8") as f:
    for d in sorted(deals, key=lambda x: x.time):
        if d.entry != mt5.DEAL_ENTRY_OUT or int(d.deal) in seen:
            continue
        pnl = float(d.profit + d.commission + d.swap)
        p = {"deal_id": int(d.deal), "ticket": int(d.position_id or 0), "symbol": str(d.symbol), "volume": float(d.volume),
             "price": float(d.price), "profit": pnl, "comment": str(d.comment), "backfill": True}
        f.write(json.dumps({"ts": dt.datetime.fromtimestamp(d.time, dt.timezone.utc).isoformat(), "event": "trade_closed",
                            "payload": p}) + "\n")
        n += 1
        scored += bool(trade_scorecard.on_close(p))
print(f"backfilled {n} closed trades ({scored} scored in 'Who called it')")
