"""Remove duplicate trade_closed events (same deal id recorded again after a bot restart) from
logs/trade_events.jsonl, and duplicate scorecard lines (same ticket) from logs/voter_scorecard.jsonl.
Keeps the first copy; writes a .bak-dedupe copy of each file first.
    python3 tools/dedupe_closes.py
"""
import json
import os
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def dedupe(path, key):
    if not os.path.exists(path):
        return 0, 0
    shutil.copy2(path, path + ".bak-dedupe")
    seen, keep, dropped = set(), [], 0
    for ln in open(path, encoding="utf-8"):
        try:
            k = key(json.loads(ln))
        except ValueError:
            k = None
        if k is not None:
            if k in seen:
                dropped += 1
                continue
            seen.add(k)
        keep.append(ln)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.writelines(keep)
    os.replace(tmp, path)
    return len(keep), dropped


def close_key(e):
    if e.get("event") != "trade_closed":
        return None
    d = int((e.get("payload") or {}).get("deal_id") or 0)
    return d or None


def card_key(e):
    t = e.get("ticket")
    return (int(t), e.get("close_price"), e.get("profit")) if t else None   # partial closes of one ticket stay


if __name__ == "__main__":
    print("trade_events.jsonl: kept %d lines, removed %d duplicate closes" % dedupe(os.path.join(ROOT, "logs", "trade_events.jsonl"), close_key))
    print("voter_scorecard.jsonl: kept %d lines, removed %d duplicate scorecards" % dedupe(os.path.join(ROOT, "logs", "voter_scorecard.jsonl"), card_key))
