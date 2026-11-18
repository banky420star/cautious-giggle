"""Closed deals from the Exness API must look like MT5 deals to the bot (id in .deal, close = DEAL_ENTRY_OUT)."""
from Python.exness_api import ExnessMT5


class _Client:
    def deals(self, frm, to, cursor, limit):
        return {"items": [
            {"deal": {"deal_id": "82640277", "order_id": "1", "position_id": "38383604", "type": "open",
                      "direction": "buy", "instrument": "XAUUSD", "volume": "0.01", "price": "3765.5",
                      "profit": {"amount": "0"}, "commission": "-0.11", "create_time": "2026-11-18T12:29:30+00:00"}},
            {"deal": {"deal_id": "82640999", "order_id": "2", "position_id": "38383604", "type": "close",
                      "direction": "sell", "instrument": "XAUUSD", "volume": "0.01", "price": "3767.0",
                      "profit": {"amount": "1.5", "currency": "USD"}, "create_time": "2026-11-18T12:31:00+00:00"}},
        ], "has_more": False}


def test_close_deal_has_id_and_exit_fields():
    m = ExnessMT5.__new__(ExnessMT5)
    m.client, m.symbol_map = _Client(), {}
    m._call = lambda fn, *a, **k: fn(*a, **k)
    deals = m.history_deals_get("2026-11-18T00:00:00Z", "2026-11-18T23:00:00Z")
    out = [d for d in deals if d.entry == m.DEAL_ENTRY_OUT]
    assert len(out) == 1
    d = out[0]
    assert int(getattr(d, "deal", 0)) == 82640999          # what Server_AGI uses to record the close
    assert int(d.position_id) == 38383604 and d.price == 3767.0 and d.profit == 1.5
