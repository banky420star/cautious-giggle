"""Exness Public Trader API backend: signing, MT5-shim semantics, executor integration."""
import base64
import json
from datetime import datetime, timedelta, timezone

import pytest

pytest.importorskip("cryptography")

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from Python import exness_api as ex


def _b64url_decode(s):
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def test_signed_headers_verify_and_bind_request():
    key = Ed25519PrivateKey.generate()
    h = ex.build_signed_headers("KEY1", key, "post", "/v1/x?a=1", '{"b":2}', "idem-1", 1_700_000_000_000)
    payload = _b64url_decode(h["EXN-DATA"])
    key.public_key().verify(_b64url_decode(h["EXN-SIGN"]), payload)  # raises if invalid
    data = json.loads(payload)
    assert data["method"] == "POST" and data["path"] == "/v1/x?a=1"
    assert data["idempotency_key"] == "idem-1" and data["sign_version"] == 1
    assert list(data) == ["api_key", "idempotency_key", "timestamp", "sign_version", "method", "path", "body_hash"]
    assert h["EXN-TIMESTAMP"] == "1700000000000"


def test_private_key_formats():
    key = Ed25519PrivateKey.generate()
    from cryptography.hazmat.primitives import serialization

    seed = key.private_bytes(serialization.Encoding.Raw, serialization.PrivateFormat.Raw, serialization.NoEncryption())
    pub = key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    for material in (seed.hex(), base64.b64encode(seed).decode(), base64.urlsafe_b64encode(seed).decode().rstrip("="),
                     base64.b64encode(seed + pub).decode()):
        k = ex.load_ed25519_private_key(material)
        assert k.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw) == pub


# ----------------------------------------------------------------------------- fake API
class FakeClient:
    account_id = "777"
    trading_url = "https://fake"

    def __init__(self):
        self.now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        self.positions = []
        self.orders = []
        self.calls = []
        self.fail_open_code = None

    def resolve_access_point(self):
        return self.trading_url

    def instruments(self):
        return ["BTCUSD", "XAUUSD"]

    def instrument_conditions(self, inst):
        return {"instrument": inst, "point_digits": 2, "contract_size": "1" if inst == "BTCUSD" else "100",
                "volume_min": "0.01", "volume_max": "10", "volume_step": "0.01", "filling_modes": ["ioc"],
                "trade_mode": "enabled"}

    def snapshot(self):
        return {"account_state": {"balance": "1000", "equity": "1010", "used_margin": "50"},
                "positions": list(self.positions), "orders": list(self.orders)}

    def candles(self, inst, tf, frm, to=None, count=None, price_type=None):
        self.calls.append(("candles", inst, tf, frm, count, price_type))
        step = 60 if tf == "M1" else 300
        start = datetime.fromisoformat(frm.replace("Z", "+00:00"))
        # last bar at/ before `from` is the forming one when from==now
        n = abs(int(count or 1))
        rows = []
        for i in range(n):
            t = start - timedelta(seconds=step * i) if (count or 1) < 0 else start + timedelta(seconds=step * i)
            t = t - timedelta(seconds=t.timestamp() % step)
            px = 100.0 + (t.timestamp() / step) % 7
            if price_type == "both":
                rows.append({"time": t.isoformat(), "bid": {"open": px, "high": px, "low": px, "close": px},
                             "ask": {"open": px + 0.2, "high": px + 0.2, "low": px + 0.2, "close": px + 0.2}, "volume": 3})
            else:
                rows.append({"time": t.isoformat(), "bid": {"open": px, "high": px + 1, "low": px - 1, "close": px}, "volume": 5})
        return sorted(rows, key=lambda r: r["time"])

    def open_position(self, body, key):
        self.calls.append(("open", body))
        if self.fail_open_code:
            raise ex.ExnessApiError(self.fail_open_code, "nope", 400)
        self.positions.append({"position_id": str(100 + len(self.positions)), "direction": body["side"],
                               "instrument": body["instrument"], "volume": body["volume"], "state": "open",
                               "open_price": "100.0", "create_time": self.now.isoformat()})
        return {"status": "accepted", "operation_id": "55"}

    def close_position(self, pid, params, key):
        self.calls.append(("close", pid, params))
        for p in self.positions:
            if p["position_id"] == pid:
                left = round(float(p["volume"]) - float(params["volume"]), 2)
                p["volume"] = f"{left:.2f}"
                if left <= 0:
                    p["state"] = "closed"
        return {"status": "accepted", "operation_id": "56"}

    def place_order(self, body, key):
        self.calls.append(("place_order", body))
        self.orders.append({"order_id": "900", "order_type": "sltp", "position_id": body["position_id"],
                            "stop_loss_price": body.get("stop_loss_price"), "take_profit_price": body.get("take_profit_price")})
        return {"status": "accepted", "operation_id": "57"}

    def modify_order(self, oid, body, key):
        self.calls.append(("modify_order", oid, body))
        return {"status": "accepted", "operation_id": "58"}

    def operation_status(self, op_id):
        return {"operation_id": op_id, "status": "confirmed"}

    def deals(self, frm=None, to=None, cursor=None, limit=None):
        return {"items": [{"event_time": "x", "deal": {"deal_id": "1", "type": "close", "instrument": "BTCUSD",
                                                         "direction": "sell", "volume": "0.1", "price": "101",
                                                         "profit": {"amount": "12.5", "currency": "USD"},
                                                         "commission": "0.5", "create_time": self.now.isoformat()}}],
                "has_more": False}


@pytest.fixture
def shim():
    s = ex.ExnessMT5(client=FakeClient())
    assert s.initialize()
    return s


def test_symbol_mapping_strips_exness_m_suffix(shim):
    assert shim.instrument("BTCUSDm") == "BTCUSD"
    assert shim.instrument("XAUUSDm") == "XAUUSD"


def test_symbol_info_from_conditions(shim):
    info = shim.symbol_info("XAUUSDm")
    assert info.digits == 2 and info.point == pytest.approx(0.01)
    assert info.volume_step == pytest.approx(0.01) and info.filling_mode == 2
    assert info.trade_tick_value == pytest.approx(100 * 0.01)


def test_tick_from_bid_ask_candle(shim):
    t = shim.symbol_info_tick("BTCUSDm")
    assert t.ask == pytest.approx(t.bid + 0.2)


def test_copy_rates_from_pos_drops_forming_bar_and_pages(shim):
    rates = shim.copy_rates_from_pos("BTCUSDm", shim.TIMEFRAME_M5, 1, 1500)
    assert len(rates) == 1500
    times = rates["time"]
    assert (times[1:] - times[:-1] == 300).all()
    all_rates = shim.copy_rates_from_pos("BTCUSDm", shim.TIMEFRAME_M5, 0, 1501)
    assert rates[-1]["time"] == all_rates[-2]["time"]  # start_pos=1 excludes the newest bar


def test_account_info(shim):
    a = shim.account_info()
    assert (a.balance, a.equity, a.margin_free) == (1000.0, 1010.0, 960.0)


def test_history_deals_mapped(shim):
    now = datetime.now(timezone.utc)
    d = shim.history_deals_get(now - timedelta(days=1), now)[0]
    assert d.entry == shim.DEAL_ENTRY_OUT and d.profit == 12.5 and d.symbol == "BTCUSD"


def test_order_send_error_maps_to_mt5_retcode(shim):
    shim.client.fail_open_code = 4003  # MARKET_SESSION_CLOSED
    r = shim.order_send({"action": shim.TRADE_ACTION_DEAL, "symbol": "XAUUSDm", "volume": 0.1, "type": 0})
    assert r.retcode == 10018


def test_executor_round_trip_on_exness_backend(shim, monkeypatch):
    import Python.mt5_executor as mx

    monkeypatch.setattr(mx, "mt5", shim)

    class Risk:
        max_lots = 1.0
        risk_percent = 0.0
        trades = []

        def can_open(self, s=None):
            return True

        def record_trade(self, s=None):
            self.trades.append(s)

        def record_error(self):
            raise AssertionError("no errors expected")

        def record_success(self):
            pass

        def get_symbol_profile(self, s):
            return {"entry_deviation": 20, "sl_points": 250, "tp_points": 450}

    ex_ = mx.MT5Executor(Risk())
    m = ex_.reconcile_exposure("BTCUSDm", 0.3, 1.0)
    assert m["executed"] is True
    open_call = [c for c in shim.client.calls if c[0] == "open"][0][1]
    assert open_call["instrument"] == "BTCUSD" and open_call["side"] == "buy" and open_call["volume"] == "0.30"
    assert "stop_loss_price" in open_call and "take_profit_price" in open_call

    # Shrink -> partial close of 0.10 through DELETE ?volume=
    ex_.reconcile_exposure("BTCUSDm", 0.2, 1.0)
    close_call = [c for c in shim.client.calls if c[0] == "close"][-1]
    assert close_call[2] == {"volume": "0.10"}

    # Flatten
    ex_.reconcile_exposure("BTCUSDm", 0.0, 1.0)
    assert shim.positions_get("BTCUSDm") == ()


def test_sltp_modify_creates_then_patches(shim):
    shim.order_send({"action": shim.TRADE_ACTION_DEAL, "symbol": "BTCUSDm", "volume": 0.1, "type": 0})
    pid = shim.positions_get("BTCUSDm")[0].ticket
    shim.order_send({"action": shim.TRADE_ACTION_SLTP, "symbol": "BTCUSDm", "position": pid, "sl": 95.0, "tp": 110.0})
    assert [c[0] for c in shim.client.calls].count("place_order") == 1
    p = shim.positions_get("BTCUSDm")[0]
    assert p.sl == 95.0 and p.tp == 110.0
    shim.order_send({"action": shim.TRADE_ACTION_SLTP, "symbol": "BTCUSDm", "position": pid, "sl": 99.0, "tp": 110.0})
    assert [c[0] for c in shim.client.calls].count("modify_order") == 1
