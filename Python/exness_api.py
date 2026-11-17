"""Exness Public Trader API backend (runs natively on macOS / Linux).

Implements the subset of the ``MetaTrader5`` Python module that this project uses
(initialize, account_info, symbol_info, symbol_info_tick, copy_rates_*, positions_get,
order_send, history_deals_get, ...) on top of the Exness REST API, so the existing
executor, risk and data code runs unchanged:

    AGI_BROKER_BACKEND=exness_api python Python/Server_AGI.py --live

Credentials (env vars, or the ``exness_api:`` section of config.yaml):
    EXNESS_API_KEY        key id (exnsk-...) from Personal Area -> Key Management
    EXNESS_PRIVATE_KEY    Ed25519 private key: the 128-hex-char key Exness generates, or a
                          PKCS#8 PEM if you registered your own public key;
                          or EXNESS_PRIVATE_KEY_FILE=/path/to/key
    EXNESS_ACCOUNT_ID     trading account id the key is allowed to trade
    EXNESS_BASE_URL       Gateway host shown in Key Management's DNS widget (required)

Only Exness API-enabled trading accounts work (not standard MetaTrader accounts), and
the API has no magic numbers: use a dedicated account for the bot.

Signing and endpoints follow the official Exness API Developer Portal
(https://www.exness-api.com). Run ``python -m Python.exness_api check``
against your account before trading: it exercises every read endpoint the bot uses.
"""
from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import re
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from types import SimpleNamespace
from urllib.parse import quote

import numpy as np
import requests

# The Gateway host is shown in the DNS widget of Key Management (Personal Area). Exness says
# not to guess hosts, so EXNESS_BASE_URL is required.
DEFAULT_BASE_URL = ""
_TIMESTAMP_SAFETY_MS = 1000  # server accepts 0 <= server_now - ts <= 3000 ms

# --------------------------------------------------------------------------- errors
# Exness API error code -> closest MT5 trade retcode (so MT5Executor logic applies).
_ERROR_TO_RETCODE = {
    1: 10024,     # SYSTEM_RATE_LIMIT -> TOO_MANY_REQUESTS
    1003: 10024,  # AUTH_RATE_LIMIT
    3015: 10024,  # REQUEST_RATE_LIMIT
    2: 10031,     # SYSTEM_TEMPORARY_UNAVAILABLE -> CONNECTION
    3: 10011,     # SYSTEM_INTERNAL_ERROR -> ERROR
    2002: 10044,  # ACCOUNT_CLOSE_ONLY -> CLOSE_ONLY
    3002: 10015,  # REQUEST_INVALID_PRICE -> INVALID_PRICE
    3013: 10036,  # REQUEST_POSITION_NOT_FOUND -> POSITION_CLOSED
    4001: 10017,  # MARKET_TRADE_DISABLED -> TRADE_DISABLED
    4002: 10044,  # MARKET_CLOSE_ONLY
    4003: 10018,  # MARKET_SESSION_CLOSED -> MARKET_CLOSED
    4004: 10021,  # MARKET_NO_QUOTES -> PRICE_OFF
    5000: 10014,  # TRADING_RULE_INVALID_VOLUME -> INVALID_VOLUME
    5001: 10014,
    5002: 10014,
    5003: 10014,
    5004: 10014,
    5005: 10016,  # TRADING_RULE_INVALID_PRICE_LEVELS -> INVALID_STOPS
    5006: 10019,  # INSUFFICIENT_MARGIN -> NO_MONEY
    5007: 10019,
    5008: 10017,
    5010: 10040,  # TOO_MANY_OPEN_POSITIONS -> LIMIT_POSITIONS
    6000: 10004,  # EXECUTION_REQUOTE -> REQUOTE
    6001: 10006,  # EXECUTION_REJECTED -> REJECT
}


class ExnessApiError(RuntimeError):
    def __init__(self, code: int, message: str, http_status: int):
        super().__init__(f"[{code}] {message} (HTTP {http_status})")
        self.code = int(code)
        self.message = str(message)
        self.http_status = int(http_status)


# --------------------------------------------------------------------------- signing
def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _b64_any_decode(text: str) -> bytes:
    t = text.strip()
    pad = "=" * (-len(t) % 4)
    try:
        return base64.urlsafe_b64decode(t + pad)
    except Exception:
        return base64.b64decode(t + pad)


def load_ed25519_private_key(material: str):
    """Accept a PEM private key, or a 32-byte seed encoded as hex/base64/base64url."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    material = (material or "").strip()
    if not material:
        raise ValueError("empty Ed25519 private key")
    if "BEGIN" in material:
        key = serialization.load_pem_private_key(material.encode("utf-8"), password=None)
        if not isinstance(key, Ed25519PrivateKey):
            raise ValueError("PEM key is not Ed25519")
        return key
    raw = None
    if len(material) in (64, 128):  # Exness-generated keys are 128 hex chars (seed || public key)
        try:
            raw = bytes.fromhex(material)
        except ValueError:
            raw = None
    if raw is None:
        raw = _b64_any_decode(material)
    if len(raw) == 64:  # seed||public (NaCl format)
        raw = raw[:32]
    if len(raw) != 32:
        raise ValueError(f"Ed25519 seed must be 32 bytes, got {len(raw)}")
    return Ed25519PrivateKey.from_private_bytes(raw)


def build_signed_headers(api_key: str, private_key, method: str, path_with_query: str, body: str,
                         idempotency_key: str, timestamp_ms: int) -> dict:
    body_hash = _b64url(hashlib.sha256((body or "").encode("utf-8")).digest())
    payload = {
        "api_key": api_key,
        "idempotency_key": idempotency_key,
        "timestamp": int(timestamp_ms),
        "sign_version": 1,
        "method": method.upper(),
        "path": path_with_query,
        "body_hash": body_hash,
    }
    # Compact JSON with the key order above (matches JSON.stringify in the reference SDK).
    payload_bytes = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    signature = private_key.sign(payload_bytes)
    return {
        "EXN-API-KEY": api_key,
        "EXN-TIMESTAMP": str(int(timestamp_ms)),
        "EXN-SIGN-VERSION": "1",
        "EXN-IDEMPOTENCY-KEY": idempotency_key,
        "EXN-DATA": _b64url(payload_bytes),
        "EXN-SIGN": _b64url(signature),
    }


# --------------------------------------------------------------------------- client
class ExnessClient:
    def __init__(self, api_key: str, private_key, account_id: str, base_url: str = DEFAULT_BASE_URL,
                 timeout: float = 10.0, session: requests.Session | None = None):
        self.api_key = str(api_key)
        self.private_key = private_key
        self.account_id = str(account_id)
        self.base_url = base_url.rstrip("/")
        self.trading_url = self.base_url
        self.timeout = float(timeout)
        self.session = session or requests.Session()
        self.clock_offset_ms = 0
        self._lock = threading.Lock()
        # Rate limiting: global per-account limit plus per-endpoint limits (from GET .../limits).
        self._global_limit = (15, 1.0)
        self._method_limits: list[tuple[str, "re.Pattern", int, float]] = []
        self._history: dict[str, list[float]] = {}

    def load_limits(self) -> None:
        """Pace requests by the limits Exness publishes for this account (never hardcode them)."""
        data = (self.trading("GET", f"/v1/configuration/accounts/{self.account_id}/limits") or {}).get("limits") or {}
        rest = data.get("rest") or {}
        g = rest.get("global_account_rate") or {}
        if g.get("limit"):
            self._global_limit = (int(g["limit"]), float(g.get("window_seconds") or 1))
        out = []
        for m in rest.get("methods") or []:
            rl = m.get("rate_limit") or {}
            if not (rl.get("limit") and m.get("path")):
                continue
            pattern = "^" + re.sub(r"\\{[^}]+\\}", "[^/]+", re.escape(str(m["path"]))) + "$"
            out.append((str(m.get("http_method", "GET")).upper(), re.compile(pattern), int(rl["limit"]),
                        float(rl.get("window_seconds") or 1)))
        self._method_limits = out

    def _pace(self, method: str, path: str) -> None:
        buckets = [("*", self._global_limit[0], self._global_limit[1])]
        for m, rx, lim, win in self._method_limits:
            if m == method.upper() and rx.match(path):
                buckets.append((f"{m} {rx.pattern}", lim, win))
        while True:
            now = time.monotonic()
            wait = 0.0
            for key, lim, win in buckets:
                hist = [t for t in self._history.get(key, []) if now - t < win]
                self._history[key] = hist
                if len(hist) >= lim:
                    wait = max(wait, win - (now - hist[0]) + 0.02)
            if wait <= 0:
                for key, _, _ in buckets:
                    self._history[key].append(now)
                return
            time.sleep(wait)

    def _qs(self, query: dict | None) -> str:
        if not query:
            return ""
        parts = [f"{quote(str(k), safe='')}={quote(str(v), safe='')}" for k, v in query.items() if v is not None]
        return ("?" + "&".join(parts)) if parts else ""

    def _update_clock(self, resp) -> None:
        date = resp.headers.get("Date") or resp.headers.get("date")
        if not date:
            return
        try:
            server_ms = parsedate_to_datetime(date).timestamp() * 1000.0
            self.clock_offset_ms = int(server_ms - time.time() * 1000.0)
        except Exception:
            pass

    def request(self, method: str, path: str, query: dict | None = None, body=None,
                idempotency_key: str = "", base: str | None = None):
        body_str = json.dumps(body, separators=(",", ":")) if body is not None else ""
        path_q = path + self._qs(query)
        url = (base or self.base_url) + path_q
        last_exc = None
        for attempt in range(3):
            with self._lock:
                self._pace(method, path)
            ts = int(time.time() * 1000) + self.clock_offset_ms - _TIMESTAMP_SAFETY_MS
            headers = {"Content-Type": "application/json"}
            headers.update(build_signed_headers(self.api_key, self.private_key, method, path_q, body_str,
                                                idempotency_key, ts))
            resp = self.session.request(method, url, data=body_str or None, headers=headers, timeout=self.timeout)
            self._update_clock(resp)
            if resp.ok:
                return resp.json() if resp.text else None
            try:
                err = resp.json()
                if isinstance(err.get("error"), dict):
                    err = err["error"]
                code, msg = int(err.get("code", resp.status_code)), str(err.get("error_message", resp.reason))
            except Exception:
                code, msg = resp.status_code, resp.reason
            last_exc = ExnessApiError(code, msg, resp.status_code)
            # One retry after re-syncing the clock on auth/timestamp failures.
            if attempt == 0 and (code in (1000, 1001) or "timestamp" in msg.lower()):
                continue
            # Rate limited despite pacing (shared limits / other clients): back off once.
            if attempt < 2 and (code in (1, 1003, 3015) or resp.status_code == 429):
                time.sleep(1.0 + attempt)
                continue
            raise last_exc
        raise last_exc

    # Trading requests go to the account's trading access point.
    def resolve_access_point(self) -> str:
        data = self.request("GET", "/v1/trading/access-point", query={"account_id": self.account_id}) or {}
        ap = str(data.get("access_point") or "").rstrip("/")
        if ap:
            self.trading_url = ap
        return self.trading_url

    def trading(self, method, path, **kw):
        return self.request(method, path, base=self.trading_url, **kw)

    # Endpoints
    def account_details(self):
        return self.trading("GET", f"/v1/configuration/accounts/{self.account_id}/account")

    def instruments(self):
        return (self.trading("GET", f"/v1/configuration/accounts/{self.account_id}/instruments") or {}).get("instruments") or []

    def instrument_conditions(self, instrument: str):
        return self.trading("GET", f"/v1/configuration/accounts/{self.account_id}/instruments/{quote(instrument, safe='')}/conditions")

    def limits(self):
        return self.trading("GET", f"/v1/configuration/accounts/{self.account_id}/limits")

    def snapshot(self):
        return self.trading("GET", f"/v1/trading/accounts/{self.account_id}/snapshot") or {}

    def candles(self, instrument: str, timeframe: str, frm: str, to: str | None = None, count: int | None = None,
                price_type: str | None = None):
        q = {"instrument": instrument, "timeframe": timeframe, "from": frm, "to": to, "count": count,
             "price_type": price_type}
        return (self.trading("GET", f"/v1/market-data/accounts/{self.account_id}/candles", query=q) or {}).get("candles") or []

    def open_position(self, req: dict, key: str):
        return self.trading("POST", f"/v1/trading/accounts/{self.account_id}/positions", body=req, idempotency_key=key)

    def close_position(self, position_id: str, params: dict, key: str):
        return self.trading("DELETE", f"/v1/trading/accounts/{self.account_id}/positions/{position_id}",
                            query=params, idempotency_key=key)

    def place_order(self, req: dict, key: str):
        return self.trading("POST", f"/v1/trading/accounts/{self.account_id}/orders", body=req, idempotency_key=key)

    def modify_order(self, order_id: str, req: dict, key: str):
        return self.trading("PATCH", f"/v1/trading/accounts/{self.account_id}/orders/{order_id}", body=req,
                            idempotency_key=key)

    def operation_status(self, operation_id: str):
        return self.trading("GET", f"/v1/trading/accounts/{self.account_id}/operations/{operation_id}")

    def deals(self, frm: str | None = None, to: str | None = None, cursor: str | None = None, limit: int | None = None):
        limit = max(1, min(200, int(limit or 200)))
        q = {"cursor": cursor, "limit": limit} if cursor else {"from": frm, "to": to, "limit": limit}
        return self.trading("GET", f"/v1/history/accounts/{self.account_id}/deals", query=q) or {}


# --------------------------------------------------------------------------- helpers
def _iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _to_dt(value) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    s = str(value).replace("Z", "+00:00")
    dt = datetime.fromisoformat(s)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _f(value, default=0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _decimals(step: float) -> int:
    if step <= 0:
        return 2
    return max(0, int(round(-math.log10(step)))) if step < 1 else 0


RATES_DTYPE = np.dtype([("time", "<i8"), ("open", "<f8"), ("high", "<f8"), ("low", "<f8"), ("close", "<f8"),
                        ("tick_volume", "<i8"), ("spread", "<i4"), ("real_volume", "<i8")])


# --------------------------------------------------------------------------- MT5 shim
class ExnessMT5:
    """Drop-in for the parts of the MetaTrader5 module used by this project."""

    ORDER_TYPE_BUY = 0
    ORDER_TYPE_SELL = 1
    ORDER_FILLING_FOK = 0
    ORDER_FILLING_IOC = 1
    ORDER_FILLING_RETURN = 2
    ORDER_TIME_GTC = 0
    TRADE_ACTION_DEAL = 1
    TRADE_ACTION_SLTP = 6
    TRADE_RETCODE_DONE = 10009
    DEAL_ENTRY_IN = 0
    DEAL_ENTRY_OUT = 1
    POSITION_TYPE_BUY = 0
    POSITION_TYPE_SELL = 1
    TIMEFRAME_M1 = 1
    TIMEFRAME_M5 = 5
    TIMEFRAME_M15 = 15
    TIMEFRAME_M30 = 30
    TIMEFRAME_H1 = 16385
    TIMEFRAME_H4 = 16388
    TIMEFRAME_D1 = 16408
    _TF = {1: ("M1", 60), 5: ("M5", 300), 15: ("M15", 900), 30: ("M30", 1800), 16385: ("H1", 3600),
           16388: ("H4", 14400), 16408: ("D1", 86400)}

    class Tick(SimpleNamespace):
        pass

    def __init__(self, client: ExnessClient | None = None, symbol_map: dict | None = None):
        self.client = client
        self.symbol_map = dict(symbol_map or {})
        self._last_error = (0, "ok")
        self._connected = False
        self._cond_cache: dict[str, tuple[float, dict]] = {}
        self._tick_cache: dict[str, tuple[float, SimpleNamespace]] = {}
        self._snap_cache: tuple[float, dict] | None = None
        self._instruments: list[str] = []

    # ---------------------------------------------------------------- lifecycle
    def initialize(self, *args, **kwargs) -> bool:
        try:
            if self.client is None:
                self.client = client_from_env()
            try:
                self.client.resolve_access_point()
            except ExnessApiError as exc:
                if exc.http_status not in (404, 405):
                    raise
            try:
                self.client.load_limits()
            except Exception:
                pass  # fall back to the default global pace
            self._instruments = list(self.client.instruments())
            self._connected = True
            self._last_error = (0, "ok")
            return True
        except Exception as exc:
            self._connected = False
            self._last_error = (-1, f"exness_api init failed: {exc}")
            return False

    def shutdown(self):
        self._connected = False
        return True

    def last_error(self):
        return self._last_error

    def terminal_info(self):
        return SimpleNamespace(connected=True, name="Exness Public Trader API") if self._connected else None

    def _call(self, fn, *a, **kw):
        try:
            out = fn(*a, **kw)
            self._last_error = (0, "ok")
            return out
        except ExnessApiError as exc:
            self._last_error = (exc.code, exc.message)
            if exc.code in (2, 3) or exc.http_status >= 500:
                self._connected = False
            raise
        except requests.RequestException as exc:
            self._last_error = (-2, str(exc))
            self._connected = False
            raise

    # ---------------------------------------------------------------- symbols
    def instrument(self, symbol: str) -> str:
        s = str(symbol)
        if s in self.symbol_map:
            return self.symbol_map[s]
        if not self._instruments or s in self._instruments:
            return s
        for cand in (s[:-1] if s.endswith("m") else None, s + "m", s.upper()):
            if cand and cand in self._instruments:
                self.symbol_map[s] = cand
                return cand
        return s

    def symbol_select(self, symbol, enable=True) -> bool:
        return self.symbol_info(symbol) is not None

    def _conditions(self, symbol: str) -> dict | None:
        inst = self.instrument(symbol)
        cached = self._cond_cache.get(inst)
        if cached and time.time() - cached[0] < 600:
            return cached[1]
        try:
            cond = self._call(self.client.instrument_conditions, inst) or {}
        except Exception:
            return cached[1] if cached else None
        self._cond_cache[inst] = (time.time(), cond)
        return cond

    def symbol_info(self, symbol):
        cond = self._conditions(symbol)
        if not cond:
            return None
        digits = int(cond.get("point_digits") or 2)
        point = 10.0 ** (-digits)
        contract = _f(cond.get("contract_size"), 1.0) or 1.0
        fm = 0
        for mode in cond.get("filling_modes") or []:
            fm |= {"fok": 1, "ioc": 2}.get(str(mode).lower(), 0)
        return SimpleNamespace(
            name=str(symbol), digits=digits, point=point,
            volume_min=_f(cond.get("volume_min"), 0.01), volume_max=_f(cond.get("volume_max"), 100.0),
            volume_step=_f(cond.get("volume_step"), 0.01),
            trade_contract_size=contract,
            trade_tick_size=point,
            # Exact for USD-quoted instruments on USD accounts (XAUUSD, BTCUSD).
            trade_tick_value=contract * point, trade_tick_value_loss=contract * point,
            trade_stops_level=0, trade_freeze_level=0, filling_mode=fm,
            trade_mode=0 if str(cond.get("trade_mode", "enabled")) != "enabled" else 4,
            currency_profit=str(cond.get("quote_currency") or "USD"),
        )

    # ---------------------------------------------------------------- prices
    def symbol_info_tick(self, symbol):
        inst = self.instrument(symbol)
        cached = self._tick_cache.get(inst)
        if cached and time.time() - cached[0] < 1.0:
            return cached[1]
        try:
            now = datetime.now(timezone.utc)
            rows = self._call(self.client.candles, inst, "M1", _iso(now), None, -1, "both")
        except Exception:
            return None
        if not rows:
            return None
        last = rows[-1]
        bid = _f((last.get("bid") or {}).get("close"))
        ask = _f((last.get("ask") or {}).get("close"))
        if bid <= 0 or ask <= 0:
            return None
        tick = self.Tick(bid=bid, ask=ask, last=(bid + ask) / 2.0, time=int(_to_dt(last.get("time")).timestamp()) + 59,
                         volume=int(_f(last.get("volume"))))
        self._tick_cache[inst] = (time.time(), tick)
        return tick

    def _candle_rows(self, inst: str, tf_name: str, frm: datetime, count: int) -> list[dict]:
        return self._call(self.client.candles, inst, tf_name, _iso(frm), None, int(count), "bid") or []

    @staticmethod
    def _row_ohlc(row: dict):
        side = row.get("bid") if isinstance(row.get("bid"), dict) else row
        return _f(side.get("open")), _f(side.get("high")), _f(side.get("low")), _f(side.get("close"))

    def _rows_to_rates(self, rows: list[dict]) -> np.ndarray:
        out = np.zeros(len(rows), dtype=RATES_DTYPE)
        for i, r in enumerate(rows):
            o, h, l, c = self._row_ohlc(r)
            out[i] = (int(_to_dt(r.get("time")).timestamp()), o, h, l, c, int(_f(r.get("volume"))), 0, 0)
        return out

    def copy_rates_from_pos(self, symbol, timeframe, start_pos, count):
        """MT5 semantics: position 0 is the bar currently forming."""
        tf_name, tf_sec = self._TF.get(int(timeframe), ("M5", 300))
        inst = self.instrument(symbol)
        need = int(start_pos) + int(count)
        rows: dict[int, dict] = {}
        cursor = datetime.now(timezone.utc)
        try:
            while len(rows) < need:
                batch = self._candle_rows(inst, tf_name, cursor, -min(1000, need - len(rows) + 1))
                if not batch:
                    break
                for r in batch:
                    rows[int(_to_dt(r.get("time")).timestamp())] = r
                earliest = min(_to_dt(r.get("time")) for r in batch)
                new_cursor = earliest - timedelta(seconds=1)
                if new_cursor >= cursor:
                    break
                cursor = new_cursor
                if len(batch) < 2:
                    break
        except Exception:
            return None
        if not rows:
            return None
        ordered = [rows[k] for k in sorted(rows)]
        rates = self._rows_to_rates(ordered)
        end = len(rates) - int(start_pos)
        if end <= 0:
            return None
        return rates[max(0, end - int(count)):end]

    def copy_rates_from(self, symbol, timeframe, date_from, count):
        tf_name, _ = self._TF.get(int(timeframe), ("M5", 300))
        try:
            rows = self._candle_rows(self.instrument(symbol), tf_name, _to_dt(date_from), -min(1000, int(count)))
        except Exception:
            return None
        return self._rows_to_rates(sorted(rows, key=lambda r: _to_dt(r.get("time")))) if rows else None

    def copy_rates_range(self, symbol, timeframe, date_from, date_to):
        tf_name, tf_sec = self._TF.get(int(timeframe), ("M5", 300))
        inst = self.instrument(symbol)
        start, end = _to_dt(date_from), _to_dt(date_to)
        rows: dict[int, dict] = {}
        cursor = start
        try:
            while cursor < end:
                batch = self._candle_rows(inst, tf_name, cursor, 1000)
                batch = [r for r in batch if _to_dt(r.get("time")) <= end]
                if not batch:
                    break
                for r in batch:
                    rows[int(_to_dt(r.get("time")).timestamp())] = r
                latest = max(_to_dt(r.get("time")) for r in batch)
                if latest + timedelta(seconds=tf_sec) <= cursor:
                    break
                cursor = latest + timedelta(seconds=tf_sec)
        except Exception:
            return None
        return self._rows_to_rates([rows[k] for k in sorted(rows)]) if rows else None

    # ---------------------------------------------------------------- account
    def _snapshot(self, max_age: float = 1.0) -> dict:
        if self._snap_cache and time.time() - self._snap_cache[0] < max_age:
            return self._snap_cache[1]
        snap = self._call(self.client.snapshot) or {}
        self._snap_cache = (time.time(), snap)
        return snap

    def account_info(self):
        try:
            st = self._snapshot().get("account_state") or {}
        except Exception:
            return None
        bal, eq, used = _f(st.get("balance")), _f(st.get("equity")), _f(st.get("used_margin"))
        return SimpleNamespace(login=int(self.client.account_id) if str(self.client.account_id).isdigit() else 0,
                               balance=bal, equity=eq, margin=used, margin_free=eq - used,
                               profit=eq - bal, currency="USD")

    def positions_get(self, symbol=None, **kwargs):
        try:
            snap = self._snapshot()
        except Exception:
            return None
        sltp = {}
        for o in snap.get("orders") or []:
            if str(o.get("order_type")) == "sltp" and o.get("position_id"):
                sltp[str(o["position_id"])] = o
        reverse = {v: k for k, v in self.symbol_map.items()}
        out = []
        for p in snap.get("positions") or []:
            if str(p.get("state", "open")) != "open":
                continue
            inst = str(p.get("instrument"))
            sym = reverse.get(inst, inst)
            if symbol is not None and str(symbol) not in (sym, inst):
                continue
            is_buy = str(p.get("direction")) == "buy"
            vol, open_px = _f(p.get("volume")), _f(p.get("open_price"))
            o = sltp.get(str(p.get("position_id"))) or {}
            profit = 0.0
            tick = self.symbol_info_tick(sym)
            info = self.symbol_info(sym)
            if tick is not None and info is not None:
                px = tick.bid if is_buy else tick.ask
                profit = (px - open_px) * (1 if is_buy else -1) * vol * info.trade_contract_size
            out.append(SimpleNamespace(
                ticket=str(p.get("position_id")), identifier=str(p.get("position_id")), symbol=sym,
                volume=vol, type=self.ORDER_TYPE_BUY if is_buy else self.ORDER_TYPE_SELL,
                price_open=open_px, sl=_f(o.get("stop_loss_price")), tp=_f(o.get("take_profit_price")),
                profit=profit + _f(p.get("swap")) - abs(_f(p.get("commission"))), swap=_f(p.get("swap")),
                magic=None, comment="", time=int(_to_dt(p.get("create_time")).timestamp()) if p.get("create_time") else 0,
                _sltp_order_id=o.get("order_id"),
            ))
        return tuple(out)

    def history_deals_get(self, date_from, date_to, **kwargs):
        reverse = {v: k for k, v in self.symbol_map.items()}
        deals, cursor = [], None
        try:
            for _ in range(50):
                page = self._call(self.client.deals, _iso(_to_dt(date_from)), _iso(_to_dt(date_to)), cursor, 200)
                for item in page.get("items") or []:
                    d = item.get("deal") or {}
                    inst = str(d.get("instrument") or "")
                    deals.append(SimpleNamespace(
                        ticket=str(d.get("deal_id")), order=str(d.get("order_id") or ""),
                        position_id=str(d.get("position_id") or ""), symbol=reverse.get(inst, inst),
                        type=self.ORDER_TYPE_BUY if str(d.get("direction")) == "buy" else self.ORDER_TYPE_SELL,
                        entry=self.DEAL_ENTRY_OUT if str(d.get("type")) == "close" else self.DEAL_ENTRY_IN,
                        volume=_f(d.get("volume")), price=_f(d.get("price")),
                        profit=_f((d.get("profit") or {}).get("amount")), commission=-abs(_f(d.get("commission"))),
                        swap=0.0, magic=0, comment=str(d.get("type") or ""),
                        time=int(_to_dt(d.get("create_time")).timestamp()) if d.get("create_time") else 0,
                    ))
                cursor = page.get("next_cursor")
                if not page.get("has_more") or not cursor:
                    break
        except Exception:
            return None
        return tuple(deals)

    # ---------------------------------------------------------------- trading
    def _fmt_volume(self, symbol, volume) -> str:
        info = self.symbol_info(symbol)
        step = info.volume_step if info else 0.01
        return f"{float(volume):.{_decimals(step)}f}"

    def _fmt_price(self, symbol, price) -> str | None:
        if price in (None, 0, 0.0):
            return None
        info = self.symbol_info(symbol)
        return f"{float(price):.{info.digits if info else 5}f}"

    def _await_operation(self, ack: dict, timeout_s: float = 8.0) -> tuple[bool, int, str]:
        op_id = (ack or {}).get("operation_id")
        if not op_id:
            return True, self.TRADE_RETCODE_DONE, "accepted"
        deadline = time.time() + timeout_s
        status = {}
        while time.time() < deadline:
            status = self._call(self.client.operation_status, str(op_id)) or {}
            st = str(status.get("status", "pending"))
            if st == "confirmed":
                return True, self.TRADE_RETCODE_DONE, "confirmed"
            if st in ("rejected", "failed"):
                code = int(status.get("error_code") or 0)
                return False, _ERROR_TO_RETCODE.get(code, 10006), str(status.get("error_message") or st)
            time.sleep(0.25)
        return False, 10012, "operation timeout"  # TRADE_RETCODE_TIMEOUT

    def _result(self, retcode, request, comment="", order=0):
        self._snap_cache = None  # positions changed
        order = int(order) if str(order).isdigit() else 0
        return SimpleNamespace(retcode=int(retcode), order=order, deal=0, volume=float(request.get("volume", 0) or 0),
                               price=float(request.get("price", 0) or 0), comment=str(comment), request=request)

    def order_send(self, request: dict):
        req = dict(request or {})
        symbol = str(req.get("symbol", ""))
        inst = self.instrument(symbol)
        key = uuid.uuid4().hex
        try:
            action = int(req.get("action", 0))
            if action == self.TRADE_ACTION_DEAL and req.get("position"):
                params = {"volume": self._fmt_volume(symbol, req["volume"])}
                ack = self._call(self.client.close_position, str(req["position"]), params, key)
            elif action == self.TRADE_ACTION_DEAL:
                body = {
                    "instrument": inst,
                    "side": "buy" if int(req.get("type", 0)) == self.ORDER_TYPE_BUY else "sell",
                    "volume": self._fmt_volume(symbol, req["volume"]),
                    "comment": str(req.get("comment", ""))[:31],
                }
                sl, tp = self._fmt_price(symbol, req.get("sl")), self._fmt_price(symbol, req.get("tp"))
                if sl:
                    body["stop_loss_price"] = sl
                if tp:
                    body["take_profit_price"] = tp
                ack = self._call(self.client.open_position, body, key)
            elif action == self.TRADE_ACTION_SLTP:
                pos_id = str(req.get("position"))
                existing = next((p for p in (self.positions_get() or []) if str(p.ticket) == pos_id), None)
                sl, tp = self._fmt_price(symbol, req.get("sl")), self._fmt_price(symbol, req.get("tp"))
                order_id = getattr(existing, "_sltp_order_id", None) if existing else None
                if order_id:
                    ack = self._call(self.client.modify_order, str(order_id),
                                     {"stop_loss_price": sl, "take_profit_price": tp}, key)
                else:
                    body = {"type": "sltp", "position_id": pos_id}
                    if sl:
                        body["stop_loss_price"] = sl
                    if tp:
                        body["take_profit_price"] = tp
                    ack = self._call(self.client.place_order, body, key)
            else:
                return self._result(10013, req, f"unsupported action {action}")  # INVALID
            ok, retcode, msg = self._await_operation(ack or {})
            return self._result(retcode, req, msg, order=(ack or {}).get("operation_id") or 0)
        except ExnessApiError as exc:
            return self._result(_ERROR_TO_RETCODE.get(exc.code, 10006), req, exc.message)
        except requests.RequestException as exc:
            return self._result(10031, req, str(exc))


# --------------------------------------------------------------------------- config
def _cfg_section() -> dict:
    try:
        import yaml

        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        sec = cfg.get("exness_api", {}) or {}
        return sec if isinstance(sec, dict) else {}
    except Exception:
        return {}


def _resolve(value):
    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        return os.environ.get(value[2:-1], "")
    return value


def client_from_env() -> ExnessClient:
    sec = _cfg_section()
    api_key = os.environ.get("EXNESS_API_KEY") or _resolve(sec.get("api_key", ""))
    account_id = os.environ.get("EXNESS_ACCOUNT_ID") or _resolve(sec.get("account_id", ""))
    base_url = os.environ.get("EXNESS_BASE_URL") or _resolve(sec.get("base_url", "")) or DEFAULT_BASE_URL
    key_material = os.environ.get("EXNESS_PRIVATE_KEY") or _resolve(sec.get("private_key", ""))
    key_file = os.environ.get("EXNESS_PRIVATE_KEY_FILE") or _resolve(sec.get("private_key_file", ""))
    if not key_material and key_file:
        with open(os.path.expanduser(str(key_file)), "r", encoding="utf-8") as f:
            key_material = f.read()
    missing = [n for n, v in (("EXNESS_API_KEY", api_key), ("EXNESS_ACCOUNT_ID", account_id),
                              ("EXNESS_BASE_URL (Gateway host from Key Management)", base_url),
                              ("EXNESS_PRIVATE_KEY(_FILE)", key_material)) if not v]
    if missing:
        raise RuntimeError(f"missing Exness API credentials: {', '.join(missing)}")
    return ExnessClient(str(api_key), load_ed25519_private_key(str(key_material)), str(account_id), str(base_url))


def symbol_map_from_cfg() -> dict:
    m = _cfg_section().get("symbol_map", {}) or {}
    return {str(k): str(v) for k, v in m.items()} if isinstance(m, dict) else {}


# --------------------------------------------------------------------------- CLI
def _check(symbols: list[str]) -> int:
    """Read-only connectivity check. Places no orders."""
    shim = ExnessMT5(symbol_map=symbol_map_from_cfg())
    if not shim.initialize():
        print("INIT FAILED:", shim.last_error())
        return 1
    c = shim.client
    print("access point:", c.trading_url)
    print("account:", json.dumps(c.account_details(), default=str)[:400])
    try:
        print("limits:", json.dumps(c.limits(), default=str)[:600])
    except Exception as exc:
        print("limits: unavailable", exc)
    print("instruments:", len(shim._instruments), shim._instruments[:20])
    acc = shim.account_info()
    print("balance/equity/free:", acc and (acc.balance, acc.equity, acc.margin_free))
    for sym in symbols:
        inst = shim.instrument(sym)
        info = shim.symbol_info(sym)
        tick = shim.symbol_info_tick(sym)
        rates = shim.copy_rates_from_pos(sym, shim.TIMEFRAME_M5, 1, 1200)
        print(f"\n{sym} -> {inst}")
        print("  conditions:", info and {k: getattr(info, k) for k in ("digits", "volume_min", "volume_max", "volume_step",
                                                                      "trade_contract_size", "filling_mode", "trade_mode")})
        print("  tick:", tick and (tick.bid, tick.ask, tick.time))
        if rates is not None and len(rates):
            last = datetime.fromtimestamp(int(rates[-1]["time"]), tz=timezone.utc)
            print(f"  M5 closed bars: {len(rates)} (last {last.isoformat()}, close {rates[-1]['close']})")
        else:
            print("  M5 bars: NONE")
    print("\npositions:", [(p.symbol, p.volume, p.type, p.sl, p.tp) for p in shim.positions_get() or []])
    now = datetime.now(timezone.utc)
    deals = shim.history_deals_get(now - timedelta(days=7), now)
    print("deals (7d):", None if deals is None else len(deals))
    return 0


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "check":
        syms = sys.argv[2:] or ["BTCUSDm", "XAUUSDm"]
        raise SystemExit(_check(syms))
    print("usage: python -m Python.exness_api check [SYMBOL ...]")
