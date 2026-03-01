import os
import sys
import json
import asyncio
from loguru import logger

from Python.hybrid_brain import HybridBrain

# --- Config ---
HOST = os.environ.get("AGI_HOST", "0.0.0.0")
PORT = int(os.environ.get("AGI_PORT", "9090"))

AGI_TOKEN = os.environ.get("AGI_TOKEN", "").strip()
ENABLE_PNL_POLL = os.environ.get("AGI_PNL_POLL", "true").lower() == "true"
PNL_POLL_INTERVAL_SEC = int(os.environ.get("AGI_PNL_POLL_INTERVAL_SEC", "20"))

MAX_LINE_BYTES = int(os.environ.get("AGI_MAX_LINE_BYTES", str(1024 * 1024)))  # 1MB

# Live toggle
IS_LIVE = "--live" in sys.argv

if IS_LIVE and not AGI_TOKEN:
    logger.error("🔥 CRITICAL ENFORCEMENT: Live mode refused without AGI_TOKEN. Shutting down.")
    sys.exit(1)

if not AGI_TOKEN and HOST == "0.0.0.0":
    logger.warning("No AGI_TOKEN provided. Forcing local binding (127.0.0.1) for safety.")
    HOST = "127.0.0.1"

brain = HybridBrain(paper_mode=not IS_LIVE)

# MT5 optional
try:
    import MetaTrader5 as mt5
except Exception:
    mt5 = None


def _auth_ok(request: dict) -> bool:
    if not AGI_TOKEN:
        return True
    token = (request.get("token") or "").strip()
    return token == AGI_TOKEN


async def _write(writer: asyncio.StreamWriter, payload: dict):
    writer.write((json.dumps(payload) + "\n").encode("utf-8"))
    await writer.drain()


async def poll_closed_trades_loop():
    """
    Feeds realized PnL into RiskEngine so:
    - daily loss kill switch works
    - canary promotion/rollback works
    """
    if brain.paper_mode:
        logger.info("PnL polling disabled (paper mode).")
        return
    if not ENABLE_PNL_POLL:
        logger.info("PnL polling disabled (AGI_PNL_POLL=false).")
        return
    if mt5 is None:
        logger.warning("PnL polling disabled (MetaTrader5 package missing).")
        return

    logger.info("✅ MT5 PnL polling loop started.")

    last_seen_ticket = 0

    while True:
        try:
            if not mt5.initialize():
                logger.warning("MT5 initialize failed in PnL poll loop; retrying...")
                await asyncio.sleep(PNL_POLL_INTERVAL_SEC)
                continue

            # today range (UTC-ish). Good enough for now.
            from datetime import datetime, timezone
            utc_from = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
            utc_to = datetime.now(timezone.utc)

            deals = mt5.history_deals_get(utc_from, utc_to)
            if deals is None:
                await asyncio.sleep(PNL_POLL_INTERVAL_SEC)
                continue

            # Process new deals
            for d in sorted(deals, key=lambda x: int(getattr(x, "ticket", 0))):
                ticket = int(getattr(d, "ticket", 0))
                if ticket <= last_seen_ticket:
                    continue

                profit = float(getattr(d, "profit", 0.0))

                # Only record deals that actually change PnL
                if profit != 0.0:
                    brain.risk_engine.record_closed_trade(profit)

                last_seen_ticket = max(last_seen_ticket, ticket)

        except Exception as e:
            logger.warning(f"PnL poll loop error: {e}")

        await asyncio.sleep(PNL_POLL_INTERVAL_SEC)


async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    addr = writer.get_extra_info("peername")
    logger.info(f"Connection from {addr}")

    try:
        raw = await asyncio.wait_for(reader.readline(), timeout=10.0)
        if not raw:
            return
        if len(raw) > MAX_LINE_BYTES:
            await _write(writer, {"status": "error", "error": "Request too large"})
            return

        message = raw.decode("utf-8", errors="replace").strip()
        if not message:
            return

        request = json.loads(message)

        if not _auth_ok(request):
            await _write(writer, {"status": "unauthorized", "error": "Invalid AGI_TOKEN"})
            logger.warning(f"Unauthorized payload blocked from {addr}")
            return

        action = (request.get("action") or "").lower().strip()

        if action in ("predict", "trade"):
            symbol = request.get("symbol", "EURUSD")
            direction = request.get("direction", "AUTO")
            confidence = float(request.get("confidence", 0.0))
            aggression = (request.get("aggression") or "moderate").lower()

            result = await brain.live_trade(symbol, direction, confidence, aggression=aggression)

            payload = {
                "status": result.get("status"),
                "symbol": symbol,
                "action": result.get("action", direction if direction != "AUTO" else "HOLD"),
                "lot": result.get("lot"),
                "price": result.get("price"),
                "sl": result.get("sl"),
                "tp": result.get("tp"),
                "ticket": result.get("ticket"),
                "paper_mode": brain.paper_mode,
                "error": result.get("error"),
            }
            await _write(writer, payload)
            return

        if action == "health":
            await _write(writer, {"status": "healthy", "paper_mode": brain.paper_mode})
            return

        if action == "risk_status":
            await _write(writer, {"status": "ok", "risk": brain.risk_engine.status()})
            return

        await _write(writer, {"status": "error", "error": f"Unknown action: {action}"})

    except asyncio.TimeoutError:
        logger.warning(f"Client timeout from {addr}")
    except Exception as e:
        logger.error(f"Client error: {e}")
        try:
            await _write(writer, {"status": "error", "error": str(e)})
        except Exception:
            pass
    finally:
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:
            pass


async def main():
    server = await asyncio.start_server(handle_client, HOST, PORT)
    addr = server.sockets[0].getsockname()
    logger.success(f"🚀 Grok AGI Server running on {addr} | Paper Mode: {brain.paper_mode} | Auth: {'ON' if AGI_TOKEN else 'OFF'}")

    # Start autonomy loop in background (DO NOT await directly)
    from Python.autonomy_loop import AutonomyLoop
    autoloop = AutonomyLoop(brain)
    asyncio.create_task(autoloop.start())

    # Start MT5 PnL polling in background
    asyncio.create_task(poll_closed_trades_loop())

    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())
