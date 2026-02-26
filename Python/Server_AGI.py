import socket
import json
import asyncio
import sys
from loguru import logger
from Python.hybrid_brain import HybridBrain

# Global hybrid brain (paper mode enabled by default)
is_live = "--live" in sys.argv
brain = HybridBrain(paper_mode=not is_live)

async def handle_client(reader, writer):
    addr = writer.get_extra_info('peername')
    logger.info(f"Connection from {addr}")

    try:
        data = await reader.read(1024)
        message = data.decode().strip()
        
        if not message:
            return

        request = json.loads(message)
        action = request.get("action")

        if action in ["predict", "trade"]:
            symbol = request.get("symbol", "EURUSD")
            direction = request.get("direction", "AUTO")
            confidence = float(request.get("confidence", 0.0))

            result = await brain.live_trade(symbol, direction, float(confidence))
            
            # Extract final direction from the result or fallback to requested if it somehow slipped
            final_action = result.get("action", direction if direction not in ["AUTO"] else "HOLD")
            
            response = json.dumps({
                "status": result.get("status"),
                "symbol": symbol,
                "action": final_action,
                "lot": result.get("lot"),
                "sl": result.get("sl"),
                "tp": result.get("tp"),
                "paper_mode": brain.paper_mode
            })

            writer.write(response.encode() + b'\n')
            await writer.drain()

        elif action == "health":
            writer.write(json.dumps({"status": "healthy", "paper_mode": brain.paper_mode}).encode() + b'\n')
            await writer.drain()

    except Exception as e:
        logger.error(f"Client error: {e}")
    finally:
        writer.close()
        await writer.wait_closed()

async def main():
    server = await asyncio.start_server(handle_client, '0.0.0.0', 9090)
    addr = server.sockets[0].getsockname()
    logger.success(f"🚀 Grok AGI Server running on {addr} | Paper Mode: {brain.paper_mode}")

    async with server:
        await server.serve_forever()

if __name__ == "__main__":
    asyncio.run(main())
