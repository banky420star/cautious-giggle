"""
n8n ↔ AGI Bridge — called by n8n Execute Command nodes.
Connects to the AGI Server socket, sends ANALYZE/TRADE commands,
parses the response, and outputs JSON for n8n to consume.
"""
import sys
import socket
import json


def main():
    if len(sys.argv) < 3:
        print(json.dumps({
            "error": "Missing arguments. Usage: python agi_n8n_bridge.py <COMMAND> <SYMBOL>",
            "action": "ERROR",
            "confidence": 0.0,
        }))
        sys.exit(1)

    command = sys.argv[1]
    symbol = sys.argv[2]

    try:
        # Connect to the AGI Server socket
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.settimeout(15.0)
        client.connect(("127.0.0.1", 9090))

        # Send command
        msg = f"{command} {symbol}"
        client.send(msg.encode())

        # Receive response (may be longer now with lots/SL/TP)
        chunks = []
        while True:
            try:
                chunk = client.recv(4096)
                if not chunk:
                    break
                chunks.append(chunk)
            except socket.timeout:
                break
        client.close()
        response = b"".join(chunks).decode().strip()

        # ── Parse the AGI Server response ────────────────────────────
        # New format:
        #   "TRADE: BUY EURUSD @ 1.05000 | conf=0.9123 | lots=0.02 | sl=1.04500 | tp=1.06000"
        #   "HOLD EURUSD @ 1.05000 | conf=0.6543"
        result = {
            "raw_response": response,
            "symbol": symbol,
            "confidence": 0.0,
            "action": "ERROR",
            "lots": 0.0,
            "sl": 0.0,
            "tp": 0.0,
        }

        if "|" in response:
            parts = [p.strip() for p in response.split("|")]
            action_part = parts[0]

            # Parse all key=value segments
            for part in parts[1:]:
                if "=" in part:
                    key, val = part.split("=", 1)
                    key = key.strip()
                    try:
                        result[key] = float(val.strip())
                    except ValueError:
                        result[key] = val.strip()

            # Determine action
            if action_part.startswith("TRADE:"):
                result["action"] = action_part.replace("TRADE:", "").strip()
            elif action_part.startswith("HOLD"):
                result["action"] = "HOLD"

        # Output JSON so n8n can parse it
        print(json.dumps(result))

    except ConnectionRefusedError:
        print(json.dumps({
            "error": "AGI Server not running on 127.0.0.1:9090",
            "symbol": symbol,
            "confidence": 0.0,
            "action": "ERROR",
        }))
        sys.exit(1)

    except Exception as e:
        print(json.dumps({
            "error": str(e),
            "symbol": symbol,
            "confidence": 0.0,
            "action": "ERROR",
        }))
        sys.exit(1)


if __name__ == "__main__":
    main()
