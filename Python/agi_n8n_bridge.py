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

        # Send command as JSON per Server_AGI expectations
        request = {
            "action": command.lower(),
            "symbol": symbol,
            "direction": "AUTO",
            "confidence": 0.0
        }
        msg = json.dumps(request)
        client.send(msg.encode())

        # Receive response
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
        # Server now replies exclusively in JSON
        try:
            result = json.loads(response)
            # Ensure n8n gets standard keys
            if "action" not in result:
                result["action"] = result.get("status", "UNKNOWN")
        except json.JSONDecodeError:
            result = {
                "raw_response": response,
                "symbol": symbol,
                "confidence": 0.0,
                "action": "ERROR",
                "error": "Failed to parse server JSON"
            }

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
