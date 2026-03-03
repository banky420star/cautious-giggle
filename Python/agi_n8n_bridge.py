"""
n8n ↔ AGI Bridge
Usage:
  python agi_n8n_bridge.py <COMMAND> <SYMBOL> [AGGRESSION]

COMMAND: predict | trade | health | risk_status
"""
import os
import sys
import socket
import json
import time

HOST = os.environ.get("AGI_HOST", "127.0.0.1")
PORT = int(os.environ.get("AGI_PORT", "9090"))
TOKEN = os.environ.get("AGI_TOKEN", "").strip()
TIMEOUT = float(os.environ.get("AGI_SOCKET_TIMEOUT", "10"))
RETRIES = int(os.environ.get("AGI_SOCKET_RETRIES", "3"))


def _die(msg: dict, code: int = 1):
    print(json.dumps(msg))
    sys.exit(code)


def _send_once(request: dict) -> dict:
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.settimeout(TIMEOUT)
    client.connect((HOST, PORT))

    payload = (json.dumps(request) + "\n").encode("utf-8")
    client.sendall(payload)

    buf = b""
    while b"\n" not in buf:
        chunk = client.recv(4096)
        if not chunk:
            break
        buf += chunk

    client.close()
    raw = buf.decode("utf-8", errors="replace").strip()
    if not raw:
        raise RuntimeError("Empty response from server")

    return json.loads(raw.splitlines()[0])


def main():
    if len(sys.argv) < 2:
        _die({
            "error": "Usage: python agi_n8n_bridge.py <COMMAND> <SYMBOL> [AGGRESSION]",
            "action": "ERROR",
            "confidence": 0.0,
        })

    command = sys.argv[1].strip().lower()
    symbol = (sys.argv[2].strip() if len(sys.argv) >= 3 else "EURUSDm")
    aggression = (sys.argv[3].strip().lower() if len(sys.argv) >= 4 else "moderate")

    request = {
        "action": command,
        "symbol": symbol,
        "direction": "AUTO",
        "confidence": 0.0,
        "aggression": aggression,
    }
    if TOKEN:
        request["token"] = TOKEN

    last_err = None
    for i in range(1, RETRIES + 1):
        try:
            result = _send_once(request)
            if "action" not in result:
                result["action"] = result.get("status", "UNKNOWN")
            print(json.dumps(result))
            return
        except ConnectionRefusedError:
            last_err = f"AGI Server not running on {HOST}:{PORT}"
        except Exception as e:
            last_err = str(e)

        if i < RETRIES:
            time.sleep(0.4 * i)

    _die({
        "error": last_err or "Unknown socket error",
        "symbol": symbol,
        "confidence": 0.0,
        "action": "ERROR",
    })


if __name__ == "__main__":
    main()
