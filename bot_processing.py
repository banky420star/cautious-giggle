"""
BOTprocessing — React SPA desktop launcher.

Opens the React trading dashboard in a pywebview window.
No page refreshes, no blinking — real-time updates via WebSocket.
"""

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

PORT = 8088
HOST = "127.0.0.1"
TITLE = "BOTprocessing — Cautious Giggle AGI"
WIDTH, HEIGHT = 1520, 960
BASE_DIR = Path(__file__).resolve().parent
PYTHON = sys.executable
ICON_PATH = BASE_DIR / "cautious_giggle.ico"
UI_SCRIPT = BASE_DIR / "tools" / "project_status_ui.py"
APP_URL = f"http://{HOST}:{PORT}/app"

NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def _port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        return sock.connect_ex((host, port)) == 0


def _ensure_ui_server():
    """Start the dashboard backend if not already running."""
    if _port_open(HOST, PORT):
        return
    subprocess.Popen(
        [PYTHON, "-u", str(UI_SCRIPT)],
        cwd=str(BASE_DIR),
        env={**os.environ, "AGI_UI_HOST": HOST, "AGI_UI_PORT": str(PORT)},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=NO_WINDOW,
    )
    deadline = time.time() + 30
    while time.time() < deadline:
        if _port_open(HOST, PORT):
            return
        time.sleep(0.5)


def main():
    import webview

    _ensure_ui_server()

    window = webview.create_window(
        TITLE,
        url=APP_URL,
        width=WIDTH,
        height=HEIGHT,
        min_size=(1024, 680),
        resizable=True,
        text_select=True,
        zoomable=True,
        easy_drag=False,
    )

    if ICON_PATH.exists():
        try:
            window.icon = str(ICON_PATH)
        except Exception:
            pass

    webview.start(debug="--debug" in sys.argv, http_server=False)


if __name__ == "__main__":
    main()
