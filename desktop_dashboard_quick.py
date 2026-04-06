"""Failsafe dashboard launcher for Cautious Giggle.

Starts the UI server on the first free local port and opens the dashboard in the
system browser. This avoids pywebview and avoids hard-failing if 8088 is busy.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent
PYTHON = sys.executable
HOST = '127.0.0.1'
PREFERRED_PORTS = [8088, 8090, 8091, 8092, 8093, 8094, 8095]
UI_SCRIPT = BASE_DIR / 'tools' / 'project_status_ui.py'
NO_WINDOW = getattr(subprocess, 'CREATE_NO_WINDOW', 0)


def port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        return sock.connect_ex((host, port)) == 0


def http_ready(url: str, timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return 200 <= int(response.status) < 500
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def candidate_urls(port: int) -> list[str]:
    return [
        f'http://{HOST}:{port}/app',
        f'http://{HOST}:{port}/',
        f'http://{HOST}:{port}/mini',
    ]


def find_working_port() -> int | None:
    for port in PREFERRED_PORTS:
        for url in candidate_urls(port):
            if http_ready(url):
                return port
    return None


def find_free_port() -> int:
    for port in PREFERRED_PORTS:
        if not port_open(HOST, port):
            return port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((HOST, 0))
        return int(sock.getsockname()[1])


def launch_ui_server(port: int) -> subprocess.Popen[Any]:
    env = os.environ.copy()
    env.update({'AGI_UI_HOST': HOST, 'AGI_UI_PORT': str(port)})
    return subprocess.Popen(
        [PYTHON, '-u', str(UI_SCRIPT)],
        cwd=str(BASE_DIR),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=NO_WINDOW,
    )


def wait_for_dashboard(port: int, timeout: float = 75.0) -> str | None:
    deadline = time.time() + timeout
    urls = candidate_urls(port)
    while time.time() < deadline:
        for url in urls:
            if http_ready(url):
                return url
        time.sleep(0.5)
    return None


def main() -> int:
    existing_port = find_working_port()
    if existing_port is not None:
        webbrowser.open(candidate_urls(existing_port)[0])
        return 0

    port = find_free_port()
    proc = launch_ui_server(port)
    url = wait_for_dashboard(port)
    if not url:
        print(f'Dashboard startup timed out on port {port}.', file=sys.stderr)
        try:
            proc.terminate()
        except Exception:
            pass
        return 1
    webbrowser.open(url)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
