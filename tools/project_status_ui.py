import json
import os
import subprocess
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(ROOT, "logs")
ACTIVE_PATH = os.path.join(ROOT, "models", "registry", "active.json")


def _run(cmd):
    try:
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, cwd=ROOT)
    except Exception as exc:
        return f"ERROR: {exc}"


def _tail(path, lines=40):
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            data = f.readlines()
        return [x.rstrip("\n") for x in data[-lines:]]
    except Exception:
        return []


def _running_processes():
    ps = [
        "powershell",
        "-NoProfile",
        "-Command",
        "Get-CimInstance Win32_Process | "
        "Where-Object { $_.Name -match 'python|powershell' } | "
        "Select-Object ProcessId,Name,CommandLine | ConvertTo-Json -Depth 3",
    ]
    raw = _run(ps)
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            parsed = [parsed]
        keep = []
        for p in parsed:
            cmd = (p.get("CommandLine") or "")
            if "cautious-giggle" in cmd.lower() or "python" in (p.get("Name") or "").lower():
                keep.append(
                    {
                        "pid": p.get("ProcessId"),
                        "name": p.get("Name"),
                        "cmd": cmd,
                    }
                )
        return keep
    except Exception:
        return []


def read_status():
    git_status = _run(["git", "-C", ROOT, "status", "-sb"]).strip()

    active = {}
    if os.path.exists(ACTIVE_PATH):
        try:
            with open(ACTIVE_PATH, "r", encoding="utf-8") as f:
                active = json.load(f)
        except Exception:
            active = {}

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": ROOT,
        "git_status": git_status,
        "active_models": active,
        "processes": _running_processes(),
        "logs": {
            "server": _tail(os.path.join(LOG_DIR, "server.log")),
            "lstm": _tail(os.path.join(LOG_DIR, "lstm_training.log")),
            "ppo": _tail(os.path.join(LOG_DIR, "ppo_training.log")),
            "backtester": _tail(os.path.join(LOG_DIR, "backtester.log")),
        },
    }
    return payload


HTML = """<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>cautious-giggle status</title>
  <style>
    :root { --bg:#0f172a; --panel:#111827; --ink:#e5e7eb; --muted:#93c5fd; --line:#1f2937; }
    body { margin:0; background:linear-gradient(120deg,#0f172a,#1e293b); color:var(--ink); font:14px/1.45 Consolas, Menlo, monospace; }
    .wrap { max-width:1200px; margin:0 auto; padding:20px; }
    h1 { margin:0 0 10px; font-size:22px; }
    .meta { color:var(--muted); margin-bottom:16px; }
    .grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(360px,1fr)); gap:12px; }
    .card { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:12px; }
    pre { margin:0; white-space:pre-wrap; word-break:break-word; }
    .proc { padding:6px 0; border-bottom:1px dashed #334155; }
  </style>
</head>
<body>
<div class=\"wrap\">
  <h1>cautious-giggle Read-Only Status</h1>
  <div class=\"meta\" id=\"meta\">loading...</div>
  <div class=\"grid\">
    <div class=\"card\"><h3>Git</h3><pre id=\"git\"></pre></div>
    <div class=\"card\"><h3>Active Models</h3><pre id=\"models\"></pre></div>
    <div class=\"card\"><h3>Processes</h3><div id=\"procs\"></div></div>
    <div class=\"card\"><h3>Server Log</h3><pre id=\"server\"></pre></div>
    <div class=\"card\"><h3>LSTM Log</h3><pre id=\"lstm\"></pre></div>
    <div class=\"card\"><h3>PPO Log</h3><pre id=\"ppo\"></pre></div>
  </div>
</div>
<script>
async function load(){
  const res = await fetch('/api/status');
  const d = await res.json();
  document.getElementById('meta').textContent = `UTC ${d.timestamp_utc} | ${d.repo_root}`;
  document.getElementById('git').textContent = d.git_status || '';
  document.getElementById('models').textContent = JSON.stringify(d.active_models || {}, null, 2);
  const procs = document.getElementById('procs');
  procs.innerHTML = '';
  (d.processes || []).forEach(p => {
    const div = document.createElement('div');
    div.className = 'proc';
    div.textContent = `[${p.pid}] ${p.name} :: ${p.cmd}`;
    procs.appendChild(div);
  });
  document.getElementById('server').textContent = (d.logs.server || []).join('\n');
  document.getElementById('lstm').textContent = (d.logs.lstm || []).join('\n');
  document.getElementById('ppo').textContent = (d.logs.ppo || []).join('\n');
}
load();
setInterval(load, 5000);
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, body: bytes, content_type: str = "text/plain", code: int = 200):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/" or self.path.startswith("/index"):
            self._send(HTML.encode("utf-8"), "text/html; charset=utf-8")
            return
        if self.path.startswith("/api/status"):
            data = json.dumps(read_status(), ensure_ascii=False).encode("utf-8")
            self._send(data, "application/json; charset=utf-8")
            return
        self._send(b"not found", "text/plain", 404)


def run(host="127.0.0.1", port=8088):
    server = HTTPServer((host, int(port)), Handler)
    print(f"Status UI running on http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    run(host=os.environ.get("AGI_UI_HOST", "127.0.0.1"), port=int(os.environ.get("AGI_UI_PORT", "8088")))
