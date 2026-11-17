"""Demo / live account profiles for the dashboard's account switch.

    demo -> ~/.exness/env        live -> ~/.exness/live.env        active -> ~/.exness/active

Secrets never leave this module: status() reports only which settings are present or missing,
and the connection check returns account type, balance and currency.

Going live requires: a complete live.env, a passing read-only connection check that reports a
REAL (non-demo) account, and the typed confirmation "GO LIVE". The demo-only settings (learning
mode, higher risk, stacking, all-markets trading) switch off by themselves on a real account,
because Server_AGI only applies them when the broker reports a demo account.
"""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR = os.path.expanduser("~/.exness")
FILES = {"demo": os.path.join(DIR, "env"), "live": os.path.join(DIR, "live.env")}
ACTIVE = os.path.join(DIR, "active")
REQUIRED = ("EXNESS_API_KEY", "EXNESS_ACCOUNT_ID", "EXNESS_BASE_URL")
CONFIRM = "GO LIVE"

TEMPLATE = """# LIVE (real money) Exness account for cautious-giggle.
# Fill in the values from Exness Personal Area -> Key Management for your REAL account,
# then switch in the dashboard (account button, top left). Never share this file.
export AGI_BROKER_BACKEND=exness_api
export EXNESS_API_KEY=
export EXNESS_PRIVATE_KEY_FILE=~/.exness/live_api_private_key
export EXNESS_ACCOUNT_ID=
export EXNESS_BASE_URL=
"""


def active() -> str:
    try:
        with open(ACTIVE, encoding="utf-8") as f:
            v = f.read().strip()
        return v if v in FILES else "demo"
    except OSError:
        return "demo"


def parse_env(path: str) -> dict:
    out = {}
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[7:]
                if "=" not in line:
                    continue
                k, v = line.split("=", 1)
                try:
                    v = " ".join(shlex.split(v, comments=True))
                except ValueError:
                    v = v.strip().strip("'\"")
                out[k.strip()] = os.path.expanduser(v)
    except OSError:
        pass
    return out


def status(profile: str) -> dict:
    path = FILES[profile]
    env = parse_env(path)
    missing = [k for k in REQUIRED if not env.get(k)]
    key_file = env.get("EXNESS_PRIVATE_KEY_FILE")
    if not env.get("EXNESS_PRIVATE_KEY") and not (key_file and os.path.isfile(key_file)):
        missing.append("EXNESS_PRIVATE_KEY_FILE (file not found)" if key_file else "EXNESS_PRIVATE_KEY_FILE")
    return {"profile": profile, "exists": os.path.isfile(path), "ready": os.path.isfile(path) and not missing,
            "missing": missing, "file": path.replace(os.path.expanduser("~"), "~")}


def ensure_template() -> None:
    p = FILES["live"] + ".example"
    if not os.path.exists(FILES["live"]) and not os.path.exists(p):
        os.makedirs(DIR, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write(TEMPLATE)
        os.chmod(p, 0o600)


CHECK = r"""
import json
from Python.mt5_compat import mt5
from Python import demo_exploration
ok = bool(mt5.initialize())
a = mt5.account_info() if ok else None
print(json.dumps({"ok": bool(ok and a is not None), "is_demo": demo_exploration.account_is_demo(mt5) if ok else None,
                  "balance": float(getattr(a, "balance", 0) or 0) if a else None,
                  "currency": str(getattr(a, "currency", "") or "") if a else None}))
"""


def check(profile: str, timeout: int = 60) -> dict:
    """Read-only connection check with that profile's settings (no orders)."""
    st = status(profile)
    if not st["ready"]:
        return {"ok": False, "error": "settings incomplete: " + ", ".join(st["missing"])}
    env = {k: v for k, v in os.environ.items() if not k.startswith("EXNESS_")}
    env.update(parse_env(FILES[profile]))
    env["PYTHONPATH"] = ROOT
    try:
        r = subprocess.run([os.path.join(ROOT, ".venv", "bin", "python"), "-c", CHECK], cwd=ROOT, env=env,
                           capture_output=True, text=True, timeout=timeout)
        line = [x for x in r.stdout.splitlines() if x.startswith("{")]
        if not line:
            return {"ok": False, "error": "could not connect with these settings (check the key, account id and gateway)"}
        return json.loads(line[-1])
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "the broker did not answer in time"}
    except Exception as exc:
        return {"ok": False, "error": type(exc).__name__}


def restart_bot() -> None:
    subprocess.call(["scripts/mac/stop_bot.sh"], cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        os.remove(os.path.join(ROOT, ".tmp", "stop"))
    except FileNotFoundError:
        pass
    subprocess.call(["zsh", "-c", "sleep 3; rm -f .tmp/stop; scripts/mac/start_bot.sh"], cwd=ROOT,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def switch(profile: str, confirm: str = "") -> dict:
    if profile not in FILES:
        return {"ok": False, "error": "unknown profile"}
    if profile == "live":
        if confirm.strip().upper() != CONFIRM:
            return {"ok": False, "error": f'type "{CONFIRM}" to confirm'}
        chk = check("live")
        if not chk.get("ok"):
            return {"ok": False, "error": chk.get("error") or "live connection check failed"}
        if chk.get("is_demo"):
            return {"ok": False, "error": "those keys belong to a DEMO account; live.env must point at your real account"}
    else:
        chk = check("demo")
        if not chk.get("ok"):
            return {"ok": False, "error": chk.get("error") or "demo connection check failed"}
    os.makedirs(DIR, exist_ok=True)
    with open(ACTIVE, "w", encoding="utf-8") as f:
        f.write(profile)
    restart_bot()
    return {"ok": True, "profile": profile, "account": chk}


def env_for(profile: str) -> dict:
    return parse_env(FILES[profile])


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "status":
        print(json.dumps({"active": active(), "demo": status("demo"), "live": status("live")}, indent=1))
    elif cmd == "check":
        print(json.dumps(check(sys.argv[2] if len(sys.argv) > 2 else active())))
