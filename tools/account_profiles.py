"""Demo / live account profiles for the dashboard's account switch.

    demo -> ~/.exness/env        live -> ~/.exness/live.env        active -> ~/.exness/active

Secrets never leave this module: status() reports only which settings are present or missing,
and the connection check returns account type, balance and currency.

Going live requires: a complete live.env, a passing read-only connection check that reports a
REAL (non-demo) account, and the typed confirmation "GO LIVE". With
trading.demo_learning.same_on_live (the default) the live account then runs exactly like demo:
same risk, markets, stacking, learning, council, exploration and news trading.
"""
from __future__ import annotations

import contextlib
import json
import os
import re
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


LIVE_KEY_FILE = os.path.join(DIR, "live_api_private_key")
_TOKEN = re.compile(r"^[A-Za-z0-9._:-]{4,200}$")
_GATEWAY = re.compile(r"^(https?://)?[A-Za-z0-9.-]+(:\d+)?(/[A-Za-z0-9._/-]*)?$")


def _write_private(path: str, text: str) -> None:
    """Write a secret file readable only by this user (created 0600, swapped in atomically)."""
    os.makedirs(DIR, exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(text)
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)


def live_form() -> dict:
    """What the setup form may show: which fields are already saved (never their values) and the gateway
    host the demo account uses (a public address, not a secret) as a suggestion."""
    env = parse_env(FILES["live"])
    return {"saved": {"api_key": bool(env.get("EXNESS_API_KEY")), "account_id": bool(env.get("EXNESS_ACCOUNT_ID")),
                      "gateway": bool(env.get("EXNESS_BASE_URL")), "private_key": os.path.isfile(LIVE_KEY_FILE)},
            "gateway_hint": env.get("EXNESS_BASE_URL") or parse_env(FILES["demo"]).get("EXNESS_BASE_URL", "")}


def save_live(api_key: str = "", account_id: str = "", gateway: str = "", private_key: str = "") -> dict:
    """Save the live account settings typed into the dashboard. Blank fields keep what is already saved.
    Values are validated, written 0600 to ~/.exness, and never returned."""
    old = parse_env(FILES["live"])
    api_key = (api_key or "").strip() or old.get("EXNESS_API_KEY", "")
    account_id = (account_id or "").strip() or old.get("EXNESS_ACCOUNT_ID", "")
    gateway = (gateway or "").strip().rstrip("/") or old.get("EXNESS_BASE_URL", "")
    private_key = (private_key or "").strip()
    errors = {}
    if not _TOKEN.match(api_key):
        errors["api_key"] = "Paste the API key ID from Key Management (letters, digits, - _ . only)."
    if not re.match(r"^[A-Za-z0-9_-]{3,40}$", account_id):
        errors["account_id"] = "Enter the trading account number of your REAL account."
    if not _GATEWAY.match(gateway or ""):
        errors["gateway"] = "Enter the gateway host shown in Key Management (for example api.exness.com)."
    if private_key:
        try:
            sys.path.insert(0, ROOT)
            from Python.exness_api import load_ed25519_private_key
            load_ed25519_private_key(private_key)
        except Exception:
            errors["private_key"] = "That isn't a valid Exness private key (128 hex characters, or a PEM key)."
        else:
            demo_key = parse_env(FILES["demo"]).get("EXNESS_PRIVATE_KEY_FILE", "")
            try:
                with open(demo_key, encoding="utf-8") if demo_key else contextlib.nullcontext() as _f:
                    _same = bool(_f) and _f.read().strip() == private_key
                if _same:
                    errors["private_key"] = "This is your DEMO key. Create a separate key for the real account."
            except OSError:
                pass
    elif not os.path.isfile(LIVE_KEY_FILE):
        errors["private_key"] = "Paste the private key you downloaded when creating the API key."
    if errors:
        return {"ok": False, "errors": errors}
    if private_key:
        _write_private(LIVE_KEY_FILE, private_key + "\n")
    _write_private(FILES["live"], "# LIVE (real money) Exness account, saved from the dashboard. Never share this file.\n"
                   "export AGI_BROKER_BACKEND=exness_api\n"
                   f"export EXNESS_API_KEY={shlex.quote(api_key)}\n"
                   f"export EXNESS_PRIVATE_KEY_FILE={shlex.quote(LIVE_KEY_FILE)}\n"
                   f"export EXNESS_ACCOUNT_ID={shlex.quote(account_id)}\n"
                   f"export EXNESS_BASE_URL={shlex.quote(gateway)}\n")
    return {"ok": True, "live": status("live")}


def env_for(profile: str) -> dict:
    return parse_env(FILES[profile])


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "status":
        print(json.dumps({"active": active(), "demo": status("demo"), "live": status("live")}, indent=1))
    elif cmd == "check":
        print(json.dumps(check(sys.argv[2] if len(sys.argv) > 2 else active())))
