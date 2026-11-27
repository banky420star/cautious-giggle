"""Interactive key and config setup for the installer (scripts/mac/install.sh). Asks in the terminal, validates,
saves secrets readable only by this user, and tests each one against the real service. Nothing is printed back.

  python tools/install_setup.py exness demo|live   Exness API account (demo -> ~/.exness/env, live -> ~/.exness/live.env)
                                    add --later to only save it (the installer asks for keys before packages exist)
  python tools/install_setup.py test demo|live     read-only connection test of a saved account
  python tools/install_setup.py jev                Jev keys (OpenRouter / jevai) -> ~/.config/cautious-giggle/jev.env
  python tools/install_setup.py config             config.yaml from config.yaml.example, live risk choice
  python tools/install_setup.py telegram           optional Telegram alerts
"""
from __future__ import annotations

import getpass
import os
import re
import shlex
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import account_profiles as ap  # noqa: E402

JEV_FILE = os.path.expanduser("~/.config/cautious-giggle/jev.env")
KEY_FILES = {"demo": os.path.join(ap.DIR, "api_private_key"), "live": os.path.join(ap.DIR, "live_api_private_key")}
B, D, G, Y, R, X = "\033[1m", "\033[2m", "\033[32m", "\033[33m", "\033[31m", "\033[0m"


def ask(prompt: str, default: str = "", secret: bool = False) -> str:
    shown = f" {D}[{default}]{X}" if default and not secret else (f" {D}[keep saved]{X}" if default else "")
    while True:
        try:
            v = (getpass.getpass if secret else input)(f"  {prompt}{shown}: ").strip()
        except EOFError:
            v = ""
        if v or default:
            return v or default
        print(f"  {Y}Required.{X}")


def optional_secret(prompt: str) -> str:
    try:
        return getpass.getpass(f"  {prompt}: ").strip()
    except EOFError:
        return ""


def yes(prompt: str, default: bool = True) -> bool:
    try:
        v = input(f"  {prompt} {D}[{'Y/n' if default else 'y/N'}]{X}: ").strip().lower()
    except EOFError:
        v = ""
    return default if not v else v.startswith("y")


def read_private_key() -> str:
    """Paste the key (PEM or 128 hex) or give the path of the downloaded key file."""
    print(f"  Private key: drag the downloaded key file into this window and press Enter,")
    print(f"  {D}or paste the key text and finish with an empty line.{X}")
    lines = []
    while True:
        try:
            ln = input("  > " if not lines else "    ")
        except EOFError:
            break
        p = os.path.expanduser(ln.strip().strip("'\"").replace("\\ ", " "))
        if not lines and p and os.path.isfile(p):
            with open(p, encoding="utf-8") as f:
                return f.read().strip()
        if not ln.strip():
            if lines:
                break
            continue
        lines.append(ln.rstrip())
        if re.fullmatch(r"[0-9a-fA-F]{128}", ln.strip()):
            break
    return "\n".join(lines).strip()


def write_private(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(text)
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)


# ------------------------------------------------------------------ Exness
def key_looks_valid(pk: str) -> bool:
    """Full check with the bot's key loader once packages are installed; a format check before that."""
    try:
        from Python.exness_api import load_ed25519_private_key
    except ImportError:
        import base64
        if "BEGIN" in pk and "PRIVATE KEY" in pk:
            return True
        if re.fullmatch(r"[0-9a-fA-F]{64}|[0-9a-fA-F]{128}", pk):
            return True
        try:
            return len(base64.urlsafe_b64decode(pk + "=" * (-len(pk) % 4))) in (32, 64)
        except Exception:
            return False
    try:
        load_ed25519_private_key(pk)
        return True
    except Exception:
        return False


def exness(profile: str, test: bool = True) -> int:
    real = profile == "live"
    old = ap.parse_env(ap.FILES[profile])
    print(f"\n{B}Exness {'LIVE (real money)' if real else 'DEMO'} account{X}")
    print(f"  {D}Exness Personal Area -> Key Management: create an API key for the {'real' if real else 'demo'} trading account.")
    print(f"  You need the key ID, the private key file you downloaded, the account number and the gateway host.{X}")
    if old and not yes("Settings are already saved. Replace them?", False):
        return show_check(profile) if test else 0
    while True:
        key_id = ask("API key ID", old.get("EXNESS_API_KEY", ""), secret=True)
        account = ask("Trading account number", old.get("EXNESS_ACCOUNT_ID", ""))
        gateway = ask("Gateway host (from Key Management)", old.get("EXNESS_BASE_URL", "") or "https://api.exness.com").rstrip("/")
        if gateway and not gateway.startswith("http"):
            gateway = "https://" + gateway
        pk = read_private_key()
        errs = []
        if not ap._TOKEN.match(key_id):
            errs.append("the API key ID should be letters, digits, - _ . only")
        if not re.fullmatch(r"[A-Za-z0-9_-]{3,40}", account):
            errs.append("the account number looks wrong")
        if not ap._GATEWAY.match(gateway):
            errs.append("the gateway should be a host like https://api.exness.com")
        if pk:
            if not key_looks_valid(pk):
                errs.append("that isn't a valid Exness private key (PEM, or 128 hex characters)")
            other = ap.parse_env(ap.FILES["demo" if real else "live"]).get("EXNESS_PRIVATE_KEY_FILE", "")
            if other and os.path.isfile(other) and open(other, encoding="utf-8").read().strip() == pk:
                errs.append(f"this is the {'demo' if real else 'live'} account's key; each account needs its own")
        elif not os.path.isfile(KEY_FILES[profile]):
            errs.append("the private key is missing")
        if not errs:
            break
        for e in errs:
            print(f"  {R}✗ {e}{X}")
        if not yes("Try again?"):
            return 1
    if pk:
        write_private(KEY_FILES[profile], pk + "\n")
    write_private(ap.FILES[profile],
                  f"# {'LIVE (real money)' if real else 'DEMO'} Exness account for cautious-giggle (installer). Never share this file.\n"
                  "export AGI_BROKER_BACKEND=exness_api\n"
                  f"export EXNESS_API_KEY={shlex.quote(key_id)}\n"
                  f"export EXNESS_PRIVATE_KEY_FILE={shlex.quote(KEY_FILES[profile])}\n"
                  f"export EXNESS_ACCOUNT_ID={shlex.quote(account)}\n"
                  f"export EXNESS_BASE_URL={shlex.quote(gateway)}\n")
    if not os.path.exists(ap.ACTIVE):
        write_private(ap.ACTIVE, "demo")
    if not test:
        print(f"  {G}✓ Saved. The connection is tested once the packages are installed.{X}")
        return 0
    return show_check(profile)


def show_check(profile: str) -> int:
    print(f"  {D}Testing the connection (read-only, no orders)…{X}")
    r = ap.check(profile)
    if not r.get("ok"):
        print(f"  {R}✗ Could not connect: {r.get('error') or 'check the key, account number and gateway'}{X}")
        return 1
    kind = "demo" if r.get("is_demo") else "REAL"
    print(f"  {G}✓ Connected: {kind} account, balance {r.get('balance', 0):,.2f} {r.get('currency', '')}{X}")
    if profile == "live" and r.get("is_demo"):
        print(f"  {Y}! These keys belong to a DEMO account; the live slot needs your real account's key.{X}")
        return 1
    if profile == "demo" and not r.get("is_demo"):
        print(f"  {Y}! This is a REAL account. Use a demo account here; set the real one up as live.{X}")
    return 0


# ------------------------------------------------------------------ Jev
def jev() -> int:
    old = ap.parse_env(JEV_FILE)
    print(f"\n{B}Jev (decision model used by the Jev trader){X}")
    print(f"  {D}Jev is reached through OpenRouter (openrouter.ai -> Keys) or jevai.org. One key is enough;")
    print(f"  leave a field empty to skip it.{X}")
    if old and not yes("Jev keys are already saved. Replace them?", False):
        return 0
    ork = optional_secret(f"OpenRouter API key {D}(sk-or-…, Enter to skip){X}")
    jk = optional_secret(f"jevai.org API key {D}(optional, Enter to skip){X}")
    if not ork and not jk:
        print(f"  {Y}No key: the Jev trader will stay idle (it holds when it can't reach Jev).{X}")
        return 0
    if ork:
        import urllib.error
        import urllib.request
        try:
            req = urllib.request.Request("https://openrouter.ai/api/v1/auth/key", headers={"Authorization": f"Bearer {ork}"})
            urllib.request.urlopen(req, timeout=15).close()
            print(f"  {G}✓ OpenRouter key works{X}")
        except urllib.error.HTTPError as exc:
            print(f"  {Y}! OpenRouter said HTTP {exc.code} (check the key); saved anyway{X}")
        except Exception as exc:
            print(f"  {Y}! Couldn't test the key ({type(exc).__name__}); saved anyway{X}")
    lines = ["# Jev keys for cautious-giggle (installer). Never share this file."]
    if ork:
        lines.append(f"OPENROUTER_API_KEY={ork}")
    if jk:
        lines.append(f"JEV_API_KEY={jk}")
    write_private(JEV_FILE, "\n".join(lines) + "\n")
    print(f"  {G}✓ Saved to ~/.config/cautious-giggle/jev.env{X}")
    return 0


# ------------------------------------------------------------------ config
def config() -> int:
    cfg = os.path.join(ROOT, "config.yaml")
    if os.path.exists(cfg):
        print(f"  {D}config.yaml already exists: kept as it is.{X}")
        return 0
    shutil.copy(os.path.join(ROOT, "config.yaml.example"), cfg)
    print(f"\n{B}Live account risk{X}")
    print(f"  {D}The demo account trades aggressively to learn fast (high risk per trade, no drawdown halt, every market).")
    print(f"  For the live account you can use the same settings, or the bot's real-money rules")
    print(f"  (smaller risk, daily loss and drawdown limits, only markets whose models passed their tests).{X}")
    same = yes("Run the LIVE account exactly like demo?", False)
    s = open(cfg, encoding="utf-8").read()
    s = re.sub(r"(\n    same_on_live: )(true|false)", lambda m: m.group(1) + ("true" if same else "false"), s, count=1)
    open(cfg, "w", encoding="utf-8").write(s)
    print(f"  {G}✓ config.yaml created (live {'= demo settings' if same else 'uses the safer defaults'}){X}")
    return 0


def telegram() -> int:
    import yaml
    cfg = os.path.join(ROOT, "config.yaml")
    t = (yaml.safe_load(open(cfg, encoding="utf-8")) or {}).get("telegram") or {}
    have = t.get("token") and "YOUR_" not in str(t.get("token"))
    print(f"\n{B}Telegram alerts (optional){X}")
    print(f"  {D}Trade alerts and the nightly P&L report on your phone. Create a bot with @BotFather, then send it a")
    print(f"  message and look up your chat id (e.g. with @userinfobot).{X}")
    if not yes("Set up Telegram now?" if not have else "Telegram is set up. Change it?", False):
        return 0
    token = getpass.getpass("  Bot token: ").strip()
    chat = input("  Chat id: ").strip()
    if not token or not chat:
        return 0
    try:
        import requests
        r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                          json={"chat_id": chat, "text": "Cautious Giggle is connected ✅"}, timeout=15)
        print(f"  {G}✓ Test message sent{X}" if r.ok else f"  {Y}! Telegram said HTTP {r.status_code}; check the token and chat id{X}")
    except Exception as exc:
        print(f"  {Y}! Couldn't reach Telegram ({type(exc).__name__}){X}")
    s = open(cfg, encoding="utf-8").read()
    s = re.sub(r"(\ntelegram:\n  token: ).*", lambda m: m.group(1) + f'"{token}"', s, count=1)
    s = re.sub(r"(\ntelegram:\n  token: .*\n  chat_id: ).*", lambda m: m.group(1) + f'"{chat}"', s, count=1)
    open(cfg, "w", encoding="utf-8").write(s)
    os.chmod(cfg, 0o600)
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "exness" and len(sys.argv) > 2 and sys.argv[2] in ap.FILES:
        raise SystemExit(exness(sys.argv[2], test="--later" not in sys.argv))
    if cmd == "test" and len(sys.argv) > 2 and sys.argv[2] in ap.FILES:
        print(f"\n{B}Exness {sys.argv[2].upper()} account{X}")
        raise SystemExit(show_check(sys.argv[2]))
    if cmd in ("jev", "config", "telegram"):
        raise SystemExit(globals()[cmd]())
    print(__doc__)
    raise SystemExit(2)
