import os
import stat

import tools.account_profiles as ap

SEED = "11" * 64  # 128 hex chars, same shape as an Exness key


def _isolate(tmp_path, monkeypatch):
    monkeypatch.setattr(ap, "DIR", str(tmp_path))
    monkeypatch.setattr(ap, "FILES", {"demo": str(tmp_path / "env"), "live": str(tmp_path / "live.env")})
    monkeypatch.setattr(ap, "LIVE_KEY_FILE", str(tmp_path / "live_api_private_key"))
    (tmp_path / "env").write_text("export EXNESS_BASE_URL=api.example.test\n")


def test_save_live_writes_private_files_and_returns_no_values(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    assert ap.live_form()["gateway_hint"] == "api.example.test"
    r = ap.save_live("key-123", "87654321", "api.example.test", SEED)
    assert r["ok"] and r["live"]["ready"]
    assert "key-123" not in str(r) and SEED not in str(r)
    for f in ("live.env", "live_api_private_key"):
        assert stat.S_IMODE(os.stat(tmp_path / f).st_mode) == 0o600
    env = ap.parse_env(str(tmp_path / "live.env"))
    assert env["EXNESS_API_KEY"] == "key-123" and env["EXNESS_ACCOUNT_ID"] == "87654321"
    # blanks keep what is saved
    assert ap.save_live("", "", "", "")["ok"]
    assert ap.parse_env(str(tmp_path / "live.env"))["EXNESS_API_KEY"] == "key-123"


def test_save_live_rejects_bad_input_and_demo_key(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    r = ap.save_live("bad key; rm", "1", "not a host!", "zz")
    assert not r["ok"] and set(r["errors"]) == {"api_key", "account_id", "gateway", "private_key"}
    assert not (tmp_path / "live.env").exists()
    (tmp_path / "demo_key").write_text(SEED)
    (tmp_path / "env").write_text(f"export EXNESS_PRIVATE_KEY_FILE={tmp_path / 'demo_key'}\nexport EXNESS_BASE_URL=h.test\n")
    r = ap.save_live("key-1", "12345", "h.test", SEED)
    assert not r["ok"] and "DEMO" in r["errors"]["private_key"]
