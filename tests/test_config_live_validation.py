import shutil
import tempfile
from pathlib import Path

import pytest

from Python.config_utils import load_project_config


@pytest.fixture
def temp_config_dir():
    """Create a temporary directory for config files."""
    tmpdir = Path(tempfile.mkdtemp())
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# 1. test_live_mode_blocks_without_mt5_login
# ---------------------------------------------------------------------------


def test_live_mode_blocks_without_mt5_login(temp_config_dir, monkeypatch):
    """With no MT5_LOGIN env var and no config login, verify RuntimeError."""
    # Clear MT5 env vars
    monkeypatch.delenv("MT5_LOGIN", raising=False)
    monkeypatch.delenv("MT5_PASSWORD", raising=False)
    monkeypatch.delenv("MT5_SERVER", raising=False)

    # Create config without MT5 login
    config_path = temp_config_dir / "config.yaml"
    config_path.write_text(
        """
telegram:
  token: "test_token"
  chat_id: "test_id"
""",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError) as excinfo:
        load_project_config(str(temp_config_dir), live_mode=True)

    assert "mt5 login" in str(excinfo.value).lower()
    assert "not configured" in str(excinfo.value).lower()


def test_live_mode_blocks_without_mt5_login_env_var(temp_config_dir, monkeypatch):
    """MT5_LOGIN env var must be set or have value in config."""
    monkeypatch.delenv("MT5_LOGIN", raising=False)
    monkeypatch.delenv("MT5_PASSWORD", raising=False)
    monkeypatch.delenv("MT5_SERVER", raising=False)

    config_path = temp_config_dir / "config.yaml"
    config_path.write_text("", encoding="utf-8")

    with pytest.raises(RuntimeError) as excinfo:
        load_project_config(str(temp_config_dir), live_mode=True)

    assert "mt5 login" in str(excinfo.value).lower()


def test_live_mode_blocks_with_login_zero(temp_config_dir, monkeypatch):
    """MT5_LOGIN='0' is treated as unconfigured (broker sentinel value)."""
    monkeypatch.setenv("MT5_LOGIN", "0")
    monkeypatch.delenv("MT5_PASSWORD", raising=False)
    monkeypatch.delenv("MT5_SERVER", raising=False)

    config_path = temp_config_dir / "config.yaml"
    config_path.write_text(
        """
telegram:
  token: "test_token"
  chat_id: "test_id"
""",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError) as excinfo:
        load_project_config(str(temp_config_dir), live_mode=True)

    assert "mt5 login" in str(excinfo.value).lower()


# ---------------------------------------------------------------------------
# 2. test_live_mode_blocks_without_mt5_password
# ---------------------------------------------------------------------------


def test_live_mode_blocks_without_mt5_password(temp_config_dir, monkeypatch):
    """With valid login but no password, verify RuntimeError."""
    monkeypatch.setenv("MT5_LOGIN", "123456")
    monkeypatch.delenv("MT5_PASSWORD", raising=False)
    monkeypatch.delenv("MT5_SERVER", raising=False)

    config_path = temp_config_dir / "config.yaml"
    config_path.write_text(
        """
telegram:
  token: "test_token"
  chat_id: "test_id"
""",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError) as excinfo:
        load_project_config(str(temp_config_dir), live_mode=True)

    assert "mt5 password" in str(excinfo.value).lower()
    assert "not configured" in str(excinfo.value).lower()


def test_live_mode_blocks_with_empty_mt5_password(temp_config_dir, monkeypatch):
    """Empty MT5_PASSWORD is treated as unconfigured."""
    monkeypatch.setenv("MT5_LOGIN", "123456")
    monkeypatch.setenv("MT5_PASSWORD", "")
    monkeypatch.delenv("MT5_SERVER", raising=False)

    config_path = temp_config_dir / "config.yaml"
    config_path.write_text("", encoding="utf-8")

    with pytest.raises(RuntimeError) as excinfo:
        load_project_config(str(temp_config_dir), live_mode=True)

    assert "mt5 password" in str(excinfo.value).lower()


# ---------------------------------------------------------------------------
# 3. test_live_mode_blocks_without_mt5_server
# ---------------------------------------------------------------------------


def test_live_mode_blocks_without_mt5_server(temp_config_dir, monkeypatch):
    """With valid login+password but no server, verify RuntimeError."""
    monkeypatch.setenv("MT5_LOGIN", "123456")
    monkeypatch.setenv("MT5_PASSWORD", "password")
    monkeypatch.delenv("MT5_SERVER", raising=False)

    config_path = temp_config_dir / "config.yaml"
    config_path.write_text(
        """
telegram:
  token: "test_token"
  chat_id: "test_id"
""",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError) as excinfo:
        load_project_config(str(temp_config_dir), live_mode=True)

    assert "mt5 server" in str(excinfo.value).lower()
    assert "not configured" in str(excinfo.value).lower()


def test_live_mode_blocks_with_empty_mt5_server(temp_config_dir, monkeypatch):
    """Empty MT5_SERVER is treated as unconfigured."""
    monkeypatch.setenv("MT5_LOGIN", "123456")
    monkeypatch.setenv("MT5_PASSWORD", "password")
    monkeypatch.setenv("MT5_SERVER", "")

    config_path = temp_config_dir / "config.yaml"
    config_path.write_text("", encoding="utf-8")

    with pytest.raises(RuntimeError) as excinfo:
        load_project_config(str(temp_config_dir), live_mode=True)

    assert "mt5 server" in str(excinfo.value).lower()


# ---------------------------------------------------------------------------
# 4. test_live_mode_allows_env_credentials
# ---------------------------------------------------------------------------


def test_live_mode_allows_env_credentials(temp_config_dir, monkeypatch):
    """With all MT5_* env vars set, verify no error."""
    monkeypatch.setenv("MT5_LOGIN", "123456")
    monkeypatch.setenv("MT5_PASSWORD", "mypassword")
    monkeypatch.setenv("MT5_SERVER", "MyBroker-Demo")

    config_path = temp_config_dir / "config.yaml"
    config_path.write_text(
        """
telegram:
  token: "test_token"
  chat_id: "test_id"
""",
        encoding="utf-8",
    )

    # Should not raise
    cfg = load_project_config(str(temp_config_dir), live_mode=True)
    assert cfg is not None


def test_live_mode_allows_env_credentials_with_empty_config_mt5(temp_config_dir, monkeypatch):
    """Env vars take precedence even if config has no MT5 section."""
    monkeypatch.setenv("MT5_LOGIN", "999999")
    monkeypatch.setenv("MT5_PASSWORD", "env_password")
    monkeypatch.setenv("MT5_SERVER", "EnvBroker")

    config_path = temp_config_dir / "config.yaml"
    config_path.write_text(
        """
telegram:
  token: "test_token"
  chat_id: "test_id"
""",
        encoding="utf-8",
    )

    cfg = load_project_config(str(temp_config_dir), live_mode=True)
    assert cfg is not None


# ---------------------------------------------------------------------------
# 5. test_live_mode_allows_config_credentials
# ---------------------------------------------------------------------------


def test_live_mode_allows_config_credentials(temp_config_dir, monkeypatch):
    """With credentials in config.yaml (not ENV: refs), verify no error."""
    # Clear env vars
    monkeypatch.delenv("MT5_LOGIN", raising=False)
    monkeypatch.delenv("MT5_PASSWORD", raising=False)
    monkeypatch.delenv("MT5_SERVER", raising=False)

    config_path = temp_config_dir / "config.yaml"
    config_path.write_text(
        """
mt5:
  login: "555555"
  password: "configpassword"
  server: "ConfigBroker"
telegram:
  token: "test_token"
  chat_id: "test_id"
""",
        encoding="utf-8",
    )

    cfg = load_project_config(str(temp_config_dir), live_mode=True)
    assert cfg is not None


def test_live_mode_allows_config_credentials_with_env_refs(temp_config_dir, monkeypatch):
    """Config can use ENV:VARIABLE refs if the env vars are set."""
    monkeypatch.setenv("MT5_LOGIN", "777777")
    monkeypatch.setenv("MT5_PASSWORD", "envpass")
    monkeypatch.setenv("MT5_SERVER", "EnvServer")

    config_path = temp_config_dir / "config.yaml"
    config_path.write_text(
        """
mt5:
  login: "ENV:MT5_LOGIN"
  password: "ENV:MT5_PASSWORD"
  server: "ENV:MT5_SERVER"
telegram:
  token: "test_token"
  chat_id: "test_id"
""",
        encoding="utf-8",
    )

    cfg = load_project_config(str(temp_config_dir), live_mode=True)
    assert cfg is not None


def test_live_mode_blocks_config_env_refs_without_env_vars(temp_config_dir, monkeypatch):
    """Config uses ENV:VARIABLE refs but env vars are not set."""
    monkeypatch.delenv("MT5_LOGIN", raising=False)
    monkeypatch.delenv("MT5_PASSWORD", raising=False)
    monkeypatch.delenv("MT5_SERVER", raising=False)

    config_path = temp_config_dir / "config.yaml"
    config_path.write_text(
        """
mt5:
  login: "ENV:MT5_LOGIN"
  password: "ENV:MT5_PASSWORD"
  server: "ENV:MT5_SERVER"
telegram:
  token: "test_token"
  chat_id: "test_id"
""",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError) as excinfo:
        load_project_config(str(temp_config_dir), live_mode=True)

    # Should fail because env vars are not set
    assert "MT5" in str(excinfo.value)


def test_live_mode_allows_case_insensitive_env_refs_and_whitespace(temp_config_dir, monkeypatch):
    """ENV refs should work with surrounding whitespace and case-insensitive prefix."""
    monkeypatch.setenv("MT5_LOGIN", "246810")
    monkeypatch.setenv("MT5_PASSWORD", "whitespace-ok")
    monkeypatch.setenv("MT5_SERVER", "CaseBroker")
    monkeypatch.setenv("TELEGRAM_TOKEN", "telegram-token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "telegram-chat")

    config_path = temp_config_dir / "config.yaml"
    config_path.write_text(
        """
mt5:
  login: "  env:MT5_LOGIN  "
  password: "  ENV:MT5_PASSWORD  "
  server: "  env:MT5_SERVER  "
telegram:
  token: "  env:TELEGRAM_TOKEN  "
  chat_id: "  ENV:TELEGRAM_CHAT_ID  "
""",
        encoding="utf-8",
    )

    cfg = load_project_config(str(temp_config_dir), live_mode=True)
    assert cfg is not None


def test_live_mode_blocks_partial_telegram_config(temp_config_dir, monkeypatch):
    """Telegram config must be complete or omitted in live mode."""
    monkeypatch.setenv("MT5_LOGIN", "123456")
    monkeypatch.setenv("MT5_PASSWORD", "password")
    monkeypatch.setenv("MT5_SERVER", "MyBroker-Demo")
    monkeypatch.delenv("TELEGRAM_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)

    config_path = temp_config_dir / "config.yaml"
    config_path.write_text(
        """
telegram:
  token: "some-token"
  chat_id: ""
""",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError) as excinfo:
        load_project_config(str(temp_config_dir), live_mode=True)

    assert "telegram.token and telegram.chat_id" in str(excinfo.value).lower()


# ---------------------------------------------------------------------------
# 6. test_non_live_mode_skips_mt5_validation
# ---------------------------------------------------------------------------


def test_non_live_mode_skips_mt5_validation(temp_config_dir, monkeypatch):
    """In non-live mode, no MT5 credential errors even with missing creds."""
    # Clear all MT5 env vars
    monkeypatch.delenv("MT5_LOGIN", raising=False)
    monkeypatch.delenv("MT5_PASSWORD", raising=False)
    monkeypatch.delenv("MT5_SERVER", raising=False)

    config_path = temp_config_dir / "config.yaml"
    config_path.write_text(
        """
telegram:
  token: "test_token"
  chat_id: "test_id"
""",
        encoding="utf-8",
    )

    # Should NOT raise even with missing MT5 config
    cfg = load_project_config(str(temp_config_dir), live_mode=False)
    assert cfg is not None


def test_non_live_mode_allows_empty_config(temp_config_dir, monkeypatch):
    """Non-live mode works with completely empty config."""
    monkeypatch.delenv("MT5_LOGIN", raising=False)
    monkeypatch.delenv("MT5_PASSWORD", raising=False)
    monkeypatch.delenv("MT5_SERVER", raising=False)
    monkeypatch.delenv("TELEGRAM_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)

    config_path = temp_config_dir / "config.yaml"
    config_path.write_text("", encoding="utf-8")

    cfg = load_project_config(str(temp_config_dir), live_mode=False)
    assert cfg is not None


# ---------------------------------------------------------------------------
# 7. Edge cases: MT5 config precedence
# ---------------------------------------------------------------------------


def test_env_mt5_overrides_config_mt5(temp_config_dir, monkeypatch):
    """Env vars take precedence over config values."""
    monkeypatch.setenv("MT5_LOGIN", "111111")
    monkeypatch.setenv("MT5_PASSWORD", "env_pass")
    monkeypatch.setenv("MT5_SERVER", "env_server")

    config_path = temp_config_dir / "config.yaml"
    config_path.write_text(
        """
mt5:
  login: "222222"
  password: "config_pass"
  server: "config_server"
telegram:
  token: "test_token"
  chat_id: "test_id"
""",
        encoding="utf-8",
    )

    # Should succeed with env values
    cfg = load_project_config(str(temp_config_dir), live_mode=True)
    assert cfg is not None


def test_mt5_config_whitespace_trimmed(temp_config_dir, monkeypatch):
    """Whitespace in config MT5 values is trimmed."""
    monkeypatch.delenv("MT5_LOGIN", raising=False)
    monkeypatch.delenv("MT5_PASSWORD", raising=False)
    monkeypatch.delenv("MT5_SERVER", raising=False)

    config_path = temp_config_dir / "config.yaml"
    config_path.write_text(
        """
mt5:
  login: "  333333  "
  password: "  my_password  "
  server: "  BrokerServer  "
telegram:
  token: "test_token"
  chat_id: "test_id"
""",
        encoding="utf-8",
    )

    # Should succeed even with whitespace
    cfg = load_project_config(str(temp_config_dir), live_mode=True)
    assert cfg is not None


# ---------------------------------------------------------------------------
# 8. MT5 config with partial values
# ---------------------------------------------------------------------------


def test_live_mode_missing_mt5_section(temp_config_dir, monkeypatch):
    """Config missing 'mt5' section and no env vars."""
    monkeypatch.delenv("MT5_LOGIN", raising=False)
    monkeypatch.delenv("MT5_PASSWORD", raising=False)
    monkeypatch.delenv("MT5_SERVER", raising=False)

    config_path = temp_config_dir / "config.yaml"
    config_path.write_text(
        """
telegram:
  token: "test_token"
  chat_id: "test_id"
""",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError):
        load_project_config(str(temp_config_dir), live_mode=True)


def test_live_mode_partial_config_credentials(temp_config_dir, monkeypatch):
    """Config has login and server but missing password."""
    monkeypatch.delenv("MT5_LOGIN", raising=False)
    monkeypatch.delenv("MT5_PASSWORD", raising=False)
    monkeypatch.delenv("MT5_SERVER", raising=False)

    config_path = temp_config_dir / "config.yaml"
    config_path.write_text(
        """
mt5:
  login: "444444"
  server: "SomeServer"
telegram:
  token: "test_token"
  chat_id: "test_id"
""",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError) as excinfo:
        load_project_config(str(temp_config_dir), live_mode=True)

    assert "mt5 password" in str(excinfo.value).lower()
