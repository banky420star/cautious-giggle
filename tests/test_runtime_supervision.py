import os
from types import SimpleNamespace

import pytest

import desktop_app
from tools import production_orchestrator
from tools import watchdog


def test_orchestrator_uses_lock_file_for_standard_server(monkeypatch, tmp_path):
    lock_path = tmp_path / "server_agi.lock"
    lock_path.write_text(str(os.getpid()), encoding="utf-8")

    monkeypatch.setitem(
        production_orchestrator.SERVICE_LOCK_FILES,
        production_orchestrator.SVC_SERVER_AGI,
        str(lock_path),
    )
    monkeypatch.setattr(production_orchestrator, "_process_running_wmic", lambda _frag: False)

    assert production_orchestrator._is_service_running(production_orchestrator.SVC_SERVER_AGI) is True


def test_orchestrator_uses_lock_file_for_hft_server(monkeypatch, tmp_path):
    lock_path = tmp_path / "server_agi_hft.lock"
    lock_path.write_text(str(os.getpid()), encoding="utf-8")

    monkeypatch.setitem(
        production_orchestrator.SERVICE_LOCK_FILES,
        production_orchestrator.SVC_HFT,
        str(lock_path),
    )
    monkeypatch.setattr(production_orchestrator, "_process_running_wmic", lambda _frag: False)

    assert production_orchestrator._is_service_running(production_orchestrator.SVC_HFT) is True


def test_watchdog_process_detected_externally_prefers_lock_file(tmp_path):
    lock_path = tmp_path / "server_agi.lock"
    lock_path.write_text(str(os.getpid()), encoding="utf-8")
    spec = watchdog.ProcessSpec(
        name="server_agi",
        detection_tokens=[],
        cmd=["python"],
        cwd=".",
        lock_file=str(lock_path),
    )

    assert watchdog._process_detected_externally(spec, "", None) is True
    assert watchdog._process_detected_externally(spec, "", SimpleNamespace(pid=os.getpid())) is False


def test_watchdog_validate_live_env_loads_env_and_validates_hft(monkeypatch, tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text("AGI_TOKEN=watchdog-token\n", encoding="utf-8")
    hft_path = tmp_path / "config_hft.yaml"
    hft_path.write_text("mt5:\n  login: ENV:MT5_LOGIN\n", encoding="utf-8")

    calls: list[tuple[str, bool, str | None]] = []

    def fake_load_project_config(project_root: str, live_mode: bool = False):
        calls.append((project_root, live_mode, os.environ.get("AGI_CONFIG")))
        return {}

    monkeypatch.delenv("AGI_TOKEN", raising=False)
    monkeypatch.delenv("AGI_CONFIG", raising=False)
    monkeypatch.setattr(watchdog, "ENV_PATH", str(env_path))
    monkeypatch.setattr(watchdog, "PROJECT_ROOT", str(tmp_path))
    monkeypatch.setattr(watchdog, "HFT_CONFIG_PATH", str(hft_path))
    monkeypatch.setattr(watchdog, "load_project_config", fake_load_project_config)

    watchdog._validate_live_env()

    assert os.environ["AGI_TOKEN"] == "watchdog-token"
    assert calls == [
        (str(tmp_path), True, None),
        (str(tmp_path), True, str(hft_path)),
    ]


def test_desktop_launcher_uses_server_lock_before_launch(monkeypatch, tmp_path):
    lock_path = tmp_path / "server_agi.lock"
    lock_path.write_text(str(os.getpid()), encoding="utf-8")

    monkeypatch.setattr(desktop_app, "SERVER_LOCK_PATH", lock_path)
    monkeypatch.setattr(
        desktop_app,
        "_launch",
        lambda *_args, **_kwargs: pytest.fail("launcher should not spawn a duplicate standard server"),
    )

    assert desktop_app.start_server_agi() is None


def test_desktop_launcher_skips_invalid_hft_lane(monkeypatch, tmp_path):
    hft_path = tmp_path / "config_hft.yaml"
    hft_path.write_text("mt5:\n  login: ENV:MT5_LOGIN\n", encoding="utf-8")

    monkeypatch.setattr(desktop_app, "HFT_CONFIG_PATH", hft_path)
    monkeypatch.setattr(desktop_app, "_validate_live_config", lambda _path: "broken hft config")
    monkeypatch.setattr(
        desktop_app,
        "_launch",
        lambda *_args, **_kwargs: pytest.fail("launcher should not spawn HFT when its config is invalid"),
    )

    assert desktop_app.start_hft_server() is None
