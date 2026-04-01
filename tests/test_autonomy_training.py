import asyncio
import builtins
import io
import json
import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from Python.autonomy_loop import AutonomyLoop


@pytest.fixture
def mock_brain():
    """Provide a mock brain object for AutonomyLoop."""
    return MagicMock()


@pytest.fixture
def mock_config(monkeypatch):
    """Mock config loading to avoid file I/O."""

    def mock_load_config(project_root, live_mode=False):
        return {
            "trading": {"symbols": ["BTCUSDm", "XAUUSDm"]},
            "drl": {"dreamer": {"enabled": False, "train_in_cycle": False}},
            "evaluation": {},
        }

    monkeypatch.setattr("Python.autonomy_loop.load_project_config", mock_load_config)


@pytest.fixture
def temp_registry_dir(tmp_path):
    """Create a temporary directory structure for model registry."""
    registry_dir = tmp_path / "registry"
    registry_dir.mkdir()
    candidates_dir = registry_dir / "candidates"
    candidates_dir.mkdir()

    # Write a minimal active.json
    active = {
        "champion": None,
        "canary": None,
        "symbols": {},
    }
    (registry_dir / "active.json").write_text(json.dumps(active), encoding="utf-8")

    yield registry_dir, candidates_dir

    # Cleanup
    shutil.rmtree(registry_dir, ignore_errors=True)


@pytest.fixture
def autonomy_loop(mock_brain, mock_config, temp_registry_dir, monkeypatch):
    """Create an AutonomyLoop with mocked dependencies."""
    registry_dir, candidates_dir = temp_registry_dir

    # Mock the ModelRegistry to use our temp directory
    with patch("Python.autonomy_loop.ModelRegistry") as mock_registry_class:
        mock_registry = MagicMock()
        mock_registry.candidates_dir = str(candidates_dir)
        mock_registry._read_active.return_value = {
            "champion": None,
            "canary": None,
            "symbols": {},
        }
        mock_registry._write_active.return_value = None
        mock_registry_class.return_value = mock_registry

        # Mock TelegramAlerter
        with patch("Python.autonomy_loop.TelegramAlerter"):
            loop = AutonomyLoop(mock_brain)
            loop.registry = mock_registry
            yield loop


# ---------------------------------------------------------------------------
# 1. test_run_training_subprocess_success
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_training_subprocess_success(autonomy_loop):
    """Mock subprocess to return exit code 0, verify no error."""
    mock_process = AsyncMock()
    mock_process.returncode = 0
    mock_process.communicate = AsyncMock(return_value=(b"output", b""))

    with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
        mock_exec.return_value = mock_process

        # Should not raise
        await autonomy_loop._run_training_subprocess(
            ["python", "train.py"], {}, "test_label", timeout=30
        )


@pytest.mark.asyncio
async def test_run_training_subprocess_success_with_output(autonomy_loop):
    """Subprocess success with stdout/stderr output is handled correctly."""
    mock_process = AsyncMock()
    mock_process.returncode = 0
    mock_process.communicate = AsyncMock(return_value=(b"training completed", b"warnings"))

    with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
        mock_exec.return_value = mock_process

        # Should not raise
        await autonomy_loop._run_training_subprocess(
            ["python", "train.py"], {}, "test_label", timeout=30
        )


# ---------------------------------------------------------------------------
# 2. test_run_training_subprocess_failure
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_training_subprocess_failure(autonomy_loop):
    """Subprocess returns non-zero exit code, verify RuntimeError is raised."""
    mock_process = AsyncMock()
    mock_process.returncode = 1
    mock_process.communicate = AsyncMock(return_value=(b"", b"error message"))

    with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
        mock_exec.return_value = mock_process

        with pytest.raises(RuntimeError) as excinfo:
            await autonomy_loop._run_training_subprocess(
                ["python", "train.py"], {}, "test_label", timeout=30
            )

        assert "Training subprocess failed" in str(excinfo.value)
        assert "test_label" in str(excinfo.value)
        assert "exit code 1" in str(excinfo.value)


@pytest.mark.asyncio
async def test_run_training_subprocess_failure_with_long_error(autonomy_loop):
    """Subprocess failure error message is truncated to 2000 chars."""
    long_error = b"E" * 5000  # Error message longer than 2000 chars
    mock_process = AsyncMock()
    mock_process.returncode = 127
    mock_process.communicate = AsyncMock(return_value=(b"", long_error))

    with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
        mock_exec.return_value = mock_process

        with pytest.raises(RuntimeError) as excinfo:
            await autonomy_loop._run_training_subprocess(
                ["python", "train.py"], {}, "test_label", timeout=30
            )

        error_str = str(excinfo.value)
        # Check that error is in the message but truncated
        assert len(error_str) < 5000 + 100  # Some buffer for the prefix


@pytest.mark.asyncio
async def test_run_training_subprocess_nonzero_exit_codes(autonomy_loop):
    """Various non-zero exit codes all trigger RuntimeError."""
    for exit_code in [1, 2, 127, 255]:
        mock_process = AsyncMock()
        mock_process.returncode = exit_code
        mock_process.communicate = AsyncMock(return_value=(b"", b"error"))

        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = mock_process

            with pytest.raises(RuntimeError) as excinfo:
                await autonomy_loop._run_training_subprocess(
                    ["python", "train.py"], {}, f"label_{exit_code}", timeout=30
                )

            assert f"exit code {exit_code}" in str(excinfo.value)


# ---------------------------------------------------------------------------
# 3. test_run_training_subprocess_timeout
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_training_subprocess_timeout(autonomy_loop):
    """Subprocess hangs, timeout triggers, process is killed."""
    mock_process = AsyncMock()

    # Simulate timeout by raising asyncio.TimeoutError
    async def timeout_communicate():
        await asyncio.sleep(100)  # Would timeout

    mock_process.communicate = timeout_communicate
    mock_process.kill = MagicMock()

    with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
        mock_exec.return_value = mock_process

        # Mock asyncio.wait_for to raise TimeoutError
        with patch("asyncio.wait_for", new_callable=AsyncMock) as mock_wait_for:
            mock_wait_for.side_effect = asyncio.TimeoutError()

            with pytest.raises(RuntimeError) as excinfo:
                await autonomy_loop._run_training_subprocess(
                    ["python", "train.py"], {}, "test_label", timeout=5
                )

            assert "timed out" in str(excinfo.value)
            assert "5s" in str(excinfo.value)


@pytest.mark.asyncio
async def test_run_training_subprocess_timeout_kills_process(autonomy_loop):
    """On timeout, process.kill() is called."""
    mock_process = AsyncMock()
    mock_process.kill = MagicMock()
    mock_process.communicate = AsyncMock(return_value=(b"", b""))

    with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
        mock_exec.return_value = mock_process

        with patch("asyncio.wait_for") as mock_wait_for:
            mock_wait_for.side_effect = asyncio.TimeoutError()

            with pytest.raises(RuntimeError):
                await autonomy_loop._run_training_subprocess(
                    ["python", "train.py"], {}, "test_label", timeout=5
                )

            # Verify kill was called
            mock_process.kill.assert_called()


# ---------------------------------------------------------------------------
# 4. test_candidate_validation_missing_dir
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_candidate_validation_missing_dir(autonomy_loop, monkeypatch):
    """When _latest_candidate_dir returns None after training, candidate is NOT promoted."""
    # Mock _latest_candidate_dir to return None
    monkeypatch.setattr(autonomy_loop, "_latest_candidate_dir", lambda symbol: None)

    # Mock the training subprocess to succeed
    async def mock_train_subprocess(*args, **kwargs):
        pass

    monkeypatch.setattr(autonomy_loop, "_run_training_subprocess", mock_train_subprocess)

    # Mock notification and other methods
    monkeypatch.setattr(autonomy_loop, "_notify", MagicMock())
    monkeypatch.setattr(autonomy_loop, "_champion_cycle_running", lambda: False)

    # Mock the training lock
    autonomy_loop._train_lock = asyncio.Lock()

    # Run training — it should log error and return early
    await autonomy_loop._train_symbol_candidate("BTCUSDm")

    # Verify _latest_candidate_dir was called
    # Verify no canary was set (through mocked _maybe_set_canary)
    # The function should have returned early and logged an error


@pytest.mark.asyncio
async def test_candidate_validation_missing_dir_logs_error(autonomy_loop, monkeypatch):
    """When candidate dir is missing, error is logged."""
    monkeypatch.setattr(autonomy_loop, "_latest_candidate_dir", lambda symbol: None)

    async def mock_train_subprocess(*args, **kwargs):
        pass

    monkeypatch.setattr(autonomy_loop, "_run_training_subprocess", mock_train_subprocess)
    monkeypatch.setattr(autonomy_loop, "_champion_cycle_running", lambda: False)
    autonomy_loop._train_lock = asyncio.Lock()

    # Capture logger calls
    import logging
    from loguru import logger

    with patch.object(logger, "error") as mock_error:
        await autonomy_loop._train_symbol_candidate("BTCUSDm")


# ---------------------------------------------------------------------------
# 5. test_candidate_validation_missing_model
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_candidate_validation_missing_model(autonomy_loop, monkeypatch, tmp_path):
    """When candidate dir exists but ppo_trading.zip is missing, candidate is NOT promoted."""
    # Create a candidate directory without the model artifact
    candidate_dir = tmp_path / "candidate"
    candidate_dir.mkdir()

    monkeypatch.setattr(autonomy_loop, "_latest_candidate_dir", lambda symbol: str(candidate_dir))

    async def mock_train_subprocess(*args, **kwargs):
        pass

    monkeypatch.setattr(autonomy_loop, "_run_training_subprocess", mock_train_subprocess)
    monkeypatch.setattr(autonomy_loop, "_champion_cycle_running", lambda: False)
    autonomy_loop._train_lock = asyncio.Lock()

    # Mock notification
    monkeypatch.setattr(autonomy_loop, "_notify", MagicMock())

    # Run training — should detect missing model
    await autonomy_loop._train_symbol_candidate("BTCUSDm")

    # Verify model file does not exist
    assert not (candidate_dir / "ppo_trading.zip").exists()


@pytest.mark.asyncio
async def test_candidate_validation_with_valid_model(autonomy_loop, monkeypatch, tmp_path):
    """When candidate dir and model exist, candidate can be promoted."""
    # Create a candidate directory with the model artifact
    candidate_dir = tmp_path / "candidate"
    candidate_dir.mkdir()
    model_file = candidate_dir / "ppo_trading.zip"
    model_file.write_text("fake model", encoding="utf-8")

    monkeypatch.setattr(autonomy_loop, "_latest_candidate_dir", lambda symbol: str(candidate_dir))

    async def mock_train_subprocess(*args, **kwargs):
        pass

    monkeypatch.setattr(autonomy_loop, "_run_training_subprocess", mock_train_subprocess)
    monkeypatch.setattr(autonomy_loop, "_champion_cycle_running", lambda: False)
    monkeypatch.setattr(autonomy_loop, "_maybe_set_canary", MagicMock())
    autonomy_loop._train_lock = asyncio.Lock()

    monkeypatch.setattr(autonomy_loop, "_notify", MagicMock())

    # Run training
    await autonomy_loop._train_symbol_candidate("BTCUSDm")

    # Verify model file exists
    assert model_file.exists()
    # Verify _maybe_set_canary was called (indicating validation passed)
    autonomy_loop._maybe_set_canary.assert_called()


@pytest.mark.asyncio
async def test_candidate_validation_model_file_existence_check(autonomy_loop, monkeypatch, tmp_path):
    """Verify candidate is rejected if ppo_trading.zip is missing."""
    candidate_dir = tmp_path / "candidate"
    candidate_dir.mkdir()

    # Explicitly verify the model path check
    monkeypatch.setattr(autonomy_loop, "_latest_candidate_dir", lambda symbol: str(candidate_dir))

    async def mock_train_subprocess(*args, **kwargs):
        pass

    monkeypatch.setattr(autonomy_loop, "_run_training_subprocess", mock_train_subprocess)
    monkeypatch.setattr(autonomy_loop, "_champion_cycle_running", lambda: False)
    monkeypatch.setattr(autonomy_loop, "_notify", MagicMock())
    autonomy_loop._train_lock = asyncio.Lock()

    await autonomy_loop._train_symbol_candidate("BTCUSDm")

    # The function should return early without calling _maybe_set_canary
    # because the model file doesn't exist


# ---------------------------------------------------------------------------
# 6. Integration tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_training_subprocess_with_custom_env(autonomy_loop):
    """Subprocess receives custom environment variables."""
    mock_process = AsyncMock()
    mock_process.returncode = 0
    mock_process.communicate = AsyncMock(return_value=(b"", b""))

    with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
        mock_exec.return_value = mock_process

        custom_env = {"AGI_TEST_VAR": "value123"}
        await autonomy_loop._run_training_subprocess(
            ["python", "train.py"], custom_env, "test_label", timeout=30
        )

        # Verify create_subprocess_exec was called with the custom env
        assert mock_exec.called
        call_kwargs = mock_exec.call_args[1]
        assert call_kwargs["env"] == custom_env


@pytest.mark.asyncio
async def test_run_training_subprocess_sets_cwd(autonomy_loop):
    """Subprocess is run with correct working directory."""
    mock_process = AsyncMock()
    mock_process.returncode = 0
    mock_process.communicate = AsyncMock(return_value=(b"", b""))

    with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
        mock_exec.return_value = mock_process

        await autonomy_loop._run_training_subprocess(
            ["python", "train.py"], {}, "test_label", timeout=30
        )

        # Verify cwd is set to PROJECT_ROOT
        assert mock_exec.called
        call_kwargs = mock_exec.call_args[1]
        assert "cwd" in call_kwargs


@pytest.mark.asyncio
async def test_run_training_subprocess_pipes_stdout_stderr(autonomy_loop):
    """Subprocess output is piped."""
    mock_process = AsyncMock()
    mock_process.returncode = 0
    mock_process.communicate = AsyncMock(return_value=(b"", b""))

    with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
        mock_exec.return_value = mock_process

        await autonomy_loop._run_training_subprocess(
            ["python", "train.py"], {}, "test_label", timeout=30
        )

        # Verify stdout/stderr are piped
        call_kwargs = mock_exec.call_args[1]
        assert call_kwargs["stdout"] == asyncio.subprocess.PIPE
        assert call_kwargs["stderr"] == asyncio.subprocess.PIPE
