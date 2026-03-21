import os
import re

import pytest

from Python.model_registry import ModelRegistry


@pytest.fixture
def reg(tmp_path, monkeypatch):
    """
    Return a ModelRegistry rooted in tmp_path so that no real registry files
    are touched.  _load_registry_config is patched to avoid touching the real
    project config.
    """
    registry_root = str(tmp_path / "registry")

    # Prevent load_project_config from being called (it may not find config.yaml
    # in the test environment).
    monkeypatch.setattr(
        "Python.model_registry.ModelRegistry._load_registry_config",
        lambda self: {},
    )

    return ModelRegistry(root=registry_root)


# ---------------------------------------------------------------------------
# 1. Corrupt JSON in active.json → normalized empty state, no exception
# ---------------------------------------------------------------------------

def test_read_active_handles_corrupt_json(reg):
    """Writing garbage to active.json must not raise; _read_active() should
    return a normalized empty state with the required top-level keys."""
    with open(reg.active_path, "w", encoding="utf-8") as f:
        f.write("NOT VALID JSON{{{{")

    result = reg._read_active()

    assert isinstance(result, dict)
    assert "champion" in result
    assert "canary" in result
    assert "symbols" in result


# ---------------------------------------------------------------------------
# 2. Empty active.json → normalized empty state, no exception
# ---------------------------------------------------------------------------

def test_read_active_handles_empty_file(reg):
    """An empty active.json must not raise; _read_active() should return a
    normalized empty state."""
    with open(reg.active_path, "w", encoding="utf-8") as f:
        f.write("")

    result = reg._read_active()

    assert isinstance(result, dict)
    assert "champion" in result
    assert "canary" in result
    assert "symbols" in result


# ---------------------------------------------------------------------------
# 3. Missing active.json → normalized empty state, no exception
# ---------------------------------------------------------------------------

def test_read_active_handles_missing_file(reg):
    """Deleting active.json before calling _read_active() must not raise;
    the method should return a normalized empty state."""
    os.remove(reg.active_path)
    assert not os.path.exists(reg.active_path)

    result = reg._read_active()

    assert isinstance(result, dict)
    assert "champion" in result
    assert "canary" in result
    assert "symbols" in result


# ---------------------------------------------------------------------------
# 4. _timestamp_version() returns the expected format
# ---------------------------------------------------------------------------

def test_timestamp_version_format(reg):
    """_timestamp_version() should return a string matching YYYYMMDD_HHMMSS."""
    version = reg._timestamp_version()

    assert re.match(r"^\d{8}_\d{6}$", version), (
        f"_timestamp_version() returned unexpected format: {version!r}"
    )
