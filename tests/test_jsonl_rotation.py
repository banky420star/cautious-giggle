"""
Tests for _rotate_jsonl_if_needed() and _append_jsonl() in both
Python.Server_AGI and tools.project_status_ui.
"""

import json
import os

import pytest

import Python.Server_AGI as server_mod
from Python.Server_AGI import _append_jsonl, _rotate_jsonl_if_needed

import tools.project_status_ui as ui_mod
from tools.project_status_ui import _rotate_jsonl_if_needed as ui_rotate


# ---------------------------------------------------------------------------
# Server_AGI tests
# ---------------------------------------------------------------------------


def test_no_rotation_when_under_limit(tmp_path, monkeypatch):
    """File smaller than the limit must not be rotated."""
    monkeypatch.setattr(server_mod, "_JSONL_MAX_BYTES", 50)
    path = str(tmp_path / "data.jsonl")
    # Write 10 bytes — well under the 50-byte limit.
    with open(path, "w", encoding="utf-8") as f:
        f.write("x" * 10)

    _rotate_jsonl_if_needed(path)

    assert os.path.exists(path), "Original file must still exist after no-op rotation"
    assert not os.path.exists(path + ".1"), ".1 backup must NOT exist when file is under limit"


def test_rotation_when_at_limit(tmp_path, monkeypatch):
    """File at exactly the limit must be rotated to path.1."""
    monkeypatch.setattr(server_mod, "_JSONL_MAX_BYTES", 50)
    path = str(tmp_path / "data.jsonl")
    # Write exactly 50 bytes — at the limit (>= triggers rotation).
    with open(path, "w", encoding="utf-8") as f:
        f.write("x" * 50)

    _rotate_jsonl_if_needed(path)

    assert not os.path.exists(path), "Original file must be gone after rotation"
    assert os.path.exists(path + ".1"), "Backup path.1 must exist after rotation"


def test_rotation_removes_existing_backup(tmp_path, monkeypatch):
    """If path.1 already exists it must be overwritten by the new rotation."""
    monkeypatch.setattr(server_mod, "_JSONL_MAX_BYTES", 50)
    path = str(tmp_path / "data.jsonl")
    backup = path + ".1"

    with open(path, "w", encoding="utf-8") as f:
        f.write("x" * 51)
    with open(backup, "w", encoding="utf-8") as f:
        f.write("old backup content")

    _rotate_jsonl_if_needed(path)

    assert not os.path.exists(path), "Original must be gone after rotation"
    assert os.path.exists(backup), "Backup must exist after rotation"
    # The new backup is the former original, so it must not contain old content.
    with open(backup, "r", encoding="utf-8") as f:
        content = f.read()
    assert content == "x" * 51, "Backup must contain the rotated original content"


def test_no_exception_when_file_missing(tmp_path):
    """Calling _rotate_jsonl_if_needed on a nonexistent path must not raise."""
    missing = str(tmp_path / "nonexistent.jsonl")
    # Should complete silently.
    _rotate_jsonl_if_needed(missing)


def test_append_jsonl_rotates_large_file(tmp_path, monkeypatch):
    """_append_jsonl must trigger rotation before appending when file is at the limit."""
    monkeypatch.setattr(server_mod, "_JSONL_MAX_BYTES", 50)
    path = str(tmp_path / "data.jsonl")
    # File is exactly at the limit — will be rotated on the next append.
    with open(path, "w", encoding="utf-8") as f:
        f.write("x" * 50)

    _append_jsonl(path, {"x": 1})

    assert os.path.exists(path + ".1"), "Rotated backup must exist after append on at-limit file"
    assert os.path.exists(path), "New file must exist after append"
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln for ln in f.read().splitlines() if ln.strip()]
    assert len(lines) == 1, "New file must contain exactly the appended line"
    row = json.loads(lines[0])
    assert row == {"x": 1}


# ---------------------------------------------------------------------------
# project_status_ui tests
# ---------------------------------------------------------------------------


def test_ui_rotation_when_at_limit(tmp_path, monkeypatch):
    """ui._rotate_jsonl_if_needed must rotate a file that is at the size limit."""
    monkeypatch.setattr(ui_mod, "_JSONL_MAX_BYTES", 50)
    path = str(tmp_path / "ui_data.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        f.write("y" * 50)

    ui_rotate(path)

    assert not os.path.exists(path), "Original must be gone after ui rotation"
    assert os.path.exists(path + ".1"), "Backup path.1 must exist after ui rotation"


def test_ui_no_rotation_when_under_limit(tmp_path, monkeypatch):
    """ui._rotate_jsonl_if_needed must leave a small file untouched."""
    monkeypatch.setattr(ui_mod, "_JSONL_MAX_BYTES", 50)
    path = str(tmp_path / "ui_data.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        f.write("y" * 10)

    ui_rotate(path)

    assert os.path.exists(path), "Original must remain when under limit"
    assert not os.path.exists(path + ".1"), ".1 must not be created when under limit"
