"""Session config — read by all components at startup."""
import json
from pathlib import Path

SESSION_PATH = Path(__file__).resolve().parent / "session.json"


def load() -> dict | None:
    """Load current session. Returns None if no session exists."""
    try:
        if SESSION_PATH.exists():
            return json.loads(SESSION_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        pass
    return None
