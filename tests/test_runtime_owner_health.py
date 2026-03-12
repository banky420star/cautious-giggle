from tools import project_status_ui as ui


def test_runtime_owner_health_accepts_windows_venv_redirector_chain():
    procs = [
        {
            "pid": 100,
            "ppid": 50,
            "name": "python.exe",
            "cmd": r"C:\windows\system32\cautious-giggle\.venv312\Scripts\python.exe -m Python.Server_AGI --live",
        },
        {
            "pid": 101,
            "ppid": 100,
            "name": "python.exe",
            "cmd": r'"C:\Users\Administrator\Desktop\python.exe" -m Python.Server_AGI --live',
        },
    ]

    out = ui._runtime_owner_health(procs)

    assert out["ok"] is True
    assert out["issues"] == []
