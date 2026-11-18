"""Move lines written by the test suite out of the real training logs (before tests were redirected to a temp
folder by AGI_LOG_DIR in tests/conftest.py). Test runs write short bursts; each burst is found by its markers
(pytest temp paths, fake candidate names, the fake market 'X', test halt reasons) and every line within 5 s of
a marker is moved to <log>.test-lines.log. Nothing is deleted.
    python3 tools/clean_test_logs.py        (do not run while a training job is writing)
"""
import datetime as dt
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = ["ppo_training.log", "lstm_training.log", "dreamer_training.log"]
MARK = re.compile(r"pytest-of-|registry_symbol_test|test_reason|candidate_(eur|xau)?_?v\d|STACK X \||agi_test_logs_|"
                  r"/T/pytest|Canary set: candidate|Promoted champion: candidate")
TS = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)")


def clean(name):
    path = os.path.join(ROOT, "logs", name)
    if not os.path.exists(path):
        return None
    lines = open(path, encoding="utf-8", errors="replace").readlines()
    stamps = []
    for ln in lines:
        m = TS.match(ln)
        stamps.append(dt.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S") if m else None)
    marks = sorted({stamps[i] for i, ln in enumerate(lines) if stamps[i] and MARK.search(ln)})
    win = dt.timedelta(seconds=5)

    def in_burst(t):
        if t is None:
            return False
        lo, hi = 0, len(marks)
        while lo < hi:                       # nearest marker by binary search
            mid = (lo + hi) // 2
            if marks[mid] < t:
                lo = mid + 1
            else:
                hi = mid
        return any(0 <= j < len(marks) and abs(marks[j] - t) <= win for j in (lo - 1, lo))

    keep, moved, prev_moved = [], [], False
    for ln, t in zip(lines, stamps):
        m = in_burst(t) if t is not None else prev_moved     # continuation lines follow their parent
        (moved if m else keep).append(ln)
        prev_moved = m
    if moved:
        with open(path + ".test-lines.log", "a", encoding="utf-8") as f:
            f.writelines(moved)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.writelines(keep)
        os.replace(tmp, path)
    return len(keep), len(moved), len(marks)


if __name__ == "__main__":
    for n in LOGS:
        r = clean(n)
        if r:
            print(f"{n}: kept {r[0]} lines, moved {r[1]} test lines ({r[2]} marker seconds)")
