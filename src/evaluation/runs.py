"""
Run tracking: every evaluation run is saved as results/runs/<run_id>/run.json,
together with the git commit, all config values and the index metadata, so a
number in a table can always be traced back to exactly what produced it.
"""

import datetime as dt
import json
import subprocess
from pathlib import Path

import config

RESULTS_DIR = config.ROOT_DIR / "results"
RUNS_DIR = RESULTS_DIR / "runs"


def new_run_id(name):
    stamp = dt.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in name)
    return f"{stamp}_{safe}"


def git_info():
    def git(*args):
        try:
            return subprocess.run(["git", *args], cwd=config.ROOT_DIR, capture_output=True,
                                  text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    # "dirty" = the commit alone doesn't describe the code: tracked files were
    # changed, or new Python files exist that aren't committed yet. (Data files
    # such as the question review log are allowed to change.)
    status = git("status", "--porcelain", "--untracked-files=all") or ""
    dirty = any(
        not line.startswith("??") or line.rstrip().endswith(".py")
        for line in status.splitlines()
    )
    return {
        "commit": git("rev-parse", "--short", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": dirty,
    }


def config_snapshot():
    """All UPPER_CASE settings from config.py; paths shown relative to the project."""
    snap = {}
    for key, value in vars(config).items():
        if not key.isupper():
            continue
        if isinstance(value, Path):
            try:
                value = str(value.relative_to(config.ROOT_DIR))
            except ValueError:
                value = str(value)
        snap[key] = value
    snap.pop("ROOT_DIR", None)
    return snap


def save_run(run):
    run_dir = RUNS_DIR / run["run_id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "run.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(run, f, indent=2, ensure_ascii=False)
    return path


def load_runs():
    runs = []
    for path in sorted(RUNS_DIR.glob("*/run.json")):
        with open(path, encoding="utf-8") as f:
            runs.append(json.load(f))
    return runs
