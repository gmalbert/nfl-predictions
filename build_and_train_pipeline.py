"""Build and Train Pipeline

This runner builds historical game data then trains/generates predictions.

Runs, in order:
  1. `create-nfl-historical.py` (fetch/save game summaries)
  2. `nfl-gather-data.py` (feature engineering, model training, predictions)

Usage:
    python build_and_train_pipeline.py

Notes:
- Ensure your virtualenv is active and required packages (nfl_data_py, xgboost, scikit-learn, pandas) are installed.
"""
from __future__ import annotations
import subprocess
import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def run_script(script_path: Path) -> int:
    print(f"\n==> Running: {script_path.name}")
    try:
        proc = subprocess.run([sys.executable, str(script_path)], check=False)
        print(f"<== Exit code: {proc.returncode}\n")
        return proc.returncode
    except Exception as e:
        print(f"Failed to run {script_path}: {e}")
        return 2

def main() -> int:
    os.makedirs(ROOT / 'data_files', exist_ok=True)
    # Run schedule updater first (best-effort). Keeps local schedule current
    # before building historical data and training models.
    schedule_script = ROOT / 'update_schedule.py'
    if schedule_script.exists():
        code = run_script(schedule_script)
        if code != 0:
            print("Warning: update_schedule.py failed — continuing pipeline")

    create_script = ROOT / 'create-nfl-historical.py'
    gather_script = ROOT / 'nfl-gather-data.py'

    if not create_script.exists():
        print(f"Missing script: {create_script}")
        return 3
    if not gather_script.exists():
        print(f"Missing script: {gather_script}")
        return 4

    code = run_script(create_script)
    if code != 0:
        print("Aborting: create-nfl-historical failed")
        return code

    code = run_script(gather_script)
    if code != 0:
        print("Aborting: nfl-gather-data failed")
        return code

    print("\nALL STEPS COMPLETED SUCCESSFULLY")
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
