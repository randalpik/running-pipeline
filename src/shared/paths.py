"""Repo-rooted paths.

Single source of truth for where the pipeline reads and writes:
- DATA_DIR for inputs (daily.csv, races.csv, drive_snapshot.csv, ...) and
  intermediate data products (bayes_cs_*.csv/.nc, workout_*.csv).
- OUTPUT_DIR for terminal artifacts (HTML plots).
"""
from pathlib import Path

REPO_ROOT  = Path(__file__).resolve().parents[2]
DATA_DIR   = REPO_ROOT / "data"
OUTPUT_DIR = REPO_ROOT / "output"
