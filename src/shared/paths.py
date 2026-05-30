"""Repo-rooted paths.

Single source of truth for where the pipeline reads and writes:
- DATA_DIR for inputs and intermediate data products that the plot
  pipeline actually consumes (daily.csv, races.csv, drive_snapshot.csv,
  workout_decomposed.csv, bayes_cs_summary/params/auto_exclusions.csv).
- OUTPUT_DIR for terminal artifacts (HTML plots, plotly bundle).
- DEBUG_DIR for diagnostic-only outputs that are produced for human
  inspection but not consumed by anything downstream (per-script
  ``--diagnostics`` flag gates writes here so they never clutter
  default runs). Producers must ``mkdir(parents=True, exist_ok=True)``
  before writing because this directory may not exist yet.

Multi-profile: DATA_DIR and OUTPUT_DIR honor the ``RP_DATA_DIR`` and
``RP_OUTPUT_DIR`` environment variables. The per-profile orchestrator
(scripts/build_profiles.py) sets these before invoking each data/plot
script as a subprocess, so the entire pipeline reroutes to a profile's
own directories with no changes to the scripts themselves. Unset (the
default / dev case) falls back to the repo-rooted data/ and output/.
"""
import os
from pathlib import Path

REPO_ROOT  = Path(__file__).resolve().parents[2]
DATA_DIR   = Path(os.environ.get("RP_DATA_DIR", REPO_ROOT / "data"))
OUTPUT_DIR = Path(os.environ.get("RP_OUTPUT_DIR", REPO_ROOT / "output"))
DEBUG_DIR  = OUTPUT_DIR / "debug"
