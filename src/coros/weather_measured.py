"""Watch-derived daily weather, for enriching the hand-logged (drive) profile.

The drive/`max` profile is hand-fed and doesn't read the watch cache. This
bridge re-derives the per-day weather Max's Coros recorded — temperature,
wind (m/s), humidity, and time-of-day — and writes ``weather_measured.csv``
into the data dir. ``build_dataset`` left-joins it onto daily for watch-era
days (override temp_c + time_of_day, fill wind_ms + humidity_pct; the
qualitative ``weather`` bin is held — see the accuracy comparison spike).

The values come straight from ``build_current_log`` so they're identical to
what the Coros profile ships (same rep selection, same bin maps, the same
day-level "most activity time" time-of-day rule). The producer is Max-specific
by nature: it always reads the ``coros`` profile's cache regardless of which
profile's pipeline invokes it.

    python -m src.coros.weather_measured            # -> $RP_DATA_DIR/weather_measured.csv
    python -m src.coros.weather_measured --force    # rebuild even if up to date

A no-op (writes nothing) when the coros details cache is absent — e.g. CI that
hasn't synced the watch — so the join downstream simply finds no file.

Past activity days are immutable, so the rebuild is skipped entirely unless a
cached detail file is newer than the existing CSV (an `os.stat` sweep, ~ms).
The full re-derive (~5s, dominated by re-parsing the 339 MB of per-second
streams the elevation backfill cached) runs only when genuinely new activities
have synced — or on `--force` (use after a mappings.py / rule change, which
mtimes can't detect).
"""
from __future__ import annotations

import argparse

import pandas as pd

from src.coros.build_current_log import build_current_log
from src.coros.sync import _load_all_details
from src.profiles import get_profile
from src.shared.paths import DATA_DIR

# Columns the build_dataset join consumes. `weather` is intentionally excluded:
# the qualitative bin stays hand-logged (held in the accuracy comparison).
MEASURED_COLUMNS = ["date", "temp_c", "wind_ms", "humidity_pct", "time_of_day"]


def _is_fresh(out_path, details_dir) -> bool:
    """True if the cached CSV exists and no detail file is newer than it —
    i.e. nothing new has synced since the last build (past days are immutable,
    so the CSV is still correct). Returns on the first newer file found."""
    if not out_path.exists():
        return False
    csv_mtime = out_path.stat().st_mtime
    for p in details_dir.glob("*.json"):
        if p.stat().st_mtime > csv_mtime:
            return False
    return True


def build_weather_measured() -> pd.DataFrame:
    """Return the watch-derived daily-weather frame (empty if no cache)."""
    details_dir = get_profile("coros").data_dir / "details"
    if not details_dir.exists():
        return pd.DataFrame(columns=MEASURED_COLUMNS)
    details = _load_all_details(details_dir)
    if not details:
        return pd.DataFrame(columns=MEASURED_COLUMNS)
    df, _ = build_current_log(details, geocode=False)
    return df[MEASURED_COLUMNS].copy()


def main(force=False):
    out = DATA_DIR / "weather_measured.csv"
    details_dir = get_profile("coros").data_dir / "details"
    if not details_dir.exists():
        print("[weather_measured] no coros details cache — skipped")
        return
    if not force and _is_fresh(out, details_dir):
        print(f"[weather_measured] up to date (no detail newer than "
              f"{out.name}) — skipped")
        return
    df = build_weather_measured()
    if df.empty:
        print("[weather_measured] no coros details cache — skipped")
        return
    df.to_csv(out, index=False)
    print(f"[weather_measured] wrote {len(df)} watch-weather days -> {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    p.add_argument("--force", action="store_true",
                   help="rebuild even if the CSV is newer than all details "
                        "(use after a mappings/rule change)")
    main(force=p.parse_args().force)
