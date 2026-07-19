"""Unified per-day watch-derived table — one source of truth, built by presence.

`data/watch_daily.csv` holds the scalar per-day quantities derived from the
Coros cache that the pipeline consumes. It is built INCREMENTALLY: a populated
row is trusted (the per-second stream is immutable), so only days whose cached
activity set changed are re-derived — the common daily build parses just the
new run's files. Full rebuilds happen on `--force`, a schema bump, or the CS
safety valve. See docs/watch-derived-cache-spec.md.

Legacy per-producer CSVs are written as thin PROJECTIONS of this table, so
existing consumers need no changes (e.g. weather_measured.csv).

    python -m src.coros.watch_daily              # incremental
    python -m src.coros.watch_daily --force      # full rebuild

Watch-only and Max-specific by nature: always reads the `coros` profile's
cache, writes to $RP_DATA_DIR (so the drive/max build is enriched from Max's
watch). No cache -> no-op.
"""
from __future__ import annotations

import argparse

import pandas as pd

from src.coros import mappings as M
from src.coros import watch_cache as wc
from src.coros.build_current_log import (Activity, build_current_log,
                                         strip_paused)
from src.coros.long_runs import measure_day
from src.profiles import get_profile
from src.shared.paths import DATA_DIR

# Bump when the set of derived columns / their derivation changes — forces a
# full rebuild on the next run (mtimes can't see a code change).
SCHEMA_VERSION = 4   # 4: wind_ms -> wind_mph (raw/10 is km/h, displayed in mph)

WATCH_DAILY_COLUMNS = [
    "date", "n_acts", "status",
    "temp_c", "weather_bin", "wind_mph", "humidity_pct", "time_of_day",
    "any_indoor", "watch_miles", "watch_moving_s", "watch_total_s",
    "pause_s", "stall_s", "n_segs", "d_eff_frac", "longest_seg_mi",
    "label_ids", "schema_version",
]

# Lightweight per-activity index (one row per run activity), maintained
# incrementally alongside the table. Lets reps find candidate workout days
# (track-sport detection) and the per-day labelId set WITHOUT parsing streams.
WATCH_ACTIVITIES_COLUMNS = ["labelId", "date", "sport_type", "rich"]

# weather_measured.csv projection (what build_dataset._apply_weather_measured
# joins). `weather_bin` is deliberately excluded — the hand-logged bin is held.
WEATHER_MEASURED_COLUMNS = ["date", "temp_c", "wind_mph", "humidity_pct",
                            "time_of_day"]


def _build_rows(targets):
    """Derive watch_daily rows for the target days. One build_current_log call
    over all target details yields the per-day weather; n_acts/status/label_ids
    come from the day's file set."""
    all_details = [d for items in targets.values() for (_lid, d) in items]
    wrows = {}
    if all_details:
        df, _ = build_current_log(all_details, geocode=False)
        wrows = {r["date"]: r for _, r in df.iterrows()}

    rows, index_rows = [], []
    for day, items in targets.items():
        # Run activities for this day, keeping labelIds for the index.
        run_items = [(lid, d, Activity(d)) for (lid, d) in items
                     if (d.get("summary") or {}).get("sportType") in M.RUN_SPORTS]
        # measure_day wants (rec_dict, Activity), time-ordered within the day.
        # strip_paused re-virtualizes pauses into stream gaps so the stall
        # detector keeps skipping paused dwell on post-V3.1708.0 recordings.
        acts = sorted(((strip_paused(d), a) for (lid, d, a) in run_items),
                      key=lambda t: t[1].start_utc)
        w = wrows.get(day)
        if not acts or w is None:
            continue                          # run-less day -> no row
        m = measure_day(acts)
        for lid, d, a in run_items:
            index_rows.append({"labelId": lid, "date": day,
                               "sport_type": a.sport_type, "rich": "freq" in d})
        rows.append({
            "date": day,
            "n_acts": m["n_acts"],
            "status": m["status"],
            "temp_c": w["temp_c"],
            "weather_bin": w["weather"],
            "wind_mph": w["wind_mph"],
            "humidity_pct": w["humidity_pct"],
            "time_of_day": w["time_of_day"],
            "any_indoor": m["any_indoor"],
            "watch_miles": m["watch_miles"],
            "watch_moving_s": m["watch_moving_s"],
            "watch_total_s": m["watch_total_s"],
            "pause_s": m["pause_s"],
            "stall_s": m["stall_s"],
            "n_segs": m["n_segs"],
            "d_eff_frac": m["d_eff_frac"],
            "longest_seg_mi": m["longest_seg_mi"],
            # all present labelIds that day (run + non-run) so they're marked
            # consumed and not re-parsed next build:
            "label_ids": " ".join(lid for (lid, _d) in items),
            "schema_version": SCHEMA_VERSION,
        })
    return rows, index_rows


def _project_weather_measured(out_dir, table):
    table[WEATHER_MEASURED_COLUMNS].to_csv(
        out_dir / "weather_measured.csv", index=False)


def build(*, force=False, full_regen=False):
    """(Re)build watch_daily.csv and its projections. Returns the table, or
    None if there's no cache to build from."""
    details_dir = get_profile("coros").data_dir / "details"
    if not details_dir.exists():
        return None
    out = DATA_DIR / "watch_daily.csv"
    idx_out = DATA_DIR / "watch_activities.csv"

    existing = pd.read_csv(out, dtype=str) if out.exists() else None
    existing_idx = (pd.read_csv(idx_out, dtype=str)
                    if idx_out.exists() else None)
    full = force or full_regen
    existing_ids_by_date = {}
    if existing is not None and len(existing):
        if str(existing["schema_version"].iloc[0]) != str(SCHEMA_VERSION):
            full = True                       # schema changed -> rebuild all
        else:
            for _, r in existing.iterrows():
                existing_ids_by_date[r["date"]] = set(str(r["label_ids"]).split())

    targets, dropped, _ = wc.plan(existing_ids_by_date, details_dir, full=full)

    if not targets and not dropped and not full and existing is not None:
        _project_weather_measured(DATA_DIR, existing.sort_values("date"))
        print(f"[watch_daily] up to date ({len(existing)} days) — no parse")
        return existing

    new_rows, new_index = _build_rows(targets)
    new_df = pd.DataFrame(new_rows, columns=WATCH_DAILY_COLUMNS)
    new_idx = pd.DataFrame(new_index, columns=WATCH_ACTIVITIES_COLUMNS)
    if full or existing is None:
        table, idx = new_df, new_idx
    else:
        touched = set(new_df["date"]) | dropped
        kept = existing[~existing["date"].isin(touched)]
        table = pd.concat([kept, new_df], ignore_index=True)
        kept_idx = (existing_idx[~existing_idx["date"].isin(touched)]
                    if existing_idx is not None else None)
        idx = pd.concat([k for k in (kept_idx, new_idx) if k is not None],
                        ignore_index=True)
    table = table.sort_values("date").reset_index(drop=True)
    idx = idx.sort_values(["date", "labelId"]).reset_index(drop=True)

    table.to_csv(out, index=False)
    idx.to_csv(idx_out, index=False)
    _project_weather_measured(DATA_DIR, table)
    mode = "rebuilt" if (full or existing is None) else "updated"
    print(f"[watch_daily] {mode}: {len(table)} days "
          f"({len(new_rows)} derived, {len(dropped)} dropped) -> {out.name}")
    return table


def main(force=False, full_regen=False):
    if build(force=force, full_regen=full_regen) is None:
        print("[watch_daily] no coros details cache — skipped")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    p.add_argument("--force", action="store_true",
                   help="full rebuild, ignoring the presence cache")
    p.add_argument("--full-regen", action="store_true",
                   help="full rebuild (safety-valve hook; e.g. CS refit)")
    a = p.parse_args()
    main(force=a.force, full_regen=a.full_regen)
