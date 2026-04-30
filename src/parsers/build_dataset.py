"""Build the unified running dataset from frozen historical + snapshot CSV.

Reads historical_daily.csv (2016 through last finalized year) plus a single
drive_snapshot.csv containing the current-year log and three adjustment
tables (changes, additions, locations). Produces daily.csv + races.csv.

USAGE:
  # Default — resolve inputs from standard locations. If no snapshot is
  # found, fetches a fresh one from Drive automatically (requires OAuth
  # credentials configured for drive_fetch):
  python build_dataset.py

  # Explicit paths:
  python build_dataset.py --historical PATH --snapshot PATH [--out-dir PATH]

  # Force a fresh snapshot fetch even when a local one exists:
  python build_dataset.py --refresh-snapshot

  # Skip auto-fetch and fail if no snapshot is found:
  python build_dataset.py --no-fetch

  # Override current year (auto-detected from snapshot header by default):
  python build_dataset.py --current-year 2026

Snapshot auto-fetch writes to data/drive_snapshot.csv by default.

Validation: after the build, any race missing a city-state location or an
event name is reported. The goal is every race having both.
"""
import argparse
import os
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.shared.paths import DATA_DIR

from running_log_parser import (
    DAILY_COLUMNS,
    ingest_year_standard_csv,
    build_race_segments,
    apply_race_rules,
    set_race_surface_source,
    apply_adjustments_from_df,
    ingest_additions_from_df,
    append_additions,
    ingest_locations_from_df,
    apply_autopop,
    surface_from_location,
)
from snapshot import read_snapshot, find_snapshot


DEFAULT_OUT_DIR = str(DATA_DIR)
DEFAULT_HISTORICAL = str(DATA_DIR / "historical_daily.csv")
DEFAULT_SNAPSHOT   = str(DATA_DIR / "drive_snapshot.csv")


def _validate_races(races):
    """Report races missing city_state or event. Returns (n_no_city, n_no_event).

    A valid location is non-null and contains a comma (the city-state form,
    e.g. 'Boston, MA'). Event must be non-null and non-empty.
    """
    def _has_city_state(v):
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return False
        return "," in str(v)

    def _has_event(v):
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return False
        return str(v).strip() != ""

    no_city = races[~races["location"].apply(_has_city_state)].copy()
    no_event = races[~races["event"].apply(_has_event)].copy()

    print()
    print("=" * 72)
    print("VALIDATION: every race should have a city-state location AND an event")
    print("=" * 72)
    print(f"Races missing city-state: {len(no_city)} / {len(races)}")
    print(f"Races missing event:      {len(no_event)} / {len(races)}")

    def _year_of(d):
        return d.year if hasattr(d, "year") else pd.Timestamp(d).year

    if len(no_city):
        print()
        print(f"--- Missing city-state, by year ---")
        for yr, sub in no_city.groupby(no_city["date"].apply(_year_of)):
            print(f"  {yr}: {len(sub)} race(s)")
            for _, r in sub.head(3).iterrows():
                print(f"    {r['date']}  {int(r['distance_m'])}m  "
                      f"location={r['location']!r}  event={r['event']!r}  "
                      f"surface={r['surface']}")
            if len(sub) > 3:
                print(f"    ... and {len(sub) - 3} more")

    if len(no_event):
        print()
        print(f"--- Missing event, by year ---")
        for yr, sub in no_event.groupby(no_event["date"].apply(_year_of)):
            print(f"  {yr}: {len(sub)} race(s)")
            for _, r in sub.head(3).iterrows():
                print(f"    {r['date']}  {int(r['distance_m'])}m  "
                      f"location={r['location']!r}  event={r['event']!r}")
            if len(sub) > 3:
                print(f"    ... and {len(sub) - 3} more")

    return len(no_city), len(no_event)


# Columns from the locations sheet to propagate onto daily rows. Listed in
# the order they should appear if all are present. Any column missing from
# the sheet is silently skipped — the merge is forward-compatible with a
# sheet that only has city_state, or one that grows new metadata columns
# later.
LOCATION_METADATA_COLS = [
    "display_name",
    "city_state",
    "elev_per_mile",
    "altitude",
    "terrain_type",
]


def _join_location_metadata(daily, locations_df):
    """Left-join the locations sheet onto daily rows on `location`.

    Adds whichever of LOCATION_METADATA_COLS are present in locations_df.
    Daily rows whose location doesn't appear in the sheet (or whose location
    is blank) get NaN/empty in the new columns. Existing daily columns are
    not overwritten — if `city_state` already lives on daily for any reason,
    the merge is no-op'd for that column.
    """
    if locations_df is None or len(locations_df) == 0:
        print("[locations] no locations sheet — skipping daily metadata join")
        return daily
    if "log_location" not in locations_df.columns:
        print("[locations] WARNING: locations sheet has no log_location column")
        return daily

    keep = ["log_location"] + [c for c in LOCATION_METADATA_COLS
                                if c in locations_df.columns
                                and c not in daily.columns]
    if len(keep) == 1:
        # Only the key column matches — nothing to add
        print("[locations] no new metadata columns to add to daily")
        return daily

    sheet = locations_df[keep].copy()
    sheet["log_location"] = sheet["log_location"].astype(str).str.strip().str.lower()
    sheet = sheet.drop_duplicates(subset=["log_location"], keep="last")

    # Normalize the daily side without mutating the original column
    daily = daily.copy()
    daily["_loc_key"] = daily["location"].astype(str).str.strip().str.lower()
    merged = daily.merge(sheet, left_on="_loc_key", right_on="log_location",
                         how="left")
    merged = merged.drop(columns=["_loc_key", "log_location"])
    print(f"[locations] joined {len(sheet)} sheet entries onto {len(merged)} "
          f"daily rows ({len(keep) - 1} metadata col(s) added)")
    return merged


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--historical",
                   help="Path to historical_daily.csv")
    p.add_argument("--snapshot",
                   help="Path to drive_snapshot.csv")
    p.add_argument("--current-year", type=int,
                   help="Year of the current log (default: from snapshot header, "
                        "or today's year)")
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR,
                   help=f"Output directory (default: {DEFAULT_OUT_DIR})")
    p.add_argument("--refresh-snapshot", action="store_true",
                   help="Ignore any local snapshot and fetch a fresh one from Drive")
    p.add_argument("--no-fetch", action="store_true",
                   help="Fail if no local snapshot is found (skip Drive auto-fetch)")
    args = p.parse_args()

    # ---------- resolve inputs ----------
    historical = args.historical or find_snapshot([DEFAULT_HISTORICAL])
    if not historical:
        print(f"ERROR: historical_daily.csv not found. Pass --historical PATH or "
              f"place at {DEFAULT_HISTORICAL}", file=sys.stderr)
        sys.exit(1)

    # Snapshot resolution:
    #   1. --snapshot PATH (explicit, always wins)
    #   2. find_snapshot at DEFAULT_SNAPSHOT, unless --refresh-snapshot
    #   3. auto-fetch from Drive, unless --no-fetch
    snapshot_path = args.snapshot
    if not snapshot_path and not args.refresh_snapshot:
        snapshot_path = find_snapshot([DEFAULT_SNAPSHOT])

    if not snapshot_path and not args.no_fetch:
        year_for_fetch = args.current_year or date.today().year
        out_path = DEFAULT_SNAPSHOT
        print(f"[build] no local snapshot found — fetching year {year_for_fetch} "
              f"from Drive...")
        try:
            import drive_fetch
            service = drive_fetch.get_drive_service()
            drive_fetch.build_snapshot(service, year_for_fetch, out_path)
            snapshot_path = out_path
        except Exception as e:
            print(f"ERROR: snapshot auto-fetch failed: {e}", file=sys.stderr)
            print(f"  Either (a) pass --snapshot PATH to use a specific file,",
                  file=sys.stderr)
            print(f"         (b) run `python drive_fetch.py snapshot "
                  f"--year {year_for_fetch}` manually,", file=sys.stderr)
            print(f"      or (c) use --no-fetch to skip this step.",
                  file=sys.stderr)
            sys.exit(1)

    if not snapshot_path:
        print(f"ERROR: drive_snapshot.csv not found and --no-fetch was set. "
              f"Pass --snapshot PATH or place at {DEFAULT_SNAPSHOT}",
              file=sys.stderr)
        sys.exit(1)

    print(f"[build] historical  = {historical}")
    print(f"[build] snapshot    = {snapshot_path}")

    # ---------- parse snapshot ----------
    sections, metas = read_snapshot(snapshot_path)
    for name, df in sections.items():
        print(f"[build] snapshot section {name!r}: {len(df)} rows")

    current_log_df = sections.get("current_log", pd.DataFrame())
    changes_df = sections.get("changes", pd.DataFrame())
    additions_df = sections.get("additions", pd.DataFrame())
    locations_df = sections.get("locations", pd.DataFrame())

    current_year = args.current_year
    if current_year is None:
        meta = metas.get("current_log", {})
        if "year" in meta:
            try:
                current_year = int(meta["year"])
            except ValueError:
                pass
    if current_year is None:
        current_year = date.today().year
    print(f"[build] current year = {current_year}")
    print()

    # ---------- load historical ----------
    hist = pd.read_csv(historical, parse_dates=["date"])
    hist["date"] = hist["date"].dt.date
    print(f"[build] loaded {len(hist)} historical rows "
          f"({hist['year'].min()}-{hist['year'].max()})")

    # ---------- parse current-year CSV ----------
    current_rows = ingest_year_standard_csv(current_log_df, current_year)
    current_df = pd.DataFrame(current_rows, columns=DAILY_COLUMNS)
    print(f"[build] parsed {len(current_df)} rows from current-year snapshot")

    overlap = set(hist["year"].unique()) & set(current_df["year"].unique())
    if overlap:
        print(f"[build] WARNING: year overlap between historical and current: {overlap}")

    # ---------- combine ----------
    daily = pd.concat([hist, current_df], ignore_index=True)
    daily = daily.sort_values("date").reset_index(drop=True)
    daily["surface"] = daily["surface"].fillna("Unknown")
    print(f"[build] combined daily: {len(daily)} rows")

    # ---------- build races from daily ----------
    race_dicts = build_race_segments(daily.to_dict("records"))
    races = pd.DataFrame(race_dicts)
    print(f"[build] extracted {len(races)} race segments from logs")

    # ---------- additions ----------
    additions = ingest_additions_from_df(additions_df)
    races, add_count, add_dups = append_additions(races, additions)
    print(f"[additions] added={add_count}  dup_skipped={add_dups}")

    # ---------- race surface rules ----------
    races["surface"] = races.apply(apply_race_rules, axis=1)
    races["surface_source"] = races.apply(set_race_surface_source, axis=1)

    # ---------- adjustments (changes sheet) ----------
    races, adj_applied, adj_failed = apply_adjustments_from_df(races, changes_df)
    print(f"[adjustments] applied={adj_applied}  failed={adj_failed}")

    # ---------- autopop ----------
    location_lookup = ingest_locations_from_df(locations_df)
    print(f"[autopop] locations lookup: {len(location_lookup)} entries")

    autopop_stats = {"hardcoded": 0, "lookup": 0, "flag": 0, "noop": 0}
    autopop_flags = []
    for idx in races.index:
        row = races.loc[idx].to_dict()
        status = apply_autopop(row, location_lookup)
        autopop_stats[status] += 1
        if status in ("hardcoded", "lookup"):
            races.at[idx, "location"] = row["location"]
            current = races.at[idx, "event"]
            if row.get("event") and (not current or pd.isna(current)):
                races.at[idx, "event"] = row["event"]
        elif status == "flag":
            autopop_flags.append({
                "date": row.get("date"), "race_seq": row.get("race_seq"),
                "location": row.get("location") or "",
                "event": row.get("event") or "",
            })
    print(f"[autopop] hardcoded={autopop_stats['hardcoded']}  "
          f"lookup={autopop_stats['lookup']}  "
          f"flag={autopop_stats['flag']}  noop={autopop_stats['noop']}")

    # ---------- surface refresh after autopop ----------
    surface_refreshed = 0
    for idx in races.index:
        if races.at[idx, "surface_source"] == "adjustment":
            continue
        loc_surface = surface_from_location(races.at[idx, "location"])
        if loc_surface and races.at[idx, "surface"] != loc_surface:
            races.at[idx, "surface"] = loc_surface
            races.at[idx, "surface_source"] = "location"
            surface_refreshed += 1
    if surface_refreshed:
        print(f"[autopop] refreshed surface on {surface_refreshed} row(s) via location")

    races["surface"] = races["surface"].fillna("Unknown")

    # ---------- propagate location metadata to daily rows ----------
    # Left-join the locations sheet onto daily so display_name, city_state,
    # elev_per_mile, altitude, and terrain_type are available downstream
    # (e.g. recovery/long plotters, training-quality corrections) without
    # those tools needing to re-load the snapshot. Columns are added only
    # when present in the locations sheet, so this is forward-compatible
    # with sheets that don't yet have all metadata columns.
    daily = _join_location_metadata(daily, locations_df)

    # ---------- back-propagate race city_state + surface to daily ----------
    # races.csv holds the post-adjustment ground truth for race-day location
    # and surface (sourced from additions/changes for 2016-17 races, from the
    # log-location join for 2018+). Daily race rows should mirror it so
    # qualitative daily-level charts (city_state, surface) are correct on
    # race days even when the original log entry had no location (2016-17)
    # or when an adjustment corrected a 2018+ race after the fact.
    #
    # Match on date with race_seq==1 to dedupe multi-race days; race_seq>=2
    # rows are the same date and (in practice) the same location/surface.
    # Overwrite existing daily values: races.csv is the canonical source.
    if len(races):
        race_primary = races[races["race_seq"] == 1].copy()
        race_primary["date"] = pd.to_datetime(race_primary["date"]).dt.date.astype(str)
        cs_map = race_primary.set_index("date")["location"].to_dict()
        sf_map = race_primary.set_index("date")["surface"].to_dict()
        date_str = pd.to_datetime(daily["date"]).dt.date.astype(str)
        is_race = daily["run_type"] == "race"
        n_before = is_race.sum()
        cs_filled = date_str[is_race].map(cs_map)
        sf_filled = date_str[is_race].map(sf_map)
        daily.loc[is_race, "city_state"] = cs_filled.values
        daily.loc[is_race, "surface"] = sf_filled.values
        n_cs = cs_filled.notna().sum()
        n_sf = sf_filled.notna().sum()
        print(f"[backprop] race rows on daily: {n_before}; "
              f"city_state set: {n_cs}, surface set: {n_sf}")

    # ---------- final daily metadata coverage ----------
    # Reported after backprop so the numbers reflect the daily.csv state
    # actually written to disk. Each join column is checked for non-null
    # count; surface is included since back-prop overwrites it on race rows.
    coverage_cols = [c for c in (LOCATION_METADATA_COLS + ["surface"])
                     if c in daily.columns]
    n_total = len(daily)
    print(f"[coverage] final daily.csv ({n_total} rows):")
    for col in coverage_cols:
        if col == "surface":
            n_pop = (daily[col].notna() & (daily[col] != "Unknown")).sum()
            label = f"{col} (non-Unknown)"
        else:
            n_pop = daily[col].notna().sum()
            label = col
        print(f"  {label:25s} {n_pop}/{n_total}")

    # ---------- write outputs ----------
    os.makedirs(args.out_dir, exist_ok=True)
    daily_out = os.path.join(args.out_dir, "daily.csv")
    races_out = os.path.join(args.out_dir, "races.csv")
    daily.to_csv(daily_out, index=False)
    races.to_csv(races_out, index=False)

    # ---------- summary ----------
    print()
    print("=" * 72)
    print("BUILD SUMMARY")
    print("=" * 72)
    print(f"Daily rows: {len(daily)}  ({daily['date'].min()} → {daily['date'].max()})")
    print(f"Race rows:  {len(races)}  "
          f"({int(races['fatigued'].sum())} fatigued)")
    print()
    print("Miles by year:")
    print(f"{'year':>6} {'miles':>10} {'days':>6}")
    for yr in sorted(daily["year"].unique()):
        sub = daily[daily["year"] == yr]
        print(f"{yr:>6} {sub['miles'].sum():>10.2f} {len(sub):>6}")
    print()
    print("Race surface breakdown:")
    print(races["surface"].value_counts(dropna=False).to_string())
    print()
    print("Race surface_source breakdown:")
    print(races["surface_source"].value_counts(dropna=False).to_string())

    if autopop_flags:
        print()
        print(f"Autopop fell through on {len(autopop_flags)} race(s) — "
              f"add to 'locations' section of snapshot, or use 'changes':")
        for f in autopop_flags[:20]:
            # Defensive: location/event may be NaN floats when blank in the
            # race row (pandas uses NaN for missing strings in object dtype
            # columns). Coerce before slicing.
            loc = f['location']
            ev = f['event']
            loc_str = "" if (loc is None or (isinstance(loc, float) and pd.isna(loc))) else str(loc)
            ev_str = "" if (ev is None or (isinstance(ev, float) and pd.isna(ev))) else str(ev)[:50]
            print(f"  {f['date']} seq={f['race_seq']}  "
                  f"loc={loc_str!r}  event={ev_str!r}")
        if len(autopop_flags) > 20:
            print(f"  ... and {len(autopop_flags) - 20} more")

    # ---------- validation ----------
    n_no_city, n_no_event = _validate_races(races)

    print()
    print(f"Wrote {daily_out}")
    print(f"Wrote {races_out}")

    if n_no_city or n_no_event:
        print()
        print(f"VALIDATION: {n_no_city} race(s) missing city-state, "
              f"{n_no_event} missing event. Address via the 'changes' section "
              f"of drive_snapshot.csv (field=location / field=event).")


if __name__ == "__main__":
    main()
