"""Freeze the 2016-2025 running logs into a single CSV.

Run this rarely — only when a parser bug is fixed or logs are retroactively corrected.
Regular dataset builds use build_dataset.py which reads the frozen CSV plus the
current-year xlsx plus adjustments.

USAGE:
  # Default — fetch all 'Running Log YYYY' logs + adjustments xlsx from Drive:
  python freeze_historical.py [--out PATH]

  # Local override — use already-downloaded xlsx files (also what Claude uses
  # when running this in a chat, pre-staged via MCP fetches):
  python freeze_historical.py --logs-dir /path/to/xlsx/ \\
                              [--adjustments-xlsx /path/to/adjustments.xlsx] \\
                              [--out PATH]

Drive default requires drive_fetch.py and one-time OAuth setup — see
drive_fetch.py's module docstring.

Files must be named:
  runningLog2016.xlsx  OR  Running_Log_2016.xlsx  OR  'Running Log 2016.xlsx'
  Running_Log_2017.xlsx through Running_Log_2025.xlsx (or 'Running Log YYYY.xlsx')

The adjustments xlsx (Max's Running Data) is required for 2016-17 location
synthesis: its `hills` tab maps loop abbreviations (rc, lc, pwr1, ...) to
log_locations so hill workouts in 2016-17 (which have no location column in
the source schemas) get a meaningful location field. If --logs-dir is used
without --adjustments-xlsx, the freeze tries `adjustments.xlsx` in logs-dir
first, then `Max's Running Data.xlsx`; if neither is found, hill workouts
fall back to the default rule and a warning is printed.
"""
import argparse
import os
import sys
import tempfile
from datetime import date

import pandas as pd

from running_log_parser import (
    AUTHORITATIVE_TOTALS,
    DAILY_COLUMNS,
    ingest_year_standard,
    ingest_2016,
    ingest_hills_from_df,
)


DEFAULT_OUT = "./output/historical_daily.csv"
# range is exclusive on the upper bound, so this covers 2016 through last
# completed year. At refreeze time (typically early in a new year), the prior
# year has just finalized and will be included; the current in-progress year
# is deliberately excluded — build_dataset handles that from the live log.
HISTORICAL_YEARS = range(2016, date.today().year)


def resolve_log_path(logs_dir, year):
    """Find the xlsx file for a given year. Tolerates old naming."""
    if year == 2016:
        candidates = ["Running_Log_2016.xlsx", "Running Log 2016.xlsx", "runningLog2016.xlsx"]
    else:
        candidates = [f"Running_Log_{year}.xlsx", f"Running Log {year}.xlsx"]
    for name in candidates:
        p = os.path.join(logs_dir, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"No log file found for {year} in {logs_dir}; tried {candidates}")


def fetch_logs_from_drive():
    """Download all HISTORICAL_YEARS logs + adjustments xlsx from Drive into a tmp dir.

    Returns (tmp_dir, adjustments_xlsx_path).
    """
    try:
        from drive_fetch import (get_drive_service, fetch_all_historical_logs,
                                 download_file, ADJUSTMENTS_ID)
    except ImportError as e:
        print(f"ERROR: Drive fetch requires drive_fetch.py + google-api-python-client: {e}",
              file=sys.stderr)
        print(f"To use pre-downloaded xlsx files instead, pass --logs-dir PATH",
              file=sys.stderr)
        sys.exit(1)
    tmp_dir = tempfile.mkdtemp(prefix="historical-logs-")
    print(f"[freeze] fetching {len(list(HISTORICAL_YEARS))} logs from Drive -> {tmp_dir}")
    service = get_drive_service()
    paths = fetch_all_historical_logs(service, HISTORICAL_YEARS, tmp_dir)
    for year, path in sorted(paths.items()):
        print(f"[freeze]   {year}: {os.path.basename(path)}")
    adj_path = os.path.join(tmp_dir, "adjustments.xlsx")
    print(f"[freeze] fetching adjustments.xlsx (Max's Running Data) -> {tmp_dir}")
    download_file(service, ADJUSTMENTS_ID, adj_path)
    return tmp_dir, adj_path


def resolve_adjustments_path(logs_dir, explicit_path=None):
    """Find the adjustments xlsx. Returns the path or None if not found."""
    if explicit_path:
        if not os.path.exists(explicit_path):
            print(f"[freeze] ERROR: --adjustments-xlsx {explicit_path} does not exist",
                  file=sys.stderr)
            sys.exit(1)
        return explicit_path
    candidates = ["adjustments.xlsx", "Max's Running Data.xlsx",
                  "Maxs_Running_Data.xlsx"]
    for name in candidates:
        p = os.path.join(logs_dir, name)
        if os.path.exists(p):
            return p
    return None


def load_hill_lookup(adj_xlsx_path):
    """Load the hills tab from the adjustments xlsx into the {abbrev: location} dict."""
    if adj_xlsx_path is None:
        return {}
    from snapshot import adjustments_dfs_from_xlsx
    _, _, _, hills_df = adjustments_dfs_from_xlsx(adj_xlsx_path)
    return ingest_hills_from_df(hills_df)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--logs-dir",
                   help="Directory containing the xlsx logs. If omitted, fetches all "
                        "'Running Log YYYY' files + adjustments xlsx from Drive.")
    p.add_argument("--adjustments-xlsx",
                   help="Path to Max's Running Data xlsx. Used only with --logs-dir; "
                        "if omitted, tries common names in --logs-dir.")
    p.add_argument("--out", default=DEFAULT_OUT,
                   help=f"Output CSV path (default: {DEFAULT_OUT})")
    args = p.parse_args()

    if args.logs_dir:
        logs_dir = args.logs_dir
        adj_path = resolve_adjustments_path(logs_dir, args.adjustments_xlsx)
        if adj_path is None:
            print("[freeze] WARNING: no adjustments xlsx found — 2016-17 hill workouts "
                  "will fall back to the default location rule.", file=sys.stderr)
    else:
        logs_dir, adj_path = fetch_logs_from_drive()

    hill_lookup = load_hill_lookup(adj_path)
    if hill_lookup:
        print(f"[freeze] loaded hills lookup: {len(hill_lookup)} abbrev->location entries")
    else:
        print("[freeze] hills lookup is empty — 2016-17 hill workouts will use default rule")

    all_rows = []
    for year in HISTORICAL_YEARS:
        try:
            path = resolve_log_path(logs_dir, year)
        except FileNotFoundError as e:
            print(f"[freeze] ERROR: {e}", file=sys.stderr)
            sys.exit(1)
        print(f"[freeze] ingesting {year}: {os.path.basename(path)}")
        if year == 2016:
            all_rows.extend(ingest_2016(path, hill_lookup=hill_lookup))
        else:
            all_rows.extend(ingest_year_standard(path, year, hill_lookup=hill_lookup))

    all_rows.sort(key=lambda r: r["date"])
    df = pd.DataFrame(all_rows, columns=DAILY_COLUMNS)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)

    # ---------- validation ----------
    print()
    print("=" * 72)
    print("HISTORICAL DAILY — VALIDATION")
    print("=" * 72)
    print(f"Rows: {len(df)}")
    print(f"Date range: {df['date'].min()} → {df['date'].max()}")
    print()
    print(f"{'year':>6} {'parsed':>10} {'auth':>10} {'diff':>8} {'run_days':>9}")
    worst = 0.0
    for yr in sorted(df["year"].unique()):
        sub = df[df["year"] == yr]
        parsed = sub["miles"].sum()
        auth = AUTHORITATIVE_TOTALS.get(yr)
        diff = parsed - auth if auth is not None else None
        run_days = len(sub)
        auth_str = f"{auth:>10.1f}" if auth is not None else " " * 10
        diff_str = f"{diff:>8.2f}" if diff is not None else " " * 8
        print(f"{yr:>6} {parsed:>10.2f} {auth_str} {diff_str} {run_days:>9}")
        if diff is not None:
            worst = max(worst, abs(diff))
    print()
    print(f"(daily.csv is running-only; zero-mile rest/cross-train days "
          f"are pruned at ingest.)")
    print()
    if worst > 0.05:
        print(f"WARNING: worst annual diff is {worst:.2f} mi — investigate before shipping!")
        print("(If you've retroactively corrected race times, update AUTHORITATIVE_TOTALS "
              "in running_log_parser.py to match your new Lifetime Miles totals.)")
        sys.exit(2)
    print(f"All annual totals match Lifetime Miles within 0.05 mi.")

    # ---------- 2016-17 location synthesis summary ----------
    sub_1617 = df[df["year"].isin([2016, 2017])]
    if len(sub_1617):
        loc_filled = sub_1617["location"].notna().sum()
        loc_blank = len(sub_1617) - loc_filled
        print()
        print("2016-17 location synthesis:")
        print(f"  filled: {loc_filled}    blank (race rows): {loc_blank}")
        print("  by inferred location:")
        for loc, n in sub_1617["location"].value_counts(dropna=False).items():
            label = "(none — race)" if loc is None or pd.isna(loc) else loc
            print(f"    {label:<25s} {n:>5d}")

    print()
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
