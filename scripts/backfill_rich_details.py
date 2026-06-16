"""Backfill rich detail records into a Coros details cache.

The sync layer caches Track Runs rich (per-second stream) going forward; this
script upgrades the records that were cached slim before that existed, plus
plain Runs on hand-logged workout days (road workouts need the stream too).

Targets: every Track Run (sportType 103), and — when --daily is given — every
run activity whose local date carries a hand-logged quality workout or a
continuous-hill workout (hill_cont; all of that day's runs are targeted since
the loop block may be merged with a jog).

Full details come from --import-dir when a previously fetched raw JSON is
available there (no API call), otherwise from the Coros API via the cached
token (re-login needs COROS_EMAIL/COROS_PASSWORD-style env credentials).

Usage (Max's watch cache + hand log):
  python scripts/backfill_rich_details.py \
      --details-dir data/profiles/coros/details \
      --token-cache data/profiles/coros/coros_token.json \
      --daily data/daily.csv [--import-dir /tmp/coros_full/coros] [--dry-run]
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.coros import mappings as M
from src.coros.build_current_log import Activity, rich_detail
from src.coros.client import CorosClient

QUALITY_TYPES = {"tempo", "interval", "rep", "fartlek"}


def quality_dates(daily_path):
    daily = pd.read_csv(daily_path)
    q = daily[daily["run_type"].isin(QUALITY_TYPES)]
    return set(pd.to_datetime(q["date"]).dt.date.astype(str))


def hill_dates(daily_path):
    daily = pd.read_csv(daily_path)
    h = daily[daily["run_type"] == "hill_cont"]
    return set(pd.to_datetime(h["date"]).dt.date.astype(str))


def main():
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    p.add_argument("--details-dir", required=True, type=Path)
    p.add_argument("--token-cache", type=Path, default=None)
    p.add_argument("--region", default="us")
    p.add_argument("--email", default="")
    p.add_argument("--password", default="")
    p.add_argument("--daily", type=Path, default=None,
                   help="daily.csv whose quality days mark plain Runs for "
                        "enrichment (Track Runs are always targets).")
    p.add_argument("--import-dir", type=Path, default=None,
                   help="Directory of previously fetched RAW detail JSONs "
                        "(<labelId>.json) to convert without API calls.")
    p.add_argument("--dates", default="",
                   help="Comma-separated YYYY-MM-DD list; restrict targets "
                        "to these local dates (probe runs).")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    qdates = quality_dates(args.daily) if args.daily else set()
    hdates = hill_dates(args.daily) if args.daily else set()
    only_dates = {s.strip() for s in args.dates.split(",") if s.strip()}
    client = None

    # inventory pass: which dates have a Track Run (their plain runs are
    # warmups/cooldowns reps.py never consumes — don't enrich those)
    records, track_dates = [], set()
    for path in sorted(args.details_dir.glob("*.json")):
        rec = json.loads(path.read_text())
        if (rec.get("summary") or {}).get("sportType") is None:
            continue
        act = Activity(rec)
        if act.sport_type not in M.RUN_SPORTS:
            continue
        records.append((path, rec, act))
        if act.sport_type == M.SPORT_TRACK_RUN:
            track_dates.add(act.local_date.isoformat())

    targets, already, upgraded, fetched, failed = 0, 0, 0, 0, []
    for path, rec, act in records:
        date = act.local_date.isoformat()
        # Hill days have no Track Run and the loop block may hide in any of
        # the day's activities (sometimes merged with a jog) — target them all.
        is_target = (act.sport_type == M.SPORT_TRACK_RUN
                     or (date in qdates and date not in track_dates)
                     or date in hdates)
        if not is_target:
            continue
        if only_dates and date not in only_dates:
            continue
        targets += 1
        if "freq" in rec:
            already += 1
            continue
        if args.dry_run:
            print(f"would enrich {path.stem} ({act.local_date} "
                  f"sport={act.sport_type})")
            continue

        raw = None
        if args.import_dir:
            src = args.import_dir / path.name
            if src.exists():
                cand = json.loads(src.read_text())
                if "frequencyList" in cand:
                    raw = cand
        if raw is None:
            if client is None:
                client = CorosClient(args.email, args.password,
                                     region=args.region,
                                     token_cache=args.token_cache)
            try:
                raw = client.activity_detail(path.stem, act.sport_type)
                time.sleep(0.25)
                fetched += 1
            except Exception as exc:           # keep going; report at the end
                failed.append((path.stem, str(exc)[:80]))
                continue

        rich = rich_detail(raw)
        if rich is None:
            failed.append((path.stem, "source has no frequencyList"))
            continue
        path.write_text(json.dumps(rich))
        upgraded += 1

    print(f"targets={targets} already-rich={already} upgraded={upgraded} "
          f"(api-fetched={fetched}) failed={len(failed)}")
    for label, why in failed:
        print(f"  FAILED {label}: {why}")


if __name__ == "__main__":
    main()
