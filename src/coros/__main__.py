"""CLI: fetch Coros activities for a date range and emit a current_log CSV.

    python -m src.coros --from 20260101 --to 20260529 --out data/coros_current_log.csv

Credentials come from the environment (COROS_EMAIL / COROS_PASSWORD) so they
never land in argv or on disk. Activity details are optionally cached as JSON
(--cache-dir) to make dev re-runs free and avoid re-hitting the API.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from src.coros import mappings as M
from src.coros.build_current_log import build_current_log, slim_detail
from src.coros.client import CorosClient


def _fetch_details(client, from_day, to_day, cache_dir):
    details = []
    for item in client.iter_activities(from_day=from_day, to_day=to_day):
        sport = item.get("sportType")
        if sport not in M.RUN_SPORTS:
            continue
        label = str(item.get("labelId"))
        cached = cache_dir / f"{label}.json" if cache_dir else None
        if cached and cached.exists():
            raw = json.loads(cached.read_text())
            data = slim_detail(raw)
            if "frequencyList" in raw:        # migrate legacy full cache file
                cached.write_text(json.dumps(data))
        else:
            data = slim_detail(client.activity_detail(label, sport))
            if cached:
                cache_dir.mkdir(parents=True, exist_ok=True)
                cached.write_text(json.dumps(data))
        details.append(data)
        print(f"  fetched {label} sport={sport} "
              f"name={item.get('name')!r} date={item.get('date')}")
    return details


def main(argv=None):
    p = argparse.ArgumentParser(description=(__doc__ or "").strip().split("\n")[0])
    p.add_argument("--from", dest="from_day", required=True, help="YYYYMMDD")
    p.add_argument("--to", dest="to_day", required=True, help="YYYYMMDD")
    p.add_argument("--out", required=True, help="output current_log CSV path")
    p.add_argument("--region", default="us", choices=["us", "eu"])
    p.add_argument("--token-cache", default="data/coros_token.json")
    p.add_argument("--cache-dir", default=None,
                   help="optional dir to cache activity-detail JSON")
    p.add_argument("--no-geocode", action="store_true",
                   help="skip reverse geocoding (leave location blank)")
    args = p.parse_args(argv)

    email = os.environ.get("COROS_EMAIL")
    password = os.environ.get("COROS_PASSWORD")
    if not email or not password:
        print("error: set COROS_EMAIL and COROS_PASSWORD in the environment",
              file=sys.stderr)
        return 2

    client = CorosClient(email, password, region=args.region,
                         token_cache=args.token_cache)
    cache_dir = Path(args.cache_dir) if args.cache_dir else None

    print(f"[coros] fetching {args.from_day}..{args.to_day}")
    details = _fetch_details(client, args.from_day, args.to_day, cache_dir)

    df, meta = build_current_log(details, geocode=not args.no_geocode)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(f"\n[coros] {meta['run_activities']} run activities "
          f"-> {meta['days']} days -> {out}")
    if meta["weather_types_seen"]:
        print(f"[coros] weatherType values seen (tune mappings.py): "
              f"{meta['weather_types_seen']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
