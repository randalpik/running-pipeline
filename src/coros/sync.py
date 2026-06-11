"""Incremental sync of a Coros profile's current_log.

Past activity days are immutable, so we never re-fetch them: the activity
*detail* JSON is cached per labelId under ``details_dir`` (persisted across
runs), and the built ``current_log`` CSV accumulates day rows. Each sync:

  1. reads the last day already in the current_log,
  2. lists activities from that day forward only (the boundary day is
     re-listed so a late same-day run is picked up),
  3. fetches details only for labelIds not already cached,
  4. rebuilds just the affected days and merges them over the persisted log.

So after the one-time backfill, a sync is a single list call plus details for
genuinely new activities. ``rebuild=True`` reprocesses every cached detail
without re-fetching — used when a mapping in mappings.py changes.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd

from src.coros import mappings as M
from src.coros.build_current_log import build_current_log, rich_detail, slim_detail
from src.coros.client import CorosClient
from src.parsers.snapshot import CURRENT_LOG_COLUMNS


def _project(raw, sport_type):
    """Slim or rich projection: Track Runs keep the per-second data that the
    rep-extraction layer (reps.py) consumes; everything else stays slim."""
    if sport_type == M.SPORT_TRACK_RUN:
        return rich_detail(raw) or slim_detail(raw)
    return slim_detail(raw)


def _cache_detail(client, item, details_dir: Path):
    """Return the cached detail record, fetching + caching on miss.

    The cache stores a projection — slim (~600 B) for plain runs, rich
    (~100 KB, + per-second stream) for Track Runs — never the ~1.5 MB raw
    detail. A cache file that predates slimming (full detail) is migrated in
    place on read.
    """
    label = str(item.get("labelId"))
    path = details_dir / f"{label}.json"
    if path.exists():
        raw = json.loads(path.read_text())
        rec = _project(raw, item.get("sportType"))
        if "frequencyList" in raw:           # legacy full file -> migrate in place
            path.write_text(json.dumps(rec))
        return rec
    rec = _project(client.activity_detail(label, item.get("sportType")),
                   item.get("sportType"))
    details_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec))
    return rec


def _load_all_details(details_dir: Path):
    """Load every cached detail record, migrating legacy full files."""
    out = []
    for p in sorted(details_dir.glob("*.json")):
        try:
            raw = json.loads(p.read_text())
        except (ValueError, OSError):
            continue
        rec = _project(raw, (raw.get("summary") or {}).get("sportType"))
        if "frequencyList" in raw:
            p.write_text(json.dumps(rec))
        out.append(rec)
    return out


def sync_current_log(*, email, password, region, current_log_path, details_dir,
                     token_cache=None, rebuild=False, start_day=None,
                     geocode=True):
    """Sync and persist the current_log CSV. Returns the merged DataFrame.

    start_day ('YYYYMMDD' or None) bounds the first backfill; ignored once a
    persisted log exists (we resume from its last day) unless ``rebuild``.
    """
    current_log_path = Path(current_log_path)
    details_dir = Path(details_dir)

    persisted = None
    if current_log_path.exists() and not rebuild:
        persisted = pd.read_csv(current_log_path)

    if rebuild:
        from_day = start_day
    elif persisted is not None and len(persisted):
        last = pd.to_datetime(persisted["date"]).dt.date.max()
        from_day = last.strftime("%Y%m%d")        # inclusive: re-list boundary day
    else:
        from_day = start_day

    client = CorosClient(email, password, region=region, token_cache=token_cache)

    if rebuild:
        details = _load_all_details(details_dir)
        print(f"[coros-sync] rebuild from {len(details)} cached details")
    else:
        fetched = 0
        details = []
        for item in client.iter_activities(from_day=from_day):
            if item.get("sportType") not in M.RUN_SPORTS:
                continue
            label = str(item.get("labelId"))
            had = (details_dir / f"{label}.json").exists()
            details.append(_cache_detail(client, item, details_dir))
            fetched += 0 if had else 1
        print(f"[coros-sync] listed from {from_day or 'beginning'}: "
              f"{len(details)} run activities, {fetched} newly fetched")

    new_df, meta = build_current_log(details, geocode=geocode)

    if persisted is not None and len(persisted) and not rebuild and from_day:
        boundary = pd.to_datetime(from_day, format="%Y%m%d").date().isoformat()
        kept = persisted[persisted["date"] < boundary]
        merged = pd.concat([kept, new_df], ignore_index=True)
    else:
        merged = new_df
    merged = (merged.sort_values("date")
                    .drop_duplicates("date", keep="last")
                    .reset_index(drop=True))
    merged = merged.reindex(columns=CURRENT_LOG_COLUMNS)

    current_log_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(current_log_path, index=False)
    print(f"[coros-sync] current_log: {len(merged)} days "
          f"({merged['date'].min()} → {merged['date'].max()}) -> {current_log_path}")
    if meta["weather_types_seen"]:
        print(f"[coros-sync] weatherType seen this pass: {meta['weather_types_seen']}")
    return merged
