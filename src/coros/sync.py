"""Incremental sync of a Coros profile's current_log.

Past activity days are immutable, so we never re-FETCH them: the activity
*detail* JSON is cached per labelId under ``details_dir`` (persisted across
runs). The cache is the sole persisted source of truth — the ``current_log``
CSV is a pure DERIVATION of it, rebuilt in full every sync. Each sync:

  1. reads the last day already in the current_log,
  2. lists activities from that day forward only (the boundary day is
     re-listed so a late same-day run is picked up),
  3. fetches details only for labelIds not already cached,
  4. rebuilds the WHOLE current_log from the complete local cache.

So after the one-time backfill, a sync is a single list call plus details for
genuinely new activities, then a local rebuild. ``rebuild=True`` skips the API
entirely and rebuilds from cache — used offline or when a mapping changes.

Rebuilding the whole log (rather than merging only the new days over the
persisted CSV) is deliberate: it costs only local CPU + cached geocoding, and
it means every historical row always reflects the CURRENT extraction/mappings.
The previous merge-the-tail approach froze historical rows at whatever schema
built them, so later additions (e.g. wind_mph / humidity_pct) never reached
days already in the log — they stayed blank forever on the persisted cache,
including in CI. Deriving from the cache each run makes that class of staleness
impossible.
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
    """Rich or slim projection. Every OUTDOOR run keeps the per-second stream
    (rich): Track Runs for rep extraction (reps.py), and Run/Trail for the
    elevation grade enrichment + long-run stall/segment detection — so the
    regular pipeline derives those without ever re-fetching. Indoor treadmill
    has no spatial stream worth keeping, so it stays slim."""
    if sport_type in (M.SPORT_RUN, M.SPORT_TRAIL_RUN, M.SPORT_TRACK_RUN):
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


def _upgrade_slim_runs(client, details_dir: Path):
    """Re-fetch cached outdoor-run records still stored slim, upgrading them to
    rich (per-second stream) so the elevation/altitude enrichment has data.

    Outdoor runs are cached rich by ``_project``; a slim one predates rich
    caching (an old backfill, or a profile whose history was fetched before the
    rich logic existed). Indoor runs legitimately stay slim and are skipped.
    One-time per record — once rich it's a hit and never re-fetched — so this
    self-heals a stale cache (e.g. CI's persisted cache) without a manual
    backfill step. Returns the count upgraded."""
    upgraded = 0
    for p in sorted(details_dir.glob("*.json")):
        if p.stem in M.EXCLUDED_LABEL_IDS:
            continue
        try:
            rec = json.loads(p.read_text())
        except (ValueError, OSError):
            continue
        if "freq" in rec or "frequencyList" in rec:
            continue                              # already rich (or raw)
        sport = (rec.get("summary") or {}).get("sportType")
        if sport not in (M.SPORT_RUN, M.SPORT_TRAIL_RUN, M.SPORT_TRACK_RUN):
            continue                              # indoor etc. — slim is correct
        print(f"[coros-sync] upgrading slim record {p.stem} to rich "
              f"(#{upgraded + 1})…", flush=True)
        new = rich_detail(client.activity_detail(p.stem, sport)) or rec
        if "freq" in new:
            p.write_text(json.dumps(new))
            upgraded += 1
    if upgraded:
        print(f"[coros-sync] upgraded {upgraded} slim run record(s) to rich")
    return upgraded


def _load_all_details(details_dir: Path):
    """Load every cached detail record, migrating legacy full files."""
    out = []
    for p in sorted(details_dir.glob("*.json")):
        if p.stem in M.EXCLUDED_LABEL_IDS:
            continue
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

    # Incrementally fetch + cache any new activity details (skipped on rebuild).
    # Only the cache is updated here; the current_log is rebuilt from it below.
    if not rebuild:
        if current_log_path.exists():
            persisted = pd.read_csv(current_log_path)
            from_day = (pd.to_datetime(persisted["date"]).dt.date.max()
                        .strftime("%Y%m%d") if len(persisted) else start_day)
        else:
            from_day = start_day                   # inclusive: re-list boundary day
        client = CorosClient(email, password, region=region,
                             token_cache=token_cache)
        # Announce work BEFORE doing it: this loop is otherwise silent until
        # done, so a cold cache (or a slow Coros API) looks like a hang from
        # the outside — CI watched it for minutes with no output (Aug 2026).
        print(f"[coros-sync] listing activities from "
              f"{from_day or 'beginning'}…", flush=True)
        fetched = 0
        for item in client.iter_activities(from_day=from_day):
            if item.get("sportType") not in M.RUN_SPORTS:
                continue
            label = str(item.get("labelId"))
            had = (details_dir / f"{label}.json").exists()
            if not had:
                print(f"[coros-sync] fetching detail {label} "
                      f"({item.get('date', '?')}, #{fetched + 1})…",
                      flush=True)
            _cache_detail(client, item, details_dir)
            fetched += 0 if had else 1
        print(f"[coros-sync] listed from {from_day or 'beginning'}: "
              f"{fetched} newly fetched")
        # Self-heal a cache whose outdoor-run records are still slim (no
        # per-second stream) — e.g. CI's persisted cache or a pre-rich history.
        # Needed for the elevation/altitude enrichment; one-time per record.
        _upgrade_slim_runs(client, details_dir)
    else:
        print("[coros-sync] rebuild: skipping API fetch")

    # The current_log is ALWAYS the full derivation of the local detail cache,
    # so every row reflects the current extraction (no stale historical rows).
    details = _load_all_details(details_dir)
    merged, meta = build_current_log(details, geocode=geocode)
    merged = (merged.sort_values("date")
                    .drop_duplicates("date", keep="last")
                    .reset_index(drop=True))
    merged = merged.reindex(columns=CURRENT_LOG_COLUMNS)
    print(f"[coros-sync] rebuilt current_log from {len(details)} cached details")

    current_log_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(current_log_path, index=False)
    print(f"[coros-sync] current_log: {len(merged)} days "
          f"({merged['date'].min()} → {merged['date'].max()}) -> {current_log_path}")
    if meta["weather_types_seen"]:
        print(f"[coros-sync] weatherType seen this pass: {meta['weather_types_seen']}")
    return merged
