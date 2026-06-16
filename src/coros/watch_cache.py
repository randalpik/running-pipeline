"""Presence-based incremental loading of a Coros detail cache.

Past activities are immutable, so a per-day derived table only needs refreshing
where the *set* of cached activities changed. `plan()` finds, by pure
set-difference on labelIds (filenames — no stream parse), which days a derived
table must (re)derive, and parses ONLY those days' detail files. A no-op build
parses nothing. See docs/watch-derived-cache-spec.md.

The cached derived table records, per day, the labelIds it consumed (run AND
non-run files present that day), so steady-state builds find "nothing new" and
skip without opening the ~339 MB of per-second streams.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from src.coros import mappings as M


def label_files(cache_dir):
    """{labelId: Path} for cached, non-excluded detail files (no parse)."""
    return {p.stem: p for p in cache_dir.glob("*.json")
            if p.stem not in M.EXCLUDED_LABEL_IDS}


def _read(path):
    return json.loads(path.read_text())


def _local_date(detail):
    """ISO local date of an activity from its summary (no Activity build)."""
    s = detail.get("summary") or {}
    ts = s.get("startTimestamp")
    if ts is None:
        return None
    tz = int(s.get("timezone") or 0) * M.TZ_UNIT_MIN
    return (datetime.fromtimestamp(ts / M.TIMESTAMP_DIV, tz=timezone.utc)
            .astimezone(timezone(timedelta(minutes=tz))).date().isoformat())


def plan(existing_ids_by_date, cache_dir, *, full=False):
    """Decide what to (re)derive.

    existing_ids_by_date: {date_iso: set(labelId)} the cached table consumed.
    Returns (targets, dropped, present_ids):
      targets    {date_iso: [(labelId, detail_dict), ...]} — days to derive;
                 only these files are parsed. Each list is the FULL present file
                 set for that day (so a day that gained a late second run is
                 re-derived from all its activities).
      dropped    {date_iso} whose activities all disappeared — drop the row.
      present_ids set of labelIds currently in the cache.
    No-op (nothing new or gone, not `full`): targets and dropped are empty.
    """
    present = label_files(cache_dir)
    present_ids = set(present)

    if full or not existing_ids_by_date:
        targets = {}
        for lid, p in present.items():
            d = _read(p)
            day = _local_date(d)
            if day is not None:
                targets.setdefault(day, []).append((lid, d))
        return targets, set(), present_ids

    consumed = set().union(*existing_ids_by_date.values())
    new = present_ids - consumed
    gone = consumed - present_ids
    if not new and not gone:
        return {}, set(), present_ids

    affected = set()
    new_ids_by_date = {}
    for lid in new:
        day = _local_date(_read(present[lid]))
        if day is None:
            continue
        new_ids_by_date.setdefault(day, set()).add(lid)
        affected.add(day)

    dropped = set()
    for day, ids in existing_ids_by_date.items():
        if ids & gone:                       # this day lost an activity
            if (ids & present_ids) or day in new_ids_by_date:
                affected.add(day)            # re-derive from what remains
            else:
                dropped.add(day)             # nothing left — drop the row

    targets = {}
    for day in affected:
        ids = (set(existing_ids_by_date.get(day, set())) & present_ids) \
              | new_ids_by_date.get(day, set())
        targets[day] = [(i, _read(present[i])) for i in sorted(ids)]
    return targets, dropped, present_ids
