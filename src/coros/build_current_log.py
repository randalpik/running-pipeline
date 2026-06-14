"""Turn Coros activity details into the pipeline's ``current_log`` rows.

Each Coros activity is one workout; the running log is one row per day. So we
extract the fields we trust from every Run / Track Run, group by local date,
and synthesize a ``workout_raw`` string the existing parser understands:

  - single run, moving >80 min        -> ``long@<m:ss>``
  - single run otherwise              -> ``rec@<m:ss>``
  - day containing a Track Run        -> ``<wu>j, <track_m>f@<m:ss>, <cd>j``
    (runs before/after the track = warmup/cooldown jogs; the track run is
    coded as a continuous fartlek — see project notes for why reps aren't
    reconstructed)
  - multiple plain runs (doubles)     -> combined ``rec@``/``long@``

Sleep, partners, conditions, shoes and weight are out of scope for watch
imports and left blank. Output columns match snapshot.CURRENT_LOG_COLUMNS.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone

import pandas as pd

from src.coros import mappings as M
from src.coros.solar import time_of_day
from src.parsers.snapshot import CURRENT_LOG_COLUMNS


# Fields kept in the slim per-activity cache record. The raw activity detail
# is ~1.5 MB (the per-second frequencyList alone is ~1.49 MB and we use none of
# it past the first GPS fix), so we project to these scalars (~600 B) before
# caching. Values stay RAW (unscaled) so current_log can be re-derived if a
# scaling/mapping changes. See sync.py for the cache.
_SUMMARY_KEYS = ("sportType", "distance", "workoutTime", "totalTime",
                 "startTimestamp", "timezone", "name", "avgPace", "elevGain")
_WEATHER_KEYS = ("temperature", "windSpeed", "windDirection", "weatherType",
                 "humidity")


def slim_detail(d: dict) -> dict:
    """Project a full (or already-slim/rich) activity detail to the slim record.

    Idempotent: a record that's already slim or rich (has 'gps', lacks
    'frequencyList') is returned unchanged, so callers can pass any shape —
    rich records are a slim-compatible superset.
    """
    if "gps" in d and "frequencyList" not in d:
        return d
    s = d.get("summary") or {}
    w = d.get("weather") or {}
    lat, lon = _first_gps(d.get("frequencyList") or [])
    return {
        "summary": {k: s.get(k) for k in _SUMMARY_KEYS},
        "weather": {k: w.get(k) for k in _WEATHER_KEYS},
        "gps": [lat, lon],
    }


def rich_detail(d: dict):
    """Project a full activity detail to the rich record (slim + per-second).

    A slim-compatible superset (every slim consumer accepts it) adding the
    fields the rep-extraction layer (reps.py) needs, RAW/unscaled like slim:

      pauses: [[startTimestamp, endTimestamp, duration], ...]
      freq:   [[timestamp, distance, heart, gpsLat, gpsLon, altitude, speed],
               ...]  per-second

    ~100 KB vs slim's ~600 B, so it's kept only where the stream is consumed:
    reps.py (every Track Run — sync caches those rich from the start — and
    runs on hand-logged workout days, scripts/backfill_rich_details.py) and
    the elevation enrichment (altitude/speed; scripts/backfill_elevation.py).

    Schema version (``rich``):
      1 — freq points are [t, dist, heart, gpsLat, gpsLon] (pre-2026-06).
      2 — appends [altitude, speed] (raw: altitude is meters, speed is the
          Coros raw value). ``altitude``/``speed`` are None on devices that
          don't report them. Consumers index by position, so v2 is a strict
          superset — f[0..4] are unchanged.

    ``altitude`` enables the Minetti per-run grade correction; ``speed`` is
    kept for corrected-mile split pace (informational) and a possible future
    pace-vs-grade physiology model. Split pace is computed from the distance
    + timestamp streams, not this raw speed (whose scaling is unverified).

    Idempotent on rich records (returns as-is — a v1 record is NOT upgraded
    in place; the frequencyList it was built from is gone, so altitude
    requires a re-fetch). Returns None for a slim record.
    """
    if "freq" in d:
        return d
    if "frequencyList" not in d:
        return None
    rec = slim_detail(d)
    rec["rich"] = 2
    rec["pauses"] = [[p.get("startTimestamp"), p.get("endTimestamp"),
                      p.get("duration")] for p in (d.get("pauseList") or [])]
    rec["freq"] = [[p.get("timestamp"), p.get("distance"), p.get("heart"),
                    p.get("gpsLat"), p.get("gpsLon"),
                    p.get("altitude"), p.get("speed")]
                   for p in (d.get("frequencyList") or [])]
    return rec


class Activity:
    """One Coros activity with scaling applied to real units."""

    __slots__ = ("sport_type", "distance_m", "moving_s", "total_s",
                 "start_utc", "tz_min", "lat", "lon", "temp_c",
                 "wind_ms", "weather_type")

    def __init__(self, record: dict):
        rec = slim_detail(record)
        s = rec["summary"]
        self.sport_type = int(s.get("sportType"))
        self.distance_m = _num(s.get("distance")) / M.DISTANCE_DIV
        self.moving_s = _num(s.get("workoutTime")) / M.TIME_DIV
        self.total_s = _num(s.get("totalTime")) / M.TIME_DIV
        self.start_utc = datetime.fromtimestamp(
            _num(s.get("startTimestamp")) / M.TIMESTAMP_DIV, tz=timezone.utc)
        self.tz_min = int(_num(s.get("timezone"))) * M.TZ_UNIT_MIN
        gps = rec.get("gps") or [None, None]
        self.lat = gps[0] / M.GPS_DIV if gps[0] not in (None, 0) else None
        self.lon = gps[1] / M.GPS_DIV if gps[1] not in (None, 0) else None
        w = rec["weather"]
        self.temp_c = (_num(w.get("temperature")) / M.WEATHER_DIV
                       if w.get("temperature") is not None else None)
        self.wind_ms = (_num(w.get("windSpeed")) / M.WEATHER_DIV
                        if w.get("windSpeed") is not None else None)
        self.weather_type = w.get("weatherType")

    @property
    def local_date(self):
        return self.start_utc.astimezone(
            timezone(timedelta(minutes=self.tz_min))).date()

    @property
    def is_indoor(self) -> bool:
        return self.sport_type == M.SPORT_INDOOR_RUN


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _first_gps(freq_list):
    """First valid (gpsLat, gpsLon) as RAW ints (scaling applied in Activity)."""
    for p in freq_list:
        lat, lon = p.get("gpsLat"), p.get("gpsLon")
        if lat not in (None, 0) and lon not in (None, 0):
            return lat, lon
    return None, None


def _fmt_pace(moving_s: float, meters: float) -> str | None:
    """Seconds-per-mile as 'm:ss', from moving time and distance."""
    if meters <= 0 or moving_s <= 0:
        return None
    spm = round(moving_s * M.METERS_PER_MILE / meters)
    return f"{spm // 60}:{spm % 60:02d}"


def _workout_raw(runs: list[Activity]) -> str | None:
    """Compose the workout string for one day's run activities (time-ordered)."""
    track_idx = [i for i, a in enumerate(runs)
                 if a.sport_type == M.SPORT_TRACK_RUN]
    if track_idx:
        first, last = track_idx[0], track_idx[-1]
        warmup = runs[:first]
        track = runs[first:last + 1]
        cooldown = runs[last + 1:]
        track_m = sum(a.distance_m for a in track)
        track_s = sum(a.moving_s for a in track)
        pace = _fmt_pace(track_s, track_m)
        if pace is None:
            return None
        parts = []
        wu = round(sum(a.moving_s for a in warmup) / 60)
        if wu > 0:
            parts.append(f"{wu}j")
        parts.append(f"{round(track_m)}f@{pace}")
        cd = round(sum(a.moving_s for a in cooldown) / 60)
        if cd > 0:
            parts.append(f"{cd}j")
        return ", ".join(parts)

    # no track run: recovery, or long if any single run's moving time > 80 min
    total_m = sum(a.distance_m for a in runs)
    total_s = sum(a.moving_s for a in runs)
    pace = _fmt_pace(total_s, total_m)
    if pace is None:
        return None
    longest = max(a.moving_s for a in runs)
    prefix = "long" if longest > M.LONG_RUN_MIN_SECONDS else "rec"
    return f"{prefix}@{pace}"


def build_current_log(details, *, geocode=True):
    """Build the current_log DataFrame from a list of activity-detail dicts.

    Returns (DataFrame, meta) where meta carries diagnostics — notably the
    distinct weatherType values seen, for tuning the weather bin map.
    """
    acts = [Activity(d) for d in details if (d.get("summary") or {}).get("sportType") is not None]
    runs_by_day: dict = defaultdict(list)
    for a in acts:
        if a.sport_type in M.RUN_SPORTS:
            runs_by_day[a.local_date].append(a)

    # Batch reverse-geocode each day's representative (first) run start.
    rep_points = {}
    for day, runs in runs_by_day.items():
        runs.sort(key=lambda a: a.start_utc)
        rep = runs[0]
        if not rep.is_indoor and rep.lat is not None:
            rep_points[day] = (rep.lat, rep.lon)
    geo = {}
    if geocode and rep_points:
        from src.shared.geocoding import reverse_geocode
        resolved = reverse_geocode(list(rep_points.values()))
        geo = {day: resolved.get(pt) for day, pt in rep_points.items()}

    weather_types = Counter()
    rows = []
    for day in sorted(runs_by_day):
        runs = runs_by_day[day]
        rep = runs[0]
        if rep.weather_type is not None:
            weather_types[rep.weather_type] += 1
        miles = sum(a.distance_m for a in runs) / M.METERS_PER_MILE
        minutes = sum(a.moving_s for a in runs) / 60.0
        weather = "indoors" if rep.is_indoor else M.weather_bin(rep.weather_type)
        rows.append({
            "date": day.isoformat(),
            "sleep_cycles": None,
            "miles": round(miles, 2),
            "minutes": round(minutes, 1),
            "temp_c": None if rep.is_indoor else _round_or_none(rep.temp_c, 1),
            "weather": weather,
            "workout_raw": _workout_raw(runs),
            "partners": None,
            "conditions": None,
            "wind": None if rep.is_indoor else M.wind_bin(rep.wind_ms),
            "time_of_day": time_of_day(rep.start_utc,
                                       None if rep.is_indoor else rep.lat,
                                       None if rep.is_indoor else rep.lon,
                                       rep.tz_min),
            "shoes": None,
            "location": geo.get(day),
            "weight_lbs": None,
        })

    df = pd.DataFrame(rows, columns=CURRENT_LOG_COLUMNS)
    meta = {
        "activities": len(acts),
        "run_activities": sum(len(v) for v in runs_by_day.values()),
        "days": len(rows),
        "weather_types_seen": dict(weather_types),
    }
    return df, meta


def _round_or_none(v, ndigits):
    return None if v is None else round(v, ndigits)
