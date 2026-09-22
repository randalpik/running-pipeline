"""Canonical local clock-time for a day's running.

**Principle (since June 2026): the hand-logged ``city_state`` is the single
source of truth for a run's location and timezone. The watch supplies only the
absolute MOMENT of each run.** Its reported GPS and ``tz_min`` each fail on
travel days — a failed-GPS fix stamps the home coordinates, and the watch can
report the offset of the zone it last synced in — so neither is trusted for
anything solar. See ``docs/qualitative-trends-reference.md`` for the failures
that motivated this.

Two eras, one output: ``start_min`` / ``end_min`` / ``lat`` / ``lon`` /
``tz_min`` in the canonical city's local frame.

* Watch era — :func:`reproject_watch_to_canonical` recovers each run's UTC
  instant and projects it into the canonical city's IANA zone (DST-correct).
* Pre-watch era — :func:`estimate_binned_time` derives a midpoint from the
  hand-logged ``time_of_day`` bin against the same city's solar times.

Extracted from ``plot_qualitative_trends`` so the Misc. Trends Time panel and
the Recovery / Long Runs "Time of day" colour encoding share one definition.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from src.coros.solar import solar_anchors_local
from src.parsers.snapshot import (find_snapshot, read_snapshot,
                                  coord_overrides_from_sections)


# ---------- canonical city table ----------

def canonical_coords(data_dir):
    """``{city_state: (lat, lon, tz_iana)}`` from ``city_coords.csv``, with the
    snapshot's ``coordinates`` overrides applied on top (the same corrections
    the world map applies via ``ensure_coords``) so a geocoding fix reaches
    every consumer. Read-time only — ``city_coords.csv`` stays regenerable from
    source. Each city's cached tz survives a coordinate fix (a correction almost
    never crosses a timezone). No-op when the snapshot or its coordinates
    section is absent (e.g. a watch-import profile)."""
    cc_path = os.path.join(str(data_dir), 'city_coords.csv')
    coords = {}
    if os.path.exists(cc_path):
        cc = pd.read_csv(cc_path)
        has_tz = 'tz' in cc.columns
        coords = {r['city_state']: (r['latitude'], r['longitude'],
                                    r['tz'] if has_tz and pd.notna(r['tz']) else None)
                  for _, r in cc.iterrows()
                  if pd.notna(r['latitude']) and pd.notna(r['longitude'])}

    snap_path = find_snapshot([os.path.join(str(data_dir), 'drive_snapshot.csv')])
    if snap_path:
        sections, _ = read_snapshot(snap_path)
        for cs, (lat, lon) in coord_overrides_from_sections(sections).items():
            tz = coords[cs][2] if cs in coords else None
            coords[cs] = (lat, lon, tz)
    return coords


# ---------- canonical-location timezone helpers ----------

def _zone(tz_name):
    """ZoneInfo for an IANA name, or None (missing/unknown)."""
    if not tz_name or (isinstance(tz_name, float) and np.isnan(tz_name)):
        return None
    try:
        return ZoneInfo(str(tz_name))
    except Exception:
        return None


def _offset_min_at(zone, d):
    """DST-correct UTC offset (minutes) of ``zone`` at local noon on date ``d``."""
    return int(round(zone.utcoffset(
        datetime(d.year, d.month, d.day, 12)).total_seconds() / 60.0))


def reproject_watch_to_canonical(full, coords):
    """Rewrite the Time band + solar inputs of every watch-era day from the
    CANONICAL city rather than the watch's reported GPS/offset.

    The watch's absolute moment is reliable; its ``tz_min`` (and a failed-GPS
    home-default lat/lon) are not. So we recover each run's UTC instant from its
    stored local-clock minutes + the watch offset it was written with, then
    project that instant into the canonical city's IANA timezone (DST-correct)
    and coordinates. ``start_min``/``end_min`` become canonical local minutes,
    ``lat``/``lon`` the canonical city, ``tz_min`` the canonical offset (which
    the solar gradient consumes). Days whose canonical city has no cached
    coords/tz keep the watch values (graceful fallback)."""
    if 'city_state' not in full:
        return
    have = full['start_min'].notna() & full['tz_min'].notna()
    for ts in full.index[have]:
        cs = full.at[ts, 'city_state']
        info = coords.get(cs) if isinstance(cs, str) else None
        if not info:
            continue
        lat, lon, tzname = info
        zone = _zone(tzname)
        if zone is None or pd.isna(lat) or pd.isna(lon):
            continue
        day = ts.date()
        watch_zone = timezone(timedelta(minutes=float(full.at[ts, 'tz_min'])))
        base = datetime(day.year, day.month, day.day, tzinfo=watch_zone)

        def to_canon(minutes, _b=base, _z=zone, _day=day):
            utc = (_b + timedelta(minutes=float(minutes))).astimezone(timezone.utc)
            loc = utc.astimezone(_z)
            cm = loc.hour * 60 + loc.minute + loc.second / 60.0
            if loc.date() < _day:        # offset pushed it onto an adjacent day
                cm = 0.0
            elif loc.date() > _day:
                cm = 1439.0
            return cm, loc.utcoffset().total_seconds() / 60.0

        s_min, off = to_canon(full.at[ts, 'start_min'])
        e_min, _ = to_canon(full.at[ts, 'end_min'])
        if e_min <= s_min:               # crossed midnight in canonical tz
            e_min = 1439.0
        full.at[ts, 'start_min'] = round(s_min, 1)
        full.at[ts, 'end_min'] = round(e_min, 1)
        full.at[ts, 'lat'] = round(float(lat), 4)
        full.at[ts, 'lon'] = round(float(lon), 4)
        full.at[ts, 'tz_min'] = round(off)


# ---------- pre-watch time-of-day estimate ----------

def _us_dst(ts):
    """US DST window: 2nd Sunday of March 02:00 -> 1st Sunday of November."""
    import datetime as _dt
    y = ts.year
    mar1 = _dt.date(y, 3, 1)
    second_sun_mar = mar1 + _dt.timedelta(days=(6 - mar1.weekday()) % 7 + 7)
    nov1 = _dt.date(y, 11, 1)
    first_sun_nov = nov1 + _dt.timedelta(days=(6 - nov1.weekday()) % 7)
    return second_sun_mar <= ts.date() < first_sun_nov


def _tz_min_from_lon(lon, ts):
    """Approximate local tz offset (minutes) from longitude + US DST rule —
    enough for a bin-derived estimate (keeps the band aligned to its own
    solar gradient; absolute clock may be off where US DST doesn't apply)."""
    return int(round(lon / 15.0)) * 60 + (60 if _us_dst(ts) else 0)


# Bin -> run midpoint, expressed via the day's solar times (local minutes):
#   early -> 1h before sunrise; morning -> midpoint(sunrise, noon);
#   afternoon -> midpoint(noon, sunset); late -> 1h after sunset.
BIN_SHOULDER_MIN = 60.0


def estimate_binned_time(full, coords):
    """Fill start_min/end_min/lat/lon/tz_min on days that have no watch time but
    do have a hand-logged time_of_day bin, a run length, and a geocodable
    location. Returns a bool Series flagging the estimated rows."""
    est = pd.Series(False, index=full.index)
    if not coords or 'time_of_day' not in full or 'city_state' not in full:
        return est
    need = (full['start_min'].isna() & full['time_of_day'].notna()
            & (full['minutes'] > 0) & full['city_state'].notna())
    for ts in full.index[need]:
        ll = coords.get(full.at[ts, 'city_state'])
        if ll is None:
            continue
        lat, lon, tzname = ll
        # Canonical IANA tz (DST-correct) when cached; longitude+US-DST only as
        # a last-resort fallback for a city without a resolved zone.
        zone = _zone(tzname)
        tz = _offset_min_at(zone, ts.date()) if zone else _tz_min_from_lon(lon, ts)
        a = solar_anchors_local(ts.date(), lat, lon, tz)
        sr, noon, ss = a['sunrise'], a['solar_noon'], a['sunset']
        if sr is None or noon is None or ss is None:
            continue
        b = str(full.at[ts, 'time_of_day']).strip().lower()
        if b == 'early':
            mid = sr - BIN_SHOULDER_MIN
        elif b == 'morning':
            mid = (sr + noon) / 2.0
        elif b == 'afternoon':
            mid = (noon + ss) / 2.0
        elif b == 'late':
            mid = ss + BIN_SHOULDER_MIN
        else:
            continue
        half = float(full.at[ts, 'minutes']) / 2.0
        s, e = max(0.0, mid - half), min(1439.0, mid + half)
        if e <= s:
            continue
        full.at[ts, 'start_min'] = round(s, 1)
        full.at[ts, 'end_min'] = round(e, 1)
        full.at[ts, 'lat'] = lat
        full.at[ts, 'lon'] = lon
        full.at[ts, 'tz_min'] = tz
        est.at[ts] = True
    return est


def canonical_run_times(daily, data_dir, time_path=None):
    """Per-date canonical run clock-times for an arbitrary consumer.

    ``daily`` is a frame with ``date`` (+ ``minutes``, ``time_of_day``,
    ``city_state`` for the pre-watch estimate). Returns a date-indexed frame
    with ``start_min`` / ``end_min`` / ``mid_min`` / ``lat`` / ``lon`` /
    ``tz_min`` in the canonical city's local frame, both eras resolved. Days
    with neither a watch time nor a usable bin come back all-NaN.

    This is the same pipeline ``plot_qualitative_trends.load_series`` runs for
    the Time panel, packaged for callers that only need the clock.
    """
    if time_path is None:
        time_path = os.path.join(str(data_dir), 'time_daily.csv')
    keep = [c for c in ('date', 'minutes', 'time_of_day', 'city_state')
            if c in daily.columns]
    full = daily[keep].drop_duplicates(subset='date').copy()
    cols = ['start_min', 'end_min', 'lat', 'lon', 'tz_min']
    if os.path.exists(time_path):
        extra = pd.read_csv(time_path, parse_dates=['date'])
        full = full.merge(extra[['date'] + cols], on='date', how='left')
    else:
        for c in cols:
            full[c] = np.nan
    full = full.set_index('date')
    coords = canonical_coords(data_dir)
    reproject_watch_to_canonical(full, coords)
    estimate_binned_time(full, coords)
    full['mid_min'] = (full['start_min'] + full['end_min']) / 2.0
    return full[['start_min', 'end_min', 'mid_min', 'lat', 'lon', 'tz_min']]
