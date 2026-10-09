"""Run-window weather for Coros activities from NOAA's HRRR model (via Open-Meteo).

Coros attaches an AccuWeather snapshot to each activity, but it is taken at the
START of the run (verified Oct 2026: regressing Coros − ref@start on the
within-run change gives slope −0.02, 95% CI [−0.14, +0.11]; the best-matching
lag sits at −30..+10 min), so morning long runs read ~2 °C cold. A collocation
study against the hand log (Wunderground / a real thermometer, averaged over
start and end), the nearest airport (ASOS) and three ECMWF models ranked HRRR
first for temperature level, within-run change AND wind; humidity was a
three-way tie with Coros and the airport. Full evaluation:
docs/weather-source-reference.md.

So each outdoor activity gets HRRR hourly temperature / humidity / wind at its
first GPS fix (Open-Meteo lapse-adjusts temperature to that point's elevation
at 6.5 °C/km), averaged over the run's elapsed window (start → start + total
time), minute-sampled with linear interpolation between hours.

Storage: one small sidecar per activity, ``details/wx/<labelId>.json``. It
lives inside the cached details directory, so CI persists it with no change to
the actions/cache path list (the list is the cache's identity), and the
non-recursive ``*.json`` globs every details reader uses never see it. A
sidecar is written once per activity; ``src`` is ``hrrr`` or ``none`` (indoor,
no GPS, non-run sport, or outside HRRR's North-American domain — consumers fall
back to the Coros snapshot). A transient failure (network, data not yet
available) writes nothing, so the next build retries.

Values fetched within ``FINAL_AFTER_S`` of the run's end are provisional (the
archive may still settle the most recent hours); ``ensure`` re-fetches them
once after that window and marks them final. ``watch_daily`` re-derives a day
whose sidecars were missing or provisional once they are final.

    python -m src.coros.weather_hrrr                  # coros profile details
    python -m src.coros.weather_hrrr --details DIR [--limit N]
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.coros import mappings as M
from src.coros.build_current_log import Activity

API_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
MODEL = "gfs_hrrr"
HOURLY = ("temperature_2m", "relative_humidity_2m", "wind_speed_10m")
WX_DIRNAME = "wx"
SIDECAR_VERSION = 1
FINAL_AFTER_S = 48 * 3600        # provisional until fetched this long after the run ends
REQUEST_INTERVAL_S = 0.25        # per-worker politeness pause after each request
WORKERS = 4                      # concurrent requests (~4/s; free tier allows 600/min)
MAX_CONSECUTIVE_FAILURES = 5     # circuit breaker: stop fetching this build (offline)
# A hung connection stalls its whole chunk, so keep this short; a timeout is
# transient (no sidecar) and simply retries next build.
TIMEOUT_S = 20
OUTDOOR_RUN_SPORTS = {M.SPORT_RUN, M.SPORT_TRAIL_RUN, M.SPORT_TRACK_RUN}


class TransientError(Exception):
    """Fetch failed in a way worth retrying next build (no sidecar written)."""


# ---------- sidecar I/O ----------
def sidecar_path(details_dir, label_id):
    return Path(details_dir) / WX_DIRNAME / f"{label_id}.json"


def load(details_dir, label_id):
    """The activity's sidecar dict, or None if it hasn't been fetched yet."""
    p = sidecar_path(details_dir, label_id)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (ValueError, OSError):
        return None


def attach(rec, details_dir, label_id):
    """Attach the sidecar (if any) to an in-memory detail record as ``wx``.
    Never persisted into the detail file — build_current_log reads it from the
    record. Returns ``rec`` for chaining."""
    wx = load(details_dir, label_id)
    if wx is not None:
        rec["wx"] = wx
    return rec


def is_final(wx):
    return bool(wx) and bool(wx.get("final"))


def _write(details_dir, label_id, wx):
    p = sidecar_path(details_dir, label_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(wx))
    tmp.replace(p)


# ---------- derivation ----------
def run_window(act):
    """(t0, t1) epoch seconds: start → start + elapsed time (moving time if the
    elapsed total is missing)."""
    t0 = act.start_utc.timestamp()
    dur = act.total_s if act.total_s and act.total_s > 0 else (act.moving_s or 0.0)
    return t0, t0 + max(dur, 0.0)


def _none(reason, **extra):
    return {"v": SIDECAR_VERSION, "src": "none", "reason": reason,
            "final": True, **extra}


def _fetch_hourly(lat, lon, t0, t1):
    """Open-Meteo HRRR hourly series bracketing [t0, t1]. Returns
    (epoch_seconds, {var: array}, elevation_m) — out-of-domain hours come back
    as NaN. A permanent client error (4xx) returns (None, None, reason).
    Raises TransientError on network/server trouble."""
    d0 = datetime.fromtimestamp(t0 - 3600, tz=timezone.utc).date()
    d1 = datetime.fromtimestamp(t1 + 3600, tz=timezone.utc).date()
    q = urllib.parse.urlencode({
        "latitude": f"{lat:.4f}", "longitude": f"{lon:.4f}",
        "start_date": d0.isoformat(), "end_date": d1.isoformat(),
        "hourly": ",".join(HOURLY), "models": MODEL,
        "wind_speed_unit": "mph", "timezone": "GMT"})
    try:
        with urllib.request.urlopen(f"{API_URL}?{q}", timeout=TIMEOUT_S) as r:
            body = json.loads(r.read())
    except urllib.error.HTTPError as e:
        # 429 (rate limit) is retryable; any other 4xx is a permanent refusal
        # (e.g. coordinates outside the model).
        if 400 <= e.code < 500 and e.code != 429:
            try:
                reason = json.loads(e.read()).get("reason", str(e))
            except (ValueError, OSError):
                reason = str(e)
            return None, None, reason
        raise TransientError(f"HTTP {e.code}") from e
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as e:
        raise TransientError(str(e)) from e
    h = body.get("hourly") or {}
    times = np.array([datetime.fromisoformat(t).replace(tzinfo=timezone.utc).timestamp()
                      for t in h.get("time", [])], float)
    series = {v: np.array([np.nan if x is None else x for x in h.get(v, [])], float)
              for v in HOURLY}
    return times, series, body.get("elevation")


def derive(rec, *, now=None):
    """Sidecar dict for one detail record, or raise TransientError."""
    now = time.time() if now is None else now
    sport = (rec.get("summary") or {}).get("sportType")
    if sport not in OUTDOOR_RUN_SPORTS:
        return _none("not_outdoor_run", sport_type=sport)
    act = Activity(rec)
    if act.lat is None or act.lon is None:
        return _none("no_gps")
    t0, t1 = run_window(act)
    base = {"v": SIDECAR_VERSION, "model": MODEL, "lat": round(act.lat, 4),
            "lon": round(act.lon, 4), "t0": t0, "t1": t1}
    times, series, elev = _fetch_hourly(act.lat, act.lon, t0, t1)
    if times is None:                       # permanent client error from the API
        return {**_none(f"api: {elev}"), **base}
    grid = np.arange(t0, t1, 60.0)
    grid = np.append(grid, t1) if len(grid) else np.array([t0])
    vals = {v: np.interp(grid, times, series[v]) if len(times) else np.full(len(grid), np.nan)
            for v in HOURLY}
    if not all(np.isfinite(a).all() for a in vals.values()):
        if now - t1 < FINAL_AFTER_S:
            raise TransientError("hours not yet available")
        return {**_none("no_hrrr_coverage"), **base}
    T = vals["temperature_2m"]
    return {**base, "src": "hrrr", "elev_m": elev,
            "temp_c": round(float(T.mean()), 2),
            "temp_start_c": round(float(T[0]), 2),
            "temp_max_c": round(float(T.max()), 2),
            "rh_pct": round(float(vals["relative_humidity_2m"].mean()), 1),
            "wind_mph": round(float(vals["wind_speed_10m"].mean()), 2),
            "fetched_utc": now, "final": now - t1 >= FINAL_AFTER_S}


# ---------- incremental driver ----------
def _due(details_dir, label_id, now):
    """True when this activity needs a (re)fetch: no sidecar, or a provisional
    one whose settling window has passed."""
    wx = load(details_dir, label_id)
    if wx is None:
        return True
    if is_final(wx):
        return False
    return now - float(wx.get("t1") or 0) >= FINAL_AFTER_S


def ensure(details_dir, *, limit=None, now=None, verbose=True):
    """Fetch sidecars for every cached activity that lacks a final one.
    Requests run on a small pool (WORKERS). Returns a counts dict. Stops early
    once MAX_CONSECUTIVE_FAILURES transient failures pile up (offline / API
    down) — the rest retry next build."""
    details_dir = Path(details_dir)
    counts = {"hrrr": 0, "none": 0, "transient": 0, "skipped_final": 0}
    if not details_dir.exists():
        return counts
    now = time.time() if now is None else now
    have = {p.stem for p in (details_dir / WX_DIRNAME).glob("*.json")} \
        if (details_dir / WX_DIRNAME).exists() else set()
    todo = []
    for p in sorted(details_dir.glob("*.json")):
        lid = p.stem
        if lid in M.EXCLUDED_LABEL_IDS:
            continue
        if lid in have and not _due(details_dir, lid, now):
            counts["skipped_final"] += 1
            continue
        todo.append(p)
    if limit is not None:
        todo = todo[:limit]
    if verbose and todo:
        print(f"[weather_hrrr] fetching run-window weather for {len(todo)} "
              f"activit{'y' if len(todo) == 1 else 'ies'} ({details_dir})", flush=True)

    def work(p):
        try:
            rec = json.loads(p.read_text())
        except (ValueError, OSError):
            return p.stem, None, None
        try:
            wx = derive(rec, now=now)
        except TransientError as e:
            return p.stem, None, e
        if "model" in wx:                       # a request was made -> throttle
            time.sleep(REQUEST_INTERVAL_S)
        return p.stem, wx, None

    # Chunked so the circuit breaker can stop between chunks when offline.
    fails, done, chunk = 0, 0, WORKERS * 8
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for start in range(0, len(todo), chunk):
            for lid, wx, err in pool.map(work, todo[start:start + chunk]):
                if err is not None:
                    counts["transient"] += 1
                    fails += 1
                    continue
                if wx is None:
                    continue
                fails = 0
                _write(details_dir, lid, wx)
                counts[wx["src"]] += 1
            done = min(start + chunk, len(todo))
            if fails >= MAX_CONSECUTIVE_FAILURES:
                print(f"[weather_hrrr] {fails} consecutive failures; stopping — "
                      f"remaining activities retry next build")
                break
            if verbose and done % 256 == 0:
                print(f"[weather_hrrr]   {done}/{len(todo)}", flush=True)
    if verbose and (counts["hrrr"] or counts["none"] or counts["transient"]):
        print(f"[weather_hrrr] hrrr={counts['hrrr']} none={counts['none']} "
              f"transient={counts['transient']} (already final: "
              f"{counts['skipped_final']})")
    return counts


def main(argv=None):
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    p.add_argument("--details", help="details dir (default: coros profile's)")
    p.add_argument("--limit", type=int, help="fetch at most N activities")
    a = p.parse_args(argv)
    if a.details:
        d = Path(a.details)
    else:
        from src.profiles import get_profile
        d = get_profile("coros").data_dir / "details"
    if not d.exists():
        print(f"[weather_hrrr] no details cache at {d} — skipped")
        return
    ensure(d, limit=a.limit)


if __name__ == "__main__":
    main()
