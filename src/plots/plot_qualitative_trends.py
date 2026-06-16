"""Miscellaneous Trends plot — two pages of four panels, top-right pill toggle.

  Weather page:  Conditions, Temperature, Humidity, Wind
  Other page:    Volume,     Altitude,    Time,     Weight

All eight panels live in ONE 4-row figure (render_plot's single-figure
contract). Every trace is tagged ``meta.page`` ('weather' | 'other'); the
sibling ``plot_qualitative_trends.js`` flips trace visibility by page and
relayouts the four shared y-axes plus each panel's image/shape visibility.
Inset titles are HTML overlay divs (``.rp-inset``) positioned per subplot by
that same JS; each carries a ↗ link opening the panel as its own standalone
full page (``--panel <key>`` -> ``qualitative_trends_<key>.html``, built by
``build_single``). Default page is Weather.

Panel render paths:
  - strip envelope + white trend line (the classic look): Temperature, Humidity,
    Wind, Volume, Weight — a 14-day rolling min/max envelope drawn as N
    horizontal gradient strips, colored by ``gradient_at(y_mid, anchors)``, with
    a Gaussian-MA trend line.
  - range-per-day envelope + TWO edge trends: Altitude, Time — the band is the
    rolling min-of-daily-min / max-of-daily-max, and two white trend lines track
    the smoothed daily min and daily max. Altitude fills with the strip stack
    (terrain green->brown->white, feet); Time fills with a per-day SOLAR raster
    (twilight-purple -> sunrise-orange -> noon-blue), built from each run's GPS.
  - qualitative scatter: Conditions — one dot per logged day at y=temperature,
    dot fill = weather CF color, dot ring = conditions CF color (dry = no ring).

Watch-only series (Humidity, Wind, Altitude, Time) are blank before the watch
era (~late 2020) on the shared 2016->present axis; the other four carry full
history. A panel with NO data for a profile (e.g. Weight on a watch-only
profile that logs no body weight) is dropped entirely and the page reflows its
surviving panels to fill the height — see build_panels / _panel_has_data and
the per-page row sizing in main().

Tooltip (custom, per-tab): up to eight rows — a day-specific block (that day's
actual recorded value per metric) above an averages block (trend value + range).
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import PchipInterpolator
from scipy.signal import find_peaks

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.shared.paths import DATA_DIR, OUTPUT_DIR
from src.shared.plot_window import daily_floor
from src.shared.effective_mileage import effective_daily_miles
from src.coros.solar import solar_anchors_local
from src.parsers.snapshot import (find_snapshot, read_snapshot,
                                  coord_overrides_from_sections)
from src.plotting import (render_plot, CursorTooltip, apply_default_layout,
                            FG, FG_DIM, GRID, widgets,
                            yearly_x_axis_kwargs)
from src.plotting.raster import render_gradient_raster


DEFAULT_DAILY = str(DATA_DIR / 'daily.csv')
DEFAULT_ALT = str(DATA_DIR / 'altitude_daily.csv')
DEFAULT_TIME = str(DATA_DIR / 'time_daily.csv')
DEFAULT_CF = str(Path(__file__).resolve().parents[2] / 'docs'
                 / 'running_log_2025_cf.json')
DEFAULT_OUT = str(OUTPUT_DIR)
START_DATE = '2016-01-01'

MA_WINDOW = {'temp': 28, 'humidity': 28, 'wind': 56,
             'volume': 56, 'altitude': 28, 'time': 28, 'weight': 56}

RANGE_WINDOW = 7             # rolling min/max window for the tooltip "(min to max)"
PEAK_DISTANCE_DAYS = 7       # min separation (days) between envelope peaks/troughs
ENVELOPE_GAP_BREAK_DAYS = 21  # bridge gaps up to this (rest days, short breaks);
                              # break the band only across longer layoffs
WEIGHT_INTERP_MAX_GAP = 7
RASTER_H = 1080               # vertical px resolution of every gradient raster
                              # (full-page 1080p; panels can render full-height)

# Solar gradient control colors (RGB). The day's anchor MINUTES come from
# solar_anchors_local; these are the colors at each anchor. The purples are kept
# light enough to read against the #1a1a1a panel (a near-black night vanishes).
SOLAR_NIGHT = (48, 32, 82)        # before dawn / after dusk (lightened purple)
SOLAR_TWILIGHT = (96, 58, 140)    # civil twilight (purple)
SOLAR_HORIZON = (240, 138, 61)    # sunrise / sunset (orange)
SOLAR_NOON = (154, 208, 245)      # full daylight, sun >= +18deg (light blue)


# ---------- color utilities ----------

def hex_to_rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def lerp_rgb(c1, c2, t):
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))


def gradient_at(value, anchors):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return (160, 160, 160)
    if value <= anchors[0][0]:
        return hex_to_rgb(anchors[0][1])
    if value >= anchors[-1][0]:
        return hex_to_rgb(anchors[-1][1])
    for i in range(len(anchors) - 1):
        v0, h0 = anchors[i]
        v1, h1 = anchors[i + 1]
        if v0 <= value <= v1:
            t = (value - v0) / (v1 - v0) if v1 > v0 else 0
            return lerp_rgb(hex_to_rgb(h0), hex_to_rgb(h1), t)
    return hex_to_rgb(anchors[-1][1])


# ---------- data shaping ----------

def gap_runs(mask):
    runs = []
    in_run = False
    start = 0
    for i, v in enumerate(mask):
        if v and not in_run:
            in_run = True
            start = i
        elif not v and in_run:
            in_run = False
            runs.append((start, i))
    if in_run:
        runs.append((start, len(mask)))
    return runs


def interpolate_short_gaps(s, max_gap_days):
    s = s.copy()
    runs = gap_runs(s.isna().values)
    interp = s.interpolate(method='linear', limit_direction='both')
    out = s.copy()
    for start, end in runs:
        if (end - start) < max_gap_days:
            out.iloc[start:end] = interp.iloc[start:end]
    return out


def load_series(daily_path, alt_path, time_path, start_date):
    df = pd.read_csv(daily_path, parse_dates=['date'])
    df = df[df['date'] >= pd.Timestamp(start_date)].copy()
    # Source of truth: watch/route distance-corrected mileage drives the volume
    # trend. On-disk 'miles' is untouched.
    df['miles'] = effective_daily_miles(df)
    end = df['date'].max()
    cal = pd.DataFrame({'date': pd.date_range(start_date, end, freq='D')})
    keep_cols = ['date', 'miles', 'minutes', 'temp_c', 'weight_lbs',
                 'humidity_pct', 'wind_mph', 'weather', 'conditions',
                 'time_of_day', 'city_state']
    keep = df[[c for c in keep_cols if c in df.columns]]
    full = cal.merge(keep, on='date', how='left')

    # Per-day altitude / time envelopes (watch-era; graceful-empty if absent).
    for path, cols in ((alt_path, ['min_elev_ft', 'max_elev_ft']),
                       (time_path, ['start_min', 'end_min', 'lat', 'lon',
                                    'tz_min'])):
        if os.path.exists(path):
            extra = pd.read_csv(path, parse_dates=['date'])
            full = full.merge(extra[['date'] + cols], on='date', how='left')
        else:
            for c in cols:
                full[c] = np.nan

    full = full.set_index('date')
    full['miles'] = full['miles'].fillna(0.0)
    full['weight_interp'] = interpolate_short_gaps(
        full['weight_lbs'], WEIGHT_INTERP_MAX_GAP)

    # Canonical city table: {city_state: (lat, lon, tz_iana)}. Drives the Time
    # panel's location + timezone for BOTH eras (see reproject_watch_to_canonical
    # and estimate_binned_time), so the watch's reported GPS/offset is never
    # trusted for the gradient.
    cc_path = os.path.join(os.path.dirname(daily_path), 'city_coords.csv')
    coords = {}
    if os.path.exists(cc_path):
        cc = pd.read_csv(cc_path)
        has_tz = 'tz' in cc.columns
        coords = {r['city_state']: (r['latitude'], r['longitude'],
                                    r['tz'] if has_tz and pd.notna(r['tz']) else None)
                  for _, r in cc.iterrows()
                  if pd.notna(r['latitude']) and pd.notna(r['longitude'])}

    # Apply snapshot coordinate overrides on top of the Nominatim cache (same
    # corrections the world map uses via ensure_coords), so a geocoding fix
    # reaches the Time panel too. Read-time only — city_coords.csv is left
    # regenerable from source. Keep each city's cached tz (a coordinate fix
    # almost never crosses a timezone); no-op when the snapshot or its
    # coordinates section is absent (e.g. a watch-import profile).
    snap_path = find_snapshot([os.path.join(os.path.dirname(daily_path),
                                            'drive_snapshot.csv')])
    if snap_path:
        sections, _ = read_snapshot(snap_path)
        for cs, (lat, lon) in coord_overrides_from_sections(sections).items():
            tz = coords[cs][2] if cs in coords else None
            coords[cs] = (lat, lon, tz)

    # Watch era: reproject each run's stored (local-clock, watch-tz) time onto
    # the canonical city's tz + coordinates. Pre-watch era: estimate from the
    # time-of-day bin against the canonical city's solar times.
    reproject_watch_to_canonical(full, coords)
    full['time_est'] = estimate_binned_time(full, coords)
    return full


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


def rolling_minmax_raw(s, window):
    lo = s.rolling(window, min_periods=1, center=True).min()
    hi = s.rolling(window, min_periods=1, center=True).max()
    return lo, hi


def rolling_ma(s, window, min_periods=None):
    sigma = max(2.0, window / 7)
    mp = min_periods if min_periods is not None else max(1, window // 4)
    return s.rolling(window, min_periods=mp,
                     center=True, win_type='gaussian').mean(std=sigma)


# ---------- gradient ramp (x-independent panels) ----------

def gradient_ramp(anchors, y0, y1, h_px):
    """(h_px, 3) uint8 vertical colour ramp = ``gradient_at`` sampled at each
    output pixel-row's y. The x-independent case of a raster ``column_colors``
    (same ramp every day), for the anchor-based panels (temp/humidity/wind/
    volume/weight/altitude)."""
    yp = y1 - (np.arange(h_px) + 0.5) / h_px * (y1 - y0)
    return np.array([gradient_at(float(v), anchors) for v in yp],
                    dtype=np.uint8)


def _date_path(dates, yvals):
    """SVG path string over (date, y) points for a layout 'path' shape, split
    into independent subpaths at NaN gaps ('M' restarts the pen). Crisp vector
    alternative to baking a line into a raster (which the stretch-resize blurs).

    y is emitted at 3-decimal precision: the slow (56-day) trends sit on a
    near-constant value, so rounding to 0.1 quantized them into visible
    stairsteps on the zoomed-in panels (wind/volume/weight). 3 dp is smooth
    against any y-range here while keeping the path string compact.
    """
    parts, pen_up = [], True
    for d, v in zip(dates, yvals):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            pen_up = True
            continue
        parts.append(f"{'M' if pen_up else 'L'}{d.strftime('%Y-%m-%d')},{v:.3f}")
        pen_up = False
    return ' '.join(parts)


# ---------- envelope edges (peak-interpolation, shared knots) ----------

def env_edges(lo_in, hi_in, distance, gap_break=ENVELOPE_GAP_BREAK_DAYS):
    """Smooth (lo_s, hi_s) band that PASSES THROUGH the extrema yet never
    spuriously collapses to zero width.

    Both edges are PCHIP curves through a SHARED set of knot x-positions (the
    union of ``hi_in``'s maxima and ``lo_in``'s minima, found by ``find_peaks``).
    At each knot the upper value is the local max of ``hi_in`` and the lower the
    local min of ``lo_in`` over a +/- (distance/2) neighborhood — so the band
    width at every knot is the genuine local spread (never 0 just because a lone
    peak and lone trough happened to coincide), while the upper still reaches the
    true peaks (a peak knot's neighborhood-max IS that peak). Sharing the knots
    keeps the two curves moving together, so they can't drift into a pinch.

    ``hi_in``/``lo_in`` are the per-day high/low (equal for single-value panels;
    the daily max/min for range panels). Works on OBSERVED (non-NaN) points at
    their true day index, bridging rest days; the band breaks only across gaps
    longer than ``gap_break`` days. PCHIP is shape-preserving (no overshoot)."""
    hv = hi_in.to_numpy(dtype=float)
    lv = lo_in.to_numpy(dtype=float)
    n = len(hv)
    out_hi = np.full(n, np.nan)
    out_lo = np.full(n, np.nan)
    obs = np.flatnonzero(np.isfinite(hv) & np.isfinite(lv))
    if len(obs) == 0:
        return pd.Series(out_lo, index=lo_in.index), pd.Series(out_hi, index=hi_in.index)
    d = max(1, int(distance))
    half = max(1, d // 2)
    splits = np.flatnonzero(np.diff(obs) > gap_break) + 1
    for seg in np.split(obs, splits):
        m = len(seg)
        sx = seg.astype(float)                         # true day indices (gappy)
        hy, ly = hv[seg], lv[seg]                       # observed high / low
        if m == 1:
            out_hi[seg[0]], out_lo[seg[0]] = hy[0], ly[0]
            continue
        # Shared knots: union of hi-maxima and lo-minima, + global extremes + ends.
        hp, _ = find_peaks(hy, distance=d)
        lt, _ = find_peaks(-ly, distance=d)
        kset = set(hp.tolist()) | set(lt.tolist())
        kset.update((int(np.argmax(hy)), int(np.argmin(ly)), 0, m - 1))
        ks = np.array(sorted(kset))
        kx = sx[ks]
        # Knot values = local max(hi)/min(lo) over the +-half neighborhood, so the
        # knot width is the real local spread (upper still hits true peaks).
        ky_hi = np.array([hy[max(0, k - half):k + half + 1].max() for k in ks])
        ky_lo = np.array([ly[max(0, k - half):k + half + 1].min() for k in ks])
        days = np.arange(seg[0], seg[-1] + 1)
        if len(ks) >= 2:
            out_hi[days] = PchipInterpolator(kx, ky_hi, extrapolate=False)(days.astype(float))
            out_lo[days] = PchipInterpolator(kx, ky_lo, extrapolate=False)(days.astype(float))
        else:
            out_hi[days], out_lo[days] = ky_hi[0], ky_lo[0]
    # Belt-and-suspenders: forbid any residual cross from PCHIP curvature.
    out_hi = np.maximum(out_hi, out_lo)
    return pd.Series(out_lo, index=lo_in.index), pd.Series(out_hi, index=hi_in.index)


# ---------- solar raster color ramp ----------

def solar_column_colors(date_py, lat, lon, tz_min, y_p):
    """(len(y_p), 3) uint8 ramp for one day: night -> twilight -> sunrise
    (orange) -> blue (once the sun reaches +18deg) -> noon -> blue -> +18deg
    -> sunset (orange) -> twilight -> night, anchored at the day's solar times.
    The orange->blue shoulder is the sunrise..+18deg span (astronomically tied,
    wider than a fixed clock offset). ``y_p`` is the array of pixel-row clock
    minutes (descending, top=late)."""
    a = solar_anchors_local(date_py, lat, lon, tz_min)
    noon = a['solar_noon'] if a['solar_noon'] is not None else 720.0
    pts = [(0.0, SOLAR_NIGHT)]  # (minute, color)
    if a['twilight_begin'] is not None:
        pts.append((a['twilight_begin'], SOLAR_TWILIGHT))
    if a['sunrise'] is not None:
        pts.append((a['sunrise'], SOLAR_HORIZON))
    if a['blue_begin'] is not None:
        pts.append((a['blue_begin'], SOLAR_NOON))
    pts.append((noon, SOLAR_NOON))
    if a['blue_end'] is not None:
        pts.append((a['blue_end'], SOLAR_NOON))
    if a['sunset'] is not None:
        pts.append((a['sunset'], SOLAR_HORIZON))
    if a['twilight_end'] is not None:
        pts.append((a['twilight_end'], SOLAR_TWILIGHT))
    pts.append((1440.0, SOLAR_NIGHT))
    # Enforce strictly increasing minutes for np.interp (clamp any disorder).
    xs, last = [], -1.0
    for m, _ in pts:
        m = max(m, last + 1e-6)
        xs.append(m)
        last = m
    xs = np.asarray(xs)
    cols = np.asarray([c for _, c in pts], dtype=float)
    out = np.empty((len(y_p), 3), dtype=np.uint8)
    for ch in range(3):
        out[:, ch] = np.clip(np.interp(y_p, xs, cols[:, ch]), 0, 255)
    return out


# ---------- main ----------

def _round_arr(s, n=2):
    return [None if (v is None or (isinstance(v, float) and np.isnan(v)))
            else round(float(v), n) for v in s.values]


def _str_arr(s):
    return [None if (v is None or (isinstance(v, float) and np.isnan(v)))
            else str(v) for v in s.values]


# Per-page y-axis key/suffix for subplot row ``r`` (row 1 has no numeric
# suffix in Plotly's axis naming).
def _axkey(r):
    return 'yaxis' if r == 1 else f'yaxis{r}'


def _xaxkey(r):
    return 'xaxis' if r == 1 else f'xaxis{r}'


# Shared with _row_domains so the JS per-page relayout reproduces the
# make_subplots row spacing exactly.
VERTICAL_SPACING = 0.025


def _row_domains(k, vs=VERTICAL_SPACING):
    """Per-row y-axis paper domains [bottom, top] for ``k`` equal rows stacked
    top-to-bottom with ``vs`` spacing — the same split make_subplots computes,
    recomputed here so a page with fewer surviving panels can reflow its rows
    to fill the full height (the shorter of the two toggled pages)."""
    if k <= 0:
        return []
    h = (1.0 - vs * (k - 1)) / k
    doms = []
    for r in range(1, k + 1):
        top = 1.0 - (r - 1) * (h + vs)
        doms.append([round(top - h, 6), round(top, 6)])
    return doms


def _panel_has_data(pn, full):
    """True when a panel has any finite data for this profile. Drives the
    empty-panel drop (e.g. a watch-only profile logs no body weight, so the
    Weight panel must disappear rather than render an empty graph)."""
    if pn['type'] == 'scatter':                      # Conditions
        return bool((full['weather'].notna() & full['temp_c'].notna()).any())
    if 'series' in pn:
        return bool(pd.Series(pn['series']).notna().any())
    lo = pd.Series(pn.get('lo'))
    hi = pd.Series(pn.get('hi'))
    return bool((lo.notna() | hi.notna()).any())


TIME_TICKVALS = [0, 240, 480, 720, 960, 1200, 1440]
TIME_TICKTEXT = ['0:00', '4:00', '8:00', '12:00', '16:00', '20:00', '']


def build_panels(full):
    """The eight panel specs (4 Weather + 4 Other), with y-ranges derived from
    ``full``. ``type`` is 'scatter' (Conditions) or 'envelope'; an envelope is
    single-value (``series`` -> one MA trend) or range (``lo``/``hi`` -> two
    edges), coloured by anchors or the per-day ``solar`` ramp (Time)."""
    miles_max = float(full['miles'].max())

    def pad_range(series, frac=0.04, floor0=False):
        v = pd.Series(series).dropna()
        if v.empty:
            return [0.0, 1.0]
        lo, hi = float(v.min()), float(v.max())
        p = (hi - lo) * frac if hi > lo else 1.0
        return [0.0 if floor0 else lo - p, hi + p]

    temp_range = pad_range(full['temp_c'])
    # Wind is AccuWeather ground truth (mph; raw/10 km/h x KMH_TO_MPH at the
    # source) — taken at face value, no cap. Genuine outliers get invalidated by
    # hand via the changes sheet, not clipped here.
    w_v = pd.Series(full['wind_mph']).dropna()
    wind_max = float(w_v.max()) if len(w_v) else 1.0
    # Altitude floors at 0: uncalibrated-barometer dips below sea level are
    # unphysical, so the axis cuts off at 0 and those dips clip; the top tracks
    # the real ceiling (fourteener/foothill highs).
    alt_hi_v = pd.Series(full.get('max_elev_ft')).dropna()
    if len(alt_hi_v):
        a1 = float(alt_hi_v.max())
        alt_range = [0.0, a1 * 1.04]
    else:
        alt_range = [0.0, 1.0]
    weight_range = pad_range(full['weight_interp'])

    return [
        # Weather page
        dict(page='weather', row=1, key='conditions', label='Conditions',
             unit='°C', type='scatter', y_range=temp_range),
        dict(page='weather', row=2, key='temp', label='Temperature',
             unit='°C', type='envelope', series=full['temp_c'],
             anchors=[(-10.0, '#0E7BAA'), (22.0, '#5A9E3D'), (40.0, '#C82020')],
             y_range=temp_range),
        dict(page='weather', row=3, key='humidity', label='Humidity',
             unit='%', type='envelope', series=full['humidity_pct'],
             # Two-stop orange->blue (a white mid-stop washed out the trendline).
             # Lightened both stops — the originals read darker than the white
             # trend line against the panel bg.
             anchors=[(0.0, '#E2701E'), (100.0, '#2C63AE')],
             y_range=[0.0, 100.0]),
        dict(page='weather', row=4, key='wind', label='Wind',
             unit='mph', type='envelope', series=full['wind_mph'],
             # Darkened/desaturated the green->yellow->red ramp — the originals
             # read brighter than the white trend line against the panel bg.
             anchors=[(0.0, '#7BB050'), (wind_max / 2, '#D4B028'),
                      (wind_max, '#C0342A')],
             y_range=[0.0, wind_max * 1.05]),
        # Other page
        dict(page='other', row=1, key='volume', label='Daily volume',
             unit='mi', type='envelope', series=full['miles'],
             anchors=[(0.0, '#3D2208'), (8.0, '#8B5C16'),
                      (miles_max, '#F2D034')],
             y_range=[0.0, miles_max * 1.05]),
        dict(page='other', row=2, key='altitude', label='Altitude',
             unit='ft', type='envelope',
             lo=full.get('min_elev_ft'), hi=full.get('max_elev_ft'),
             anchors=[(alt_range[0], '#2E7D32'),
                      ((alt_range[0] + alt_range[1]) / 2, '#8B5A2B'),
                      (alt_range[1], '#FFFFFF')],
             y_range=alt_range),
        dict(page='other', row=3, key='time', label='Time of day',
             unit='', type='envelope', solar=True,
             lo=full.get('start_min'), hi=full.get('end_min'),
             y_range=[0.0, 1440.0]),
        dict(page='other', row=4, key='weight', label='Weight',
             unit='lbs', type='envelope', series=full['weight_interp'],
             anchors=[(145.0, '#2D1006'), (156.0, '#8C4F1F'),
                      (170.0, '#E89535')],
             y_range=weight_range),
    ]


def _prepare(args):
    """Shared load/setup for both the multi-panel and single-panel builds:
    returns ``(start, full, panels, cond_cf, weather_syn, wcolor)``."""
    os.makedirs(args.out_dir, exist_ok=True)
    start = args.start or str(daily_floor().date())
    full = load_series(args.daily, args.altitude, args.time, start)
    with open(args.cf) as f:
        cf = json.load(f)
    weather_cf = cf['rules']['weather']
    cond_cf = cf['rules']['conditions']
    # Older / variant log vocabulary that predates the CF palette keys (esp.
    # 2016, which logged 'clouds'/'drizzle'/'showers') -> nearest CF bin, so
    # those days color correctly instead of falling to the gray fallback.
    weather_syn = {'clouds': 'cloudy', 'drizzle': 'light rain',
                   'showers': 'light rain', 'fog': 'foggy', 'indoors': 'inside'}
    # The CF pastels for clear/cloudy/overcast/foggy are light-cell tints that
    # wash into indistinguishable near-grays on the dark plot bg. Remap just
    # those to dark-legible tones — clear stays a recognizable light blue, the
    # cloud family stays neutral grays at distinct lightness; every other
    # weather keeps its CF hex (already saturated enough).
    weather_dark = {'clear': '#9FC5E8', 'cloudy': '#B0B0B0',
                    'overcast': '#7F7F7F', 'foggy': '#D9D9D9'}

    def wcolor(w):
        return weather_dark.get(w) or weather_cf.get(w, '#888888')

    panels = build_panels(full)
    return start, full, panels, cond_cf, weather_syn, wcolor


def render_panel(fig, pn, row, full, dates, *, cond_cf, weather_syn, wcolor,
                 visible):
    """Build one panel (scatter or gradient envelope) into ``row`` of ``fig``.

    Drives both the multi-panel figure (``row = pn['row']``) and a single-panel
    standalone figure (``row = 1``) — same traces/images/shapes/payload either
    way. Returns ``(image_idx, shape_idx, tab, data_range)`` where ``image_idx``
    / ``shape_idx`` are the indices of the layout images / shapes this panel
    added (for the page-visibility relayout), ``tab`` is a tooltip-payload
    fragment (``{'avg':…, 'lo':…, 'hi':…, 'day':…}``) to merge into the page's
    tab, and ``data_range`` is this panel's ``(first_date, last_date)`` data
    extent (``None`` if empty) — the standalone single-panel build tightens its
    x-axis to exactly that, since it shares no axis with the other panels.
    ``visible`` is the initial visibility of the raster + trend shapes (the
    multi-panel page toggle flips it; single-panel passes ``True``)."""
    page = pn['page']
    y0, y1 = pn['y_range']
    image_idx, shape_idx = [], []
    tab = {'avg': {}, 'lo': {}, 'hi': {}, 'day': {}}

    if pn['type'] == 'scatter':
        # Conditions: dot per logged day at y=temp; fill=weather, ring=conditions.
        sub = full[full['weather'].notna() & full['temp_c'].notna()]
        fill, ring, ringw = [], [], []
        for w, c in zip(sub['weather'], sub['conditions']):
            ws = str(w).strip().lower()
            fill.append(wcolor(weather_syn.get(ws, ws)))
            # NaN conditions (the new pandas `str` dtype yields pd.NA, which
            # never equals the literal 'nan') and 'dry' mean no ring.
            cs = str(c).strip().lower() if pd.notna(c) else ''
            if cs in ('', 'dry', 'none', 'nan'):
                ring.append('rgba(0,0,0,0)')
                ringw.append(0.0)
            else:
                ring.append(cond_cf.get(cs, '#cccccc'))
                ringw.append(1.4)
        fig.add_trace(go.Scatter(
            x=sub.index, y=sub['temp_c'], mode='markers',
            marker=dict(size=6, color=fill, opacity=1.0,
                        line=dict(color=ring, width=ringw)),
            hoverinfo='skip', showlegend=False, meta={'page': page},
        ), row=row, col=1)
        tab['day']['weather'] = _str_arr(full['weather'])
        tab['day']['conditions'] = _str_arr(full['conditions'])
        tab['day']['cond_temp'] = _round_arr(full['temp_c'], 1)
        drange = (sub.index.min(), sub.index.max()) if len(sub.index) else None
        return image_idx, shape_idx, tab, drange

    # ---- envelope panel: gradient raster fill + vector trend shape(s) ----
    key = pn['key']

    if 'series' in pn:                          # single-value envelope
        s = pn['series']
        lo_s, hi_s = env_edges(s, s, PEAK_DISTANCE_DAYS)
        if key == 'weight':                     # don't bridge >7d gaps
            lo_s, hi_s = lo_s.where(s.notna()), hi_s.where(s.notna())
        # The envelope is the single source of truth for gaps; force the
        # trend (and tooltip range) onto exactly its valid days so the white
        # line starts/stops/breaks WITH the band. min_periods=1 keeps the MA
        # finite to the band edges; masking trims it to no further.
        env_valid = lo_s.notna() & hi_s.notna()
        ma = rolling_ma(s, MA_WINDOW[key], min_periods=1).where(env_valid)
        lo_raw, hi_raw = rolling_minmax_raw(s, RANGE_WINDOW)
        lo_raw, hi_raw = lo_raw.where(env_valid), hi_raw.where(env_valid)
        trends = [ma]                           # one white MA trend line
        tab['avg'][key] = _round_arr(ma)
        tab['lo'][key] = _round_arr(lo_raw)
        tab['hi'][key] = _round_arr(hi_raw)
        tab['day'][key] = _round_arr(s, 1)
    else:                                       # range (two-edge) envelope
        lo_in = pd.Series(pn['lo'], index=dates)
        hi_in = pd.Series(pn['hi'], index=dates)
        lo_s, hi_s = env_edges(lo_in, hi_in, PEAK_DISTANCE_DAYS)
        trends = []                             # range panels draw no lines
        # Tooltip avg row (unified format): avg = smoothed band midpoint;
        # (min to max) = the rolling envelope. Masked to the band's valid
        # days so the tooltip only reports where the band is drawn.
        env_valid = lo_s.notna() & hi_s.notna()
        avg = rolling_ma((lo_in + hi_in) / 2.0,
                         MA_WINDOW[key], min_periods=1).where(env_valid)
        lo_raw = lo_in.rolling(RANGE_WINDOW, min_periods=1,
                               center=True).min().where(env_valid)
        hi_raw = hi_in.rolling(RANGE_WINDOW, min_periods=1,
                               center=True).max().where(env_valid)
        d = 0 if key == 'altitude' else 1
        tab['avg'][key] = _round_arr(avg, d)
        tab['lo'][key] = _round_arr(lo_raw, d)
        tab['hi'][key] = _round_arr(hi_raw, d)
        tab['day'][key + '_lo'] = _round_arr(lo_in, 1)
        tab['day'][key + '_hi'] = _round_arr(hi_in, 1)
        if key == 'time' and 'time_est' in full:
            tab['day']['time_est'] = [bool(v) for v in full['time_est'].values]

    # Per-day colour ramp: the per-day solar gradient (Time) or a constant
    # anchor-based vertical ramp (every other panel).
    if pn.get('solar'):
        y_p = y1 - (np.arange(RASTER_H) + 0.5) / RASTER_H * (y1 - y0)
        date_py = [d.date() for d in dates]
        # The rolling envelope can reach a few days past the last GPS fix;
        # fill coords forward/back so every rendered column has a location.
        lat_arr = full['lat'].ffill().bfill().fillna(40.015).values
        lon_arr = full['lon'].ffill().bfill().fillna(-105.27).values
        tz_arr = full['tz_min'].ffill().bfill().fillna(-420).values

        def column_colors(d, _yp=y_p, _dp=date_py,
                          _lat=lat_arr, _lon=lon_arr, _tz=tz_arr):
            return solar_column_colors(
                _dp[d], float(_lat[d]), float(_lon[d]), int(_tz[d]), _yp)
    else:
        ramp = gradient_ramp(pn['anchors'], y0, y1, RASTER_H)

        def column_colors(d, _r=ramp):
            return _r

    # Gradient raster, drawn ABOVE gridlines (covers them like the old fill).
    idx = render_gradient_raster(
        fig, row, dates, lo_s, hi_s, column_colors,
        y_range=(y0, y1), h_px=RASTER_H, layer='above', visible=visible)
    image_idx.append(idx)

    # White trend(s) as crisp vector path shapes ABOVE the raster (a trace
    # would be hidden under the above-layer image; a baked line blurs under
    # the stretch-resize). Per-page visibility set in the page relayout.
    if trends:
        axn_r = '' if row == 1 else str(row)
        for tr in trends:
            pth = _date_path(dates, tr.values)
            if not pth:
                continue
            fig.add_shape(dict(
                type='path', path=pth,
                xref=f'x{axn_r}', yref=f'y{axn_r}',
                line=dict(color='#ffffff', width=2.0),
                layer='above', visible=visible))
            shape_idx.append(len(fig.layout.shapes) - 1)
    vd = env_valid.index[env_valid.to_numpy()]
    drange = (vd.min(), vd.max()) if len(vd) else None
    return image_idx, shape_idx, tab, drange


def _inset_html(panels, single_key=None):
    """The per-subplot inset label overlay(s) (replaces Plotly annotations).

    Multi-panel: eight divs (one per panel), each tagged with its row/page/key,
    carrying the label + a ↗ link to that panel's standalone full-page HTML.
    Other-page insets start hidden; the page toggle swaps them. Single-panel
    (``single_key`` set): one label-only div for that panel (no ↗, no toggle).
    Positioned per subplot at runtime by ``plot_qualitative_trends.js``."""
    parts = []
    for pn in panels:
        key = pn['key']
        if single_key is not None:
            if key != single_key:
                continue
            parts.append(
                f'<div class="rp-inset rp-inset-single" data-rp-row="1" '
                f'data-rp-page="{pn["page"]}">'
                f'<span class="rp-inset-label">{pn["label"]}</span></div>')
            continue
        href = f'qualitative_trends_{key}.html'
        hidden = '' if pn['page'] == 'weather' else ' style="display:none"'
        parts.append(
            f'<div class="rp-inset" data-rp-row="{pn["row"]}" '
            f'data-rp-page="{pn["page"]}" data-rp-key="{key}"{hidden}>'
            f'<span class="rp-inset-label">{pn["label"]}</span>'
            f'<a class="rp-inset-open" target="_blank" href="{href}" '
            f'title="Open as full page" aria-label="Open as full page">↗</a>'
            f'</div>')
    return '\n'.join(parts)


def build_single(args, key, start, full, panels, cond_cf, weather_syn, wcolor):
    """Render one panel as its own full-height, chrome-free standalone page
    (``qualitative_trends_<key>.html``) — the target of an inset ↗ link."""
    pn = next((p for p in panels if p['key'] == key), None)
    if pn is None:
        valid = ', '.join(p['key'] for p in panels)
        raise SystemExit(f"unknown --panel {key!r}; choose one of: {valid}")
    # A real panel key with no data for this profile (e.g. Weight on a watch-
    # only profile): skip cleanly so the run_plots.sh standalone-page loop
    # doesn't fail, and no empty standalone page is written (its inset ↗ link
    # was dropped from the multi-panel build alongside the panel).
    if not _panel_has_data(pn, full):
        print(f'skip --panel {key}: no data for this profile')
        return
    page = pn['page']
    dates = full.index
    n_days = len(dates)

    fig: go.Figure = make_subplots(rows=1, cols=1)
    # Anchor trace keeps the cartesian subplot (and cursor-tooltip axis lookups)
    # alive for the image/shape-only envelope panels.
    fig.add_trace(go.Scatter(
        x=[dates[0], dates[-1]], y=[0.0, 0.0], mode='markers',
        marker=dict(opacity=0), hoverinfo='skip', showlegend=False),
        row=1, col=1)

    _, _, tab, drange = render_panel(fig, pn, 1, full, dates, cond_cf=cond_cf,
                                     weather_syn=weather_syn, wcolor=wcolor,
                                     visible=True)

    # Standalone page: x-axis is exactly this panel's data extent (no shared
    # axis to line up with). Falls back to the full window if the panel is empty.
    x0, x1 = drange if drange is not None else (dates[0], dates[-1])

    kw = dict(title=dict(text=pn['unit'], font=dict(color=FG_DIM, size=11)),
              gridcolor=GRID, zerolinecolor=GRID, range=pn['y_range'])
    if pn['key'] == 'time':
        kw['tickmode'] = 'array'
        kw['tickvals'] = TIME_TICKVALS
        kw['ticktext'] = TIME_TICKTEXT
    fig.update_yaxes(row=1, col=1, **kw)
    fig.update_xaxes(**yearly_x_axis_kwargs(str(x0.date()), str(x1.date())))
    apply_default_layout(
        fig, font=dict(color=FG, size=12),
        margin=dict(t=20, l=70, r=40, b=28),
        showlegend=False, hovermode=False)

    epoch = pd.Timestamp('1970-01-01')
    # Payload arrays are indexed from the calendar start (full `dates`); the
    # tooltip hover clamp is tightened to the visible data window.
    first_day = int((pd.Timestamp(start) - epoch).days)
    vis_first = int((pd.Timestamp(x0) - epoch).days)
    vis_last = int((pd.Timestamp(x1) - epoch).days)

    # Same payload shape as the multi-panel build, populated for this one panel
    # only. buildTooltip reads window.__rpActiveTab (= this panel's page) and
    # window.__rpSingle (= this key) to render just the one metric's rows.
    pay = {'weather': {'avg': {}, 'lo': {}, 'hi': {}, 'day': {}},
           'other': {'avg': {}, 'lo': {}, 'hi': {}, 'day': {}}}
    for grp in ('avg', 'lo', 'hi', 'day'):
        pay[page][grp].update(tab[grp])
    payload = {
        'first_day': first_day,
        'n_days': n_days,
        'range_window': RANGE_WINDOW,
        'ma_window': MA_WINDOW,
        'city': _str_arr(full['city_state']) if 'city_state' in full else None,
        'tabs': pay,
    }

    sib = Path(__file__).with_suffix('.js')
    insets = _inset_html(panels, single_key=key)
    single_js = (f'<script>\nwindow.__rpActiveTab = {json.dumps(page)};\n'
                 f'window.__rpSingle = {json.dumps(key)};\n</script>')

    out_path = os.path.join(args.out_dir, f'qualitative_trends_{key}.html')
    render_plot(
        fig, out_path,
        title_slug=f'qualitative_trends_{key}',
        page_title=f'Misc. Trends — {pn["label"]}',
        # No title bar: "just that trend graph, no top bar." The inset label is
        # the only chrome; --rp-title-h:0 lets the plot fill the iframe.
        title=None,
        overlay_html=insets + '\n' + single_js,
        overlay_js_files=[str(sib)],
        extra_head_css=':root{--rp-title-h:0px;}\n'
                       '.rp-tooltip .tt-sep{height:1px;margin:6px 0;'
                       'background:#444;}',
        cursor_tooltip=CursorTooltip(
            payload=payload,
            build_js=_BUILD_JS,
            first_day=vis_first,
            last_day=vis_last,
        ),
        plotly_config={'staticPlot': True},
    )
    print(f'wrote {out_path}')


def _build_page_fig(page_panels, full, dates, start, *, cond_cf, weather_syn,
                    wcolor):
    """A complete, self-contained figure for ONE page (Weather or Other).

    Each page is a clean ``make_subplots(rows=len(panels))`` — the exact shape
    the single-page version used and the only one that reliably renders
    gridlines. The toggle swaps whole figures (Plotly.newPlot), so axes are
    never resized at runtime (which never redraws gridlines). Returns
    ``(fig, tab)`` where ``tab`` is this page's cursor-tooltip payload fragment.
    """
    k = max(len(page_panels), 1)
    for i, pn in enumerate(page_panels, start=1):
        pn['row'] = i
    fig = make_subplots(rows=k, cols=1, shared_xaxes=True,
                        vertical_spacing=VERTICAL_SPACING)
    # One invisible anchor trace per row keeps each image/shape-only subplot
    # (and the cursor-tooltip axis lookups) alive.
    for r in range(1, k + 1):
        fig.add_trace(go.Scatter(
            x=[dates[0], dates[-1]], y=[0.0, 0.0], mode='markers',
            marker=dict(opacity=0), hoverinfo='skip', showlegend=False),
            row=r, col=1)
    tab = {'avg': {}, 'lo': {}, 'hi': {}, 'day': {}}
    for pn in page_panels:
        _, _, ptab, _ = render_panel(
            fig, pn, pn['row'], full, dates,
            cond_cf=cond_cf, weather_syn=weather_syn, wcolor=wcolor, visible=True)
        for grp in ('avg', 'lo', 'hi', 'day'):
            tab[grp].update(ptab[grp])
    # The make_subplots default domains already fill the height in k equal bands
    # — no override. Bake range/ticks/grid; only the bottom row keeps x labels.
    for pn in page_panels:
        ykw = dict(range=pn['y_range'],
                   title=dict(text=pn['unit'], font=dict(color=FG_DIM, size=11)),
                   gridcolor=GRID, zerolinecolor=GRID)
        if pn['key'] == 'time':
            ykw.update(tickmode='array', tickvals=TIME_TICKVALS,
                       ticktext=TIME_TICKTEXT)
        fig.update_yaxes(row=pn['row'], col=1, **ykw)
    fig.update_xaxes(**yearly_x_axis_kwargs(start, str(dates.max().date())))
    apply_default_layout(
        fig, font=dict(color=FG, size=12),
        margin=dict(t=20, l=70, r=40, b=28), showlegend=False, hovermode=False)
    return fig, tab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--daily', default=DEFAULT_DAILY)
    ap.add_argument('--altitude', default=DEFAULT_ALT)
    ap.add_argument('--time', default=DEFAULT_TIME)
    ap.add_argument('--cf', default=DEFAULT_CF)
    ap.add_argument('--out-dir', default=DEFAULT_OUT)
    ap.add_argument('--start', default=None)
    ap.add_argument('--panel', default=None,
                    help='Build a single-panel standalone page for this panel '
                         'key (conditions/temp/humidity/wind/volume/altitude/'
                         'time/weight) instead of the full 8-panel figure.')
    args = ap.parse_args()

    start, full, panels, cond_cf, weather_syn, wcolor = _prepare(args)
    if args.panel:
        build_single(args, args.panel, start, full, panels,
                     cond_cf, weather_syn, wcolor)
        return

    dates = full.index
    n_days = len(dates)

    # Drop panels with no data for this profile (e.g. a watch-only profile logs
    # no body weight), then build each page as its OWN complete figure. A page
    # with 3 panels is a clean 3-row figure that fills the height; 4 panels, a
    # 4-row figure. The toggle swaps whole figures (Plotly.newPlot), so neither
    # page ever resizes axes at runtime — the thing that blanked the gridlines
    # in every shared-figure attempt.
    panels = [pn for pn in panels if _panel_has_data(pn, full)]
    weather_panels = [p for p in panels if p['page'] == 'weather']
    other_panels = [p for p in panels if p['page'] == 'other']

    figs = {}
    pay = {'weather': {'avg': {}, 'lo': {}, 'hi': {}, 'day': {}},
           'other': {'avg': {}, 'lo': {}, 'hi': {}, 'day': {}}}
    for page, page_panels in (('weather', weather_panels),
                              ('other', other_panels)):
        if not page_panels:
            continue
        pfig, ptab = _build_page_fig(page_panels, full, dates, start,
                                     cond_cf=cond_cf, weather_syn=weather_syn,
                                     wcolor=wcolor)
        figs[page] = pfig
        pay[page] = ptab

    epoch = pd.Timestamp('1970-01-01')
    first_day = int((pd.Timestamp(start) - epoch).days)
    last_day = first_day + n_days - 1

    payload = {
        'first_day': first_day,
        'n_days': n_days,
        'range_window': RANGE_WINDOW,
        'ma_window': MA_WINDOW,
        'city': _str_arr(full['city_state']) if 'city_state' in full else None,
        'tabs': pay,
        # Surviving panel keys per page — the tooltip skips dropped metrics
        # (e.g. Weight on a watch-only profile) so it matches the panels shown.
        'present': {pg: [p['key'] for p in panels if p['page'] == pg]
                    for pg in ('weather', 'other')},
    }

    # Serialize EVERY page's figure into a JS global so the toggle can swap it
    # into the one plot div via Plotly.newPlot (a fresh render — gridlines lay
    # down exactly like a normal page load). The Weather figure is also the
    # primary one render_plot embeds; the others are newPlot targets.
    import plotly.io as pio
    figs_json = {pg: json.loads(pio.to_json(f)) for pg, f in figs.items()}
    plot_config = {'responsive': True, 'displayModeBar': False,
                   'staticPlot': True}

    sib = Path(__file__).with_suffix('.js')
    toggle = widgets.toggle_bar(
        'trends-toggle',
        [('weather', 'Weather'), ('other', 'Other')],
        default_id='weather')
    globals_html = widgets.js_globals({'TRENDS_FIGS': figs_json,
                                       'TRENDS_CONFIG': plot_config})
    insets = _inset_html(panels)

    primary = figs.get('weather') or next(iter(figs.values()))
    out_path = os.path.join(args.out_dir, 'qualitative_trends.html')
    render_plot(
        primary, out_path,
        title_slug='qualitative_trends',
        page_title='Misc. Trends',
        title='Miscellaneous Trends',
        subtitle='Weather &amp; training conditions — moving-average trends '
                 'with rolling min-max envelopes',
        overlay_html=toggle + '\n' + globals_html + '\n' + insets,
        overlay_js_files=[str(sib)],
        extra_head_css='.rp-tooltip .tt-sep{height:1px;margin:6px 0;'
                       'background:#444;}',
        cursor_tooltip=CursorTooltip(
            payload=payload,
            build_js=_BUILD_JS,
            first_day=first_day,
            last_day=last_day,
            spike_full_plot=True,
        ),
        # Fully non-interactive: no native zoom/pan/double-click. The toggle
        # (Plotly.newPlot swap), cursor tooltip, and spike are all custom — none
        # rely on Plotly's drag interactions.
        plotly_config={'staticPlot': True},
    )
    print(f'wrote {out_path}')
    print(f'  panels={len(panels)} '
          f'figs={ {pg: len(f.data) for pg, f in figs.items()} } days={n_days}')


_BUILD_JS = r"""
function buildTooltip(day) {
  var P = window.__TT_DATA;
  var idx = day - P.first_day;
  if (idx < 0 || idx >= P.n_days) return '';
  var tab = window.__rpActiveTab || 'weather';
  var T = P.tabs[tab];
  if (!T) return '';

  var DOW = ['Sun','Mon','Tue','Wed','Thu','Fri','Sat'];
  function fmt(v, n) {
    if (v === null || v === undefined || isNaN(v)) return null;
    return Number(v).toFixed(n);
  }
  function clock(m) {
    if (m === null || m === undefined || isNaN(m)) return null;
    m = Math.round(m); var h = Math.floor(m / 60), mm = m % 60;
    return h + ':' + String(mm).padStart(2, '0');
  }
  function row(label, valHtml) {
    return '<div class="tt-row"><span>' + label + '</span><span>'
         + valHtml + '</span></div>';
  }
  function muted(t) { return '<span class="tt-mute">' + t + '</span>'; }

  var dt = new Date(day * 86400000);
  var y = dt.getUTCFullYear();
  var mo = String(dt.getUTCMonth() + 1).padStart(2, '0');
  var dd = String(dt.getUTCDate()).padStart(2, '0');
  var html = '<div class="tt-date">' + y + '-' + mo + '-' + dd
           + ' (' + DOW[dt.getUTCDay()] + ')</div>';
  // Location (both tabs) — easy to cross-reference with the data.
  var city = P.city ? P.city[idx] : null;
  if (city) html += row('Location', muted(city));

  // metric display config per tab. dec = decimals (default 1); range = the
  // day value is a lo–hi band; clock = y is minute-of-day.
  var META = {
    weather: [
      {k:'conditions', label:'Conditions'},
      {k:'temp', label:'Temp', unit:'°C'},
      {k:'humidity', label:'Humidity', unit:'%'},
      {k:'wind', label:'Wind', unit:'mph'}
    ],
    other: [
      {k:'volume', label:'Volume', unit:'mi'},
      {k:'altitude', label:'Altitude', unit:'ft', range:true, dec:0},
      {k:'time', label:'Time', clock:true, range:true},
      {k:'weight', label:'Weight', unit:'lbs'}
    ]
  };
  var metrics = META[tab];
  // Skip metrics whose panel was dropped for this profile (no data), so the
  // tooltip matches the panels actually shown.
  var present = P.present && P.present[tab];
  if (present) {
    metrics = metrics.filter(function (m) { return present.indexOf(m.k) >= 0; });
  }
  // Single-panel standalone page: render only that one metric's rows.
  if (window.__rpSingle) {
    metrics = metrics.filter(function (m) { return m.k === window.__rpSingle; });
  }
  function dec(m) { return (m.dec == null) ? 1 : m.dec; }

  function dayValue(m) {
    if (m.k === 'conditions') {
      var w = T.day.weather ? T.day.weather[idx] : null;
      var c = T.day.conditions ? T.day.conditions[idx] : null;
      if (!w) return null;
      var s = w; if (c) s += ' / ' + c; else s += ' / dry';
      return s;
    }
    if (m.range) {                              // day = the band [lo, hi]
      var lo = T.day[m.k + '_lo'] ? T.day[m.k + '_lo'][idx] : null;
      var hi = T.day[m.k + '_hi'] ? T.day[m.k + '_hi'][idx] : null;
      if (m.clock) {
        var a = clock(lo), b = clock(hi);
        if (!(a && b)) return null;
        var estA = T.day.time_est;              // bin-derived (pre-watch) estimate
        var tail = (estA && estA[idx]) ? ' ' + muted('(estimated)') : '';
        return a + '–' + b + tail;
      }
      var fa = fmt(lo, dec(m)), fb = fmt(hi, dec(m));
      return (fa && fb) ? (fa + '–' + fb + ' ' + (m.unit||'')) : null;
    }
    var v = T.day[m.k] ? T.day[m.k][idx] : null;
    var fv = fmt(v, dec(m));
    return (fv === null) ? null : (fv + ' ' + (m.unit||''));
  }

  // Average row — same shape for every metric: <avg> unit (min to max), where
  // (min to max) is the rolling envelope (the band on the graph). For range
  // metrics the avg is the smoothed band midpoint.
  function avgValue(m) {
    if (m.k === 'conditions') return null;
    var av = T.avg[m.k] ? T.avg[m.k][idx] : null;
    var lo = T.lo[m.k] ? T.lo[m.k][idx] : null;
    var hi = T.hi[m.k] ? T.hi[m.k][idx] : null;
    if (m.clock) {
      var a = clock(av);
      if (a === null) return null;
      var lc = clock(lo), hc = clock(hi);
      var rgc = (lc && hc) ? ' (' + lc + ' to ' + hc + ')' : '';
      return '<b>' + a + '</b> ' + muted(rgc);
    }
    var fa = fmt(av, dec(m));
    if (fa === null) return null;
    var rg = (fmt(lo, dec(m)) !== null && fmt(hi, dec(m)) !== null)
             ? ' (' + fmt(lo, dec(m)) + ' to ' + fmt(hi, dec(m)) + ')' : '';
    return '<b>' + fa + '</b> ' + muted((m.unit||'') + rg);
  }

  // Day-specific block (actuals for this date)
  var dayRows = '', anyDay = false;
  for (var i = 0; i < metrics.length; i++) {
    var dv = dayValue(metrics[i]);
    if (dv === null) continue;
    anyDay = true;
    dayRows += row(metrics[i].label, dv);
  }
  // Averages block
  var avgRows = '', anyAvg = false;
  for (var j = 0; j < metrics.length; j++) {
    var av = avgValue(metrics[j]);
    if (av === null) continue;
    anyAvg = true;
    avgRows += row('Avg. ' + metrics[j].label.toLowerCase(), av);
  }
  if (!anyDay && !anyAvg) return '';
  if (anyDay) html += dayRows;
  if (anyDay && anyAvg) html += '<div class="tt-sep"></div>';
  if (anyAvg) html += avgRows;
  return html;
}
"""


if __name__ == '__main__':
    main()
