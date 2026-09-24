"""The colour-by variable set shared by the Recovery and Long Runs scatters.

Both plots let you re-encode marker colour from a dropdown. This module owns
the whole spec — which variables exist, in what order, where each one's value
comes from, and what gradient it uses — so the two plots cannot drift from each
other, and so the four environmental variables (temperature, wind, altitude,
time of day) read exactly as they do on the Misc. Trends panels.

Colours are resolved to ``#rrggbb`` strings HERE, in Python, rather than being
handed to Plotly as a numeric array + ``colorscale``. Two reasons:

* Time of day cannot be expressed as a 1-D colorscale at all. Its gradient's
  anchor POSITIONS are that day's solar times, so a 6 am run is night in
  December and full daylight in June. Only per-point colour can say that.
* One evaluation path means the sidebar legend strip and the markers can never
  disagree about what a value looks like.

Scale provenance is deliberate and differs per variable:

* **Per-plot** (distance, elevation gain) — magnitudes that scale with the run
  type being plotted. A shared scale would park every recovery run in the dark
  third of a ramp topped out by a 22-mile long run.
* **Shared** (weekly volume, wind, altitude) — properties of the DAY, not the
  run, derived from the full ``daily.csv`` calendar so the same Tuesday reads
  the same colour on either tab and on Misc. Trends.
* **Fixed** (temperature, time of day, recent race) — absolute anchors.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from src.coros.solar import solar_anchors_local
from src.shared.day_clock import canonical_run_times
from src.shared.effective_mileage import effective_daily_miles
from src.shared.plot_window import daily_floor
from src.shared.recovery_model import (per_run_altitude, days_since,
                                       quality_category_dates, long_run_context)
from src.plotting.gradients import (gradient_at, rgb_to_hex, css_gradient,
                                    solar_css_gradient, solar_ramp_from_anchors,
                                    NO_DATA_GRAY)

# Distance ramp — the Long Runs encoding, kept verbatim so that tab looks
# unchanged on load.
LR_GRAD_BLUE = '#3498DB'     # bright blue   (short)
LR_GRAD_PURPLE = '#8E44AD'   # vivid purple  (mid)
LR_GRAD_MAGENTA = '#E91E63'  # magenta       (long)

# Gray end of the one-sided ramps, as hex (elevation gain at 0 ft, a race
# 14+ days back). Same value gradient_at returns for a missing input, so
# "negligible" and "unknown" deliberately render identically.
GRAY_HEX = rgb_to_hex(NO_DATA_GRAY)

# Electric purple — the highest-chroma purple available at a luminance that
# still separates from the gray low end. Measured against the alternatives at
# Max's actual gain distribution, it is the most contrast a two-stop gray ->
# purple ramp achieves. A lighter violet (#C77DFF) sits at almost exactly the
# gray's luminance and vanishes; a darker one (#8E44AD) loses chroma.
ELEV_GAIN_MAX_COLOR = '#B026FF'
RECENT_RACE_COLOR = '#FF3B30'
RECENT_RACE_FADE_DAYS = 14.0      # fully decayed to gray at/after this

DEFAULT_KEY = 'distance'

# Canonical order. Drives the dropdown AND hover.context_lines, so the two
# always agree.
ORDER = ['distance', 'weekly_volume', 'temp', 'wind',
         'elev_gain', 'altitude', 'time_of_day', 'recent_race']


# ---------- per-run value extraction ----------

def _num(df, col):
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors='coerce')


def run_miles(df):
    """Displayed distance per run: pause-aware projected distance where the
    projection carries one, else the watch/route-corrected distance, else the
    logged one. Same chain hover.elevation_line uses, so the colour and the
    tooltip's elevation row are computed off one number."""
    from src.shared.units import METERS_PER_MILE
    mi = _num(df, 'd_m') / METERS_PER_MILE
    return mi.fillna(_num(df, 'corr_miles')).fillna(_num(df, 'miles'))


def elev_gain_ft(df):
    """Total gross vertical gain per run, in feet. Per-mile gain times the run's
    distance — ``disp_gain_pm`` (measured where the watch has it, the route's
    hand-set ``elev_per_mile`` constant otherwise) falling back to the strictly
    measured ``elev_gain_pm``. Matches hover.elevation_line exactly."""
    pm = _num(df, 'disp_gain_pm').fillna(_num(df, 'elev_gain_pm')).fillna(0.0)
    return pm * run_miles(df)


def weekly_volume(daily):
    """Date-indexed mileage over the seven days BEFORE each date (d-7..d-1,
    excluding the date itself, so it reads as load carried into the run and
    stays independent of the Distance encoding).

    daily.csv is running-only — zero-mile days are pruned — so the series is
    reindexed onto a gap-free calendar before the rolling sum, or rest days
    would silently shorten the window.

    The calendar starts at ``daily_floor()`` (the first non-race daily entry),
    NOT at the frame's first date. Before that there is no daily log at all —
    only pre-2016 race stubs — so a window there would sum to a degenerate 0
    that reads as "rested all week" when it means "not logged". Legacy 2014-15
    training days are exactly this case: they never enter daily.csv by design,
    so the Long Runs tab's legacy points get NaN and render gray. Combined with
    ``min_periods=7``, a date only gets a number once a FULL seven logged days
    precede it.
    """
    mi = pd.Series(effective_daily_miles(daily).to_numpy(),
                   index=pd.DatetimeIndex(daily['date']))
    mi = mi.groupby(level=0).sum()
    floor = daily_floor(daily)
    if mi.index.max() < floor:
        return pd.Series(dtype=float)
    cal = pd.date_range(max(mi.index.min(), floor), mi.index.max(), freq='D')
    mi = mi.reindex(cal, fill_value=0.0)
    return mi.rolling(7, min_periods=7).sum().shift(1)


def days_since_race(df, daily, races):
    """Days since the most recent race of any distance, per run."""
    cats = quality_category_dates(daily, races)
    all_races = sorted(set(cats.get('marathon', []))
                       | set(cats.get('race_short', [])))
    return pd.Series(days_since(df, all_races), index=df.index)


def solar_colors(df, daily, data_dir):
    """Per-run colour from the day's own solar gradient, sampled at the run's
    midpoint minute. Runs with neither a watch time nor a usable time-of-day
    bin come back gray."""
    times = canonical_run_times(daily, data_dir)
    idx = pd.DatetimeIndex(df['date'])
    mid = pd.Series(times['mid_min'].reindex(idx).to_numpy(), index=df.index)
    lat = pd.Series(times['lat'].reindex(idx).to_numpy(), index=df.index)
    lon = pd.Series(times['lon'].reindex(idx).to_numpy(), index=df.index)
    tz = pd.Series(times['tz_min'].reindex(idx).to_numpy(), index=df.index)

    out = []
    cache = {}
    for i, d in zip(df.index, idx):
        m = mid.at[i]
        if pd.isna(m) or pd.isna(lat.at[i]) or pd.isna(tz.at[i]):
            out.append(GRAY_HEX)
            continue
        key = (d.date(), round(float(lat.at[i]), 2),
               round(float(lon.at[i]), 2), int(tz.at[i]))
        a = cache.get(key)
        if a is None:
            a = solar_anchors_local(d.date(), float(lat.at[i]),
                                    float(lon.at[i]), int(tz.at[i]))
            cache[key] = a
        out.append(rgb_to_hex(solar_ramp_from_anchors(a, np.array([float(m)]))[0]))
    return out


# ---------- scale construction ----------

def _round_up(v, step):
    return float(np.ceil(float(v) / step) * step) if np.isfinite(v) else step


def shared_scales(daily, data_dir):
    """Anchor ranges derived from the FULL daily calendar, so every consumer
    (Recovery, Long Runs, and Misc. Trends' matching panels) puts the same
    value at the same colour. Raw maxima, not percentiles — the project's
    standing convention is that genuine outliers get invalidated by hand via
    the changes sheet rather than clipped at render time."""
    wind = pd.to_numeric(daily.get('wind_mph'), errors='coerce').dropna()
    wind_max = float(wind.max()) if len(wind) else 1.0

    # Altitude ceiling: identical construction to the Misc. Trends Altitude
    # panel (0 floor, 1.04x the highest measured daily max).
    alt_path = os.path.join(str(data_dir), 'altitude_daily.csv')
    alt_max = 1.0
    if os.path.exists(alt_path):
        a = pd.read_csv(alt_path)
        hi = pd.to_numeric(a.get('max_elev_ft'), errors='coerce').dropna()
        if len(hi):
            alt_max = float(hi.max()) * 1.04

    wk = weekly_volume(daily).dropna()
    wk_med = float(wk.median()) if len(wk) else 1.0
    wk_max = float(wk.max()) if len(wk) else 1.0
    return {'wind_max': wind_max, 'alt_max': alt_max,
            'weekly_med': wk_med, 'weekly_max': wk_max}


def _spec(df, daily, races, data_dir):
    """(key -> dict) of label, unit, values, anchors, ticks for every variable.

    ``ticks`` are ``(position_fraction, label)`` pairs positioned along the
    legend strip; all but Time of day use the two endpoints.
    """
    sc = shared_scales(daily, data_dir)

    # Anchors at the RAW min/max (what the old per-trace cmin/cmax were), so
    # the Long Runs tab's default view is pixel-for-pixel what it was; only the
    # printed end labels round outward.
    dist = run_miles(df)
    d_lo = float(dist.min()) if dist.notna().any() else 0.0
    d_hi = float(dist.max()) if dist.notna().any() else 1.0
    if d_hi <= d_lo:
        d_hi = d_lo + 1.0

    gain = elev_gain_ft(df)
    g_hi = float(gain.max()) if gain.notna().any() else 1.0
    g_hi = _round_up(max(g_hi, 1.0), 100.0)

    wk = weekly_volume(daily).reindex(pd.DatetimeIndex(df['date']))
    wk = pd.Series(wk.to_numpy(), index=df.index)

    alt_ft = per_run_altitude(df) * 1000.0
    dsr = days_since_race(df, daily, races)

    return {
        'distance': dict(
            label='Distance', unit='mi', values=dist,
            anchors=[(d_lo, LR_GRAD_BLUE),
                     ((d_lo + d_hi) / 2.0, LR_GRAD_PURPLE),
                     (d_hi, LR_GRAD_MAGENTA)],
            ticks=[(0.0, f'{np.floor(d_lo):.0f}'), (1.0, f'{np.ceil(d_hi):.0f}')]),
        'weekly_volume': dict(
            label='Weekly volume', unit='mi', values=wk,
            anchors=[(0.0, '#3D2208'), (sc['weekly_med'], '#8B5C16'),
                     (sc['weekly_max'], '#F2D034')],
            ticks=[(0.0, '0'), (1.0, f"{sc['weekly_max']:.0f}")]),
        'temp': dict(
            label='Temperature', unit='°C', values=_num(df, 'temp_c'),
            # The Misc. Trends ramp verbatim — the running log's own
            # conditional-formatting colours darkened ~30%. Recovery used to
            # carry the undarkened originals; brightening the MARKERS instead
            # (and dimming the no-data gray toward the background) buys the
            # same legibility without a second temperature palette to keep in
            # sync.
            anchors=[(-10.0, '#0E7BAA'), (22.0, '#5A9E3D'), (40.0, '#C82020')],
            ticks=[(0.0, '−10'), (1.0, '40')]),
        'wind': dict(
            label='Wind', unit='mph', values=_num(df, 'wind_mph'),
            anchors=[(0.0, '#7BB050'), (sc['wind_max'] / 2, '#D4B028'),
                     (sc['wind_max'], '#C0342A')],
            ticks=[(0.0, '0'), (1.0, f"{sc['wind_max']:.0f}")]),
        'elev_gain': dict(
            label='Elevation gain', unit='ft', values=gain,
            anchors=[(0.0, GRAY_HEX), (g_hi, ELEV_GAIN_MAX_COLOR)],
            ticks=[(0.0, '0'), (1.0, f'{g_hi:.0f}')]),
        'altitude': dict(
            label='Altitude', unit='ft', values=alt_ft,
            anchors=[(0.0, '#2E7D32'), (sc['alt_max'] / 2.0, '#8B5A2B'),
                     (sc['alt_max'], '#FFFFFF')],
            ticks=[(0.0, '0'), (1.0, f"{sc['alt_max']:.0f}")]),
        'time_of_day': dict(
            label='Time of day', unit='', values=None, anchors=None,
            # An idealized equinox day, NOT a real one: point colours use each
            # run's own date and city, so no fixed clock label would be right
            # for more than a few weeks a year. Two qualitative marks only.
            gradient=solar_css_gradient(),
            ticks=[(0.25, 'Dawn'), (0.75, 'Dusk')]),
        'recent_race': dict(
            label='Recent race', unit='', values=dsr,
            anchors=[(1.0, RECENT_RACE_COLOR),
                     (RECENT_RACE_FADE_DAYS, GRAY_HEX)],
            ticks=[(0.0, '1d'), (1.0, f'{RECENT_RACE_FADE_DAYS:.0f}d+')]),
    }


def build_color_by(df, data_dir, *, daily=None, races=None):
    """Payload for ``widgets.js_globals({'COLOR_BY': ...})`` plus the sidebar.

    ``df`` is the per-run frame being plotted (needs a ``date`` column).
    ``daily`` / ``races`` are loaded from ``data_dir`` when not supplied —
    Long Runs reads neither today, Recovery has both in hand already.
    """
    data_dir = str(data_dir)
    if daily is None:
        daily = pd.read_csv(os.path.join(data_dir, 'daily.csv'),
                            parse_dates=['date'])
    if races is None:
        races = pd.read_csv(os.path.join(data_dir, 'races.csv'),
                            parse_dates=['date'])

    spec = _spec(df, daily, races, data_dir)
    solar = solar_colors(df, daily, data_dir)

    out = {'order': ORDER, 'default': DEFAULT_KEY, 'vars': {}}
    for key in ORDER:
        s = spec[key]
        if key == 'time_of_day':
            colors = solar
            gradient = s['gradient']
        else:
            anchors = s['anchors']
            colors = [rgb_to_hex(gradient_at(
                None if v is None or not np.isfinite(v) else float(v), anchors))
                for v in s['values'].to_numpy(dtype=float)]
            gradient = css_gradient(anchors)
        title = f"{s['label']} ({s['unit']})" if s['unit'] else s['label']
        out['vars'][key] = {
            'label': s['label'],
            'title': title,
            'colors': colors,
            'gradient': gradient,
            'ticks': [[float(p), t] for p, t in s['ticks']],
        }
    return out


# ---------- sidebar UI ----------

SCAFFOLD_JS = str((__import__('pathlib').Path(__file__).resolve().parent
                   / '_scaffold' / 'color_by.js'))


def color_by_ui(payload):
    """Legend strip + dropdown, rendered for the payload's default variable.

    Goes at the TOP of the plot's sidebar. The strip's static HTML matches the
    colours the figure ships with, so nothing has to run at load."""
    from src.plotting import widgets
    d = payload['vars'][payload['default']]
    return (
        widgets.gradient_strip('cb-strip', title_text=d['title'],
                               gradient=d['gradient'], ticks=d['ticks'])
        + '\n'
        + widgets.select(
            'cb-select',
            [(k, payload['vars'][k]['label']) for k in payload['order']],
            data_attr='colorby', selected=payload['default'])
    )


def default_colors(payload):
    """The colour array the figure is built with (the payload's default)."""
    return payload['vars'][payload['default']]['colors']


def attach_hover_columns(df, data_dir, *, daily=None, races=None):
    """Add the columns ``hover.context_lines`` needs that are not on the
    plotted frame: ``_wk_vol`` (mi), ``_alt_ft``, ``dsq_<cat>`` (days since
    the last race of each fatigue category) and ``dsq_long`` / ``long_prev_mi``
    (days since, and canonical miles of, the last long run). Returns a copy.

    Recovery already carries ``dsq_*`` from the model fit; re-deriving them
    here is idempotent and keeps the Long Runs call site to one line."""
    data_dir = str(data_dir)
    if daily is None:
        daily = pd.read_csv(os.path.join(data_dir, 'daily.csv'),
                            parse_dates=['date'])
    if races is None:
        races = pd.read_csv(os.path.join(data_dir, 'races.csv'),
                            parse_dates=['date'])
    out = df.copy()
    wk = weekly_volume(daily).reindex(pd.DatetimeIndex(out['date']))
    out['_wk_vol'] = wk.to_numpy()
    out['_alt_ft'] = (per_run_altitude(out) * 1000.0).to_numpy()
    cats = quality_category_dates(daily, races)
    for cat, dates in cats.items():
        out[f'dsq_{cat}'] = days_since(out, dates)
    from src.shared.workouts import long_run_miles
    out['dsq_long'], out['long_prev_mi'] = long_run_context(out, long_run_miles(daily))
    return out
