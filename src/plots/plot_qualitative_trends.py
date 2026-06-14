"""Qualitative trends plot: volume, temperature, weight.

Three subplots stacked vertically with shared x-axis (2016-01-01 → present).
Each subplot shows:
  - 14-day rolling min/max envelope, drawn as N horizontal gradient strips.
    Each strip is a fill='tonexty' Scatter pair whose lo/hi are clipped to
    (lo_smooth, hi_smooth) intersected with the strip's y-band, and NaN'd
    where the strip doesn't overlap the envelope. Smooth diagonal boundaries
    come from lo_smooth/hi_smooth following the data; the vertical gradient
    comes from N strips at fixed y-bands, each colored by gradient(y_mid).
  - Continuous gradient trendline (50 line segments).
  - Per-metric locked moving-average window:
      Volume = 56d, Temp = 28d, Weight = 56d

Color via Running Log 2025 conditional formatting:
  Volume:  0 #FFF2CC → max #F1C232
  Temp:    -10 #00B0F0 → 22 #92D050 → 40 #FF0000
  Weight:  145 #FFFFFF → 170 #E69138

Hover (custom):
  mousemove on plot div + fixed-position spike line + tooltip overlay.
  Variables with no data at the hovered day are hidden from the tooltip.

Weight MA visibility:
  MA line is hidden on days where weight_interp is null (>7d gaps).
"""
import argparse
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.shared.paths import DATA_DIR, OUTPUT_DIR
from src.shared.plot_window import daily_floor
from src.shared.effective_mileage import effective_daily_miles
from src.plotting import (render_plot, CursorTooltip, apply_default_layout,
                            FG, FG_DIM, GRID,
                            yearly_x_axis_kwargs)


DEFAULT_DAILY = str(DATA_DIR / 'daily.csv')
DEFAULT_OUT = str(OUTPUT_DIR)
START_DATE = '2016-01-01'

MA_WINDOW = {'volume': 56, 'temp': 28, 'sleep': 28, 'weight': 56}

RANGE_WINDOW = 14
RANGE_SMOOTH = 7
WEIGHT_INTERP_MAX_GAP = 7

N_ENVELOPE_STRIPS = 40        # vertical gradient resolution per subplot
ENVELOPE_X_STRIDE = 4         # x stride for strip traces (smoothed data, no visual loss)
ENVELOPE_UPSAMPLE = 8         # sub-daily x resolution for short profiles (kills strip-edge verticals)
ENVELOPE_ALPHA = 1.0          # fully opaque
STRIP_OVERLAP_FRAC = 0.005    # small overlap to prevent hairline AA gaps


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


def load_series(daily_path, start_date):
    df = pd.read_csv(daily_path, parse_dates=['date'])
    df = df[df['date'] >= pd.Timestamp(start_date)].copy()
    # Source of truth: watch/route distance-corrected mileage (decrease-only;
    # corr <= logged) drives the volume trend. On-disk 'miles' is untouched.
    df['miles'] = effective_daily_miles(df)
    end = df['date'].max()
    cal = pd.DataFrame({'date': pd.date_range(start_date, end, freq='D')})
    keep = df[['date', 'miles', 'temp_c', 'sleep_cycles', 'weight_lbs']]
    full = cal.merge(keep, on='date', how='left').set_index('date')
    full['miles'] = full['miles'].fillna(0.0)
    full['sleep_hours'] = full['sleep_cycles'] * 1.5
    full['weight_interp'] = interpolate_short_gaps(
        full['weight_lbs'], WEIGHT_INTERP_MAX_GAP)
    return full


def rolling_minmax_raw(s, window):
    lo = s.rolling(window, min_periods=1, center=True).min()
    hi = s.rolling(window, min_periods=1, center=True).max()
    return lo, hi


def smooth_series(s, window):
    return s.rolling(window, min_periods=1, center=True).mean()


def rolling_ma(s, window):
    # Gaussian-weighted MA: same window as a uniform rolling mean, but
    # the per-day weights taper at the edges, killing per-pixel jitter
    # without widening the effective kernel meaningfully (σ ≈ window/7
    # → ~4d for 28d, ~8d for 56d).
    sigma = max(2.0, window / 7)
    return s.rolling(window, min_periods=max(1, window // 4),
                     center=True, win_type='gaussian').mean(std=sigma)


# ---------- envelope strips ----------

def _strip_polygon(dates_list, lo_arr, hi_arr):
    """Build a multi-region polygon path for a single strip.

    Returns (xs, ys) where each contiguous valid run becomes one closed
    sub-polygon (forward along hi, backward along lo). Sub-polygons are
    separated by `None` so plotly's fill='toself' draws them independently.
    """
    n = len(dates_list)
    xs, ys = [], []
    i = 0
    while i < n:
        while i < n and (np.isnan(lo_arr[i]) or np.isnan(hi_arr[i])):
            i += 1
        if i >= n:
            break
        j = i
        while j < n and not (np.isnan(lo_arr[j]) or np.isnan(hi_arr[j])):
            j += 1
        # Forward along hi
        for k in range(i, j):
            xs.append(dates_list[k])
            ys.append(float(hi_arr[k]))
        # Backward along lo
        for k in range(j - 1, i - 1, -1):
            xs.append(dates_list[k])
            ys.append(float(lo_arr[k]))
        # Subpath separator
        xs.append(None)
        ys.append(None)
        i = j
    return xs, ys


def upsample_envelope(dates, lo_smooth, hi_smooth, factor):
    """Resample the smooth envelope edges onto a `factor`x-finer x-grid by
    linear interpolation between daily points.

    The gradient is drawn as horizontal color strips; each strip switches on
    where its y-band first falls under the envelope. Sampled only at daily x,
    a steep one-day rise makes several strips switch on at the *same* x, and
    their vertical closing edges stack into a visible vertical riser. On a
    finer grid each band-crossing lands at its true x, so those risers spread
    into diagonals — the edge becomes straight lines between points, no
    verticals. NaN gaps (weight's >7d stretches) are preserved: a fine point
    is NaN whenever either bracketing daily point is NaN, so gaps never bridge.
    """
    # Day offsets from the start — robust to the index's datetime resolution
    # (ns/us/s); reconstructing via raw epoch ints would mis-scale otherwise.
    day0 = dates[0]
    o = ((dates - day0) / np.timedelta64(1, 'D')).to_numpy(dtype=np.float64)
    fine_o = np.linspace(o[0], o[-1], (len(dates) - 1) * factor + 1)
    fine_dates = day0 + pd.to_timedelta(fine_o, unit='D')

    def up(s):
        v = s.to_numpy(dtype=np.float64)
        finite = np.isfinite(v)
        if finite.all():
            out = np.interp(fine_o, o, v)
        else:
            out = np.interp(fine_o, o[finite], v[finite])
            idx = np.clip(np.searchsorted(o, fine_o), 1, len(o) - 1)
            out[~finite[idx - 1] | ~finite[idx]] = np.nan
        return pd.Series(out, index=fine_dates)

    return fine_dates, up(lo_smooth), up(hi_smooth)


def add_envelope_strips(fig, dates, lo_smooth, hi_smooth, anchors,
                          n_strips, x_stride, row_i, alpha):
    """Add n_strips Scatter traces forming a gradient-filled envelope.

    Each strip k is a horizontal y-band [edges[k], edges[k+1]] (with tiny
    overlap to hide hairline seams). At each x:
      lo_arr[x] = max(strip_bottom, lo_smooth[x])
      hi_arr[x] = min(strip_top,    hi_smooth[x])
      where lo_arr >= hi_arr, both → NaN
    Each strip is rendered as a single fill='toself' Scatter whose path
    walks forward along hi then back along lo for every contiguous valid
    region. None separators ensure multi-region strips render as multiple
    independent polygons (no spurious horizontals across gaps).
    """
    dates_ds = list(dates[::x_stride])
    lo_v = lo_smooth.iloc[::x_stride].values
    hi_v = hi_smooth.iloc[::x_stride].values
    valid = ~(np.isnan(lo_v) | np.isnan(hi_v))
    if not valid.any():
        return
    y_min = float(np.nanmin(lo_v))
    y_max = float(np.nanmax(hi_v))
    if y_max <= y_min:
        return
    edges = np.linspace(y_min, y_max, n_strips + 1)
    overlap = (y_max - y_min) * STRIP_OVERLAP_FRAC

    for k in range(n_strips):
        y_strip_lo = edges[k] - overlap
        y_strip_hi = edges[k + 1] + overlap
        y_mid = (edges[k] + edges[k + 1]) / 2

        lo_arr = np.maximum(y_strip_lo, lo_v).astype(np.float64)
        hi_arr = np.minimum(y_strip_hi, hi_v).astype(np.float64)
        gone = (lo_arr >= hi_arr) | (~valid)
        lo_arr[gone] = np.nan
        hi_arr[gone] = np.nan

        xs, ys = _strip_polygon(dates_ds, lo_arr, hi_arr)
        if not xs:
            continue

        r, g, b = gradient_at(y_mid, anchors)
        fillcolor = f'rgba({r},{g},{b},{alpha})'

        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode='lines',
            line=dict(color='rgba(0,0,0,0)', width=0),
            fill='toself', fillcolor=fillcolor,
            hoverinfo='skip', showlegend=False,
            connectgaps=False,
        ), row=row_i, col=1)


# ---------- continuous gradient trendline ----------

def add_trendline(fig, dates, ma_values, row_i, line_width=2.2):
    """Single white trendline trace. NaN values break the line cleanly
    (used for weight's >7d gaps)."""
    fig.add_trace(go.Scatter(
        x=dates, y=ma_values, mode='lines',
        line=dict(color='#ffffff', width=line_width),
        hoverinfo='skip', showlegend=False,
        connectgaps=False,
    ), row=row_i, col=1)


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--daily', default=DEFAULT_DAILY)
    ap.add_argument('--out-dir', default=DEFAULT_OUT)
    ap.add_argument('--start', default=None,
                    help='Left date bound (default: first non-race daily entry)')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    start = args.start or str(daily_floor().date())
    full = load_series(args.daily, start)
    dates = full.index
    miles_max = float(full['miles'].max())

    # Envelope x-stride: daily for short ranges (the 4-day stride stairsteps
    # visibly on a few-month profile like a new watch import), 4-day beyond
    # 1000 days where the difference is imperceptible and traces stay light.
    short = len(dates) <= 1000
    x_stride = 1 if short else ENVELOPE_X_STRIDE

    def env_smooth(s):
        # Smooth the rolling min/max envelope edges. The min/max are plateau
        # step functions (an extreme persists across RANGE_WINDOW), which on a
        # dense decade-wide axis reads smooth but on a short (few-month) axis
        # shows as visible stair treads. For short profiles, widen the window
        # and use a gaussian kernel (round corners, no boxcar facets); dense
        # profiles keep the original 7-day boxcar exactly (Max unchanged).
        if not short:
            return smooth_series(s, RANGE_SMOOTH)
        w = max(RANGE_SMOOTH, round(len(dates) * 0.12))
        return s.rolling(w, min_periods=1, center=True,
                         win_type='gaussian').mean(std=max(2.0, w / 3))

    metrics: list[dict[str, Any]] = [
        dict(key='volume', label='Daily volume', unit='mi',
             series=full['miles'],
             # 3-stop yellow-family gradient. Mid-stop at 8 mi places
             # dramatic color shift across the typical envelope (~3-12 mi).
             anchors=[(0.0, '#3D2208'), (8.0, '#8B5C16'),
                      (miles_max, '#F2D034')]),
        dict(key='temp', label='Temperature', unit='°C',
             series=full['temp_c'],
             # Same Excel-defaults palette but each anchor darkened
             # ~30% so white trendline pops at full opacity.
             anchors=[(-10.0, '#0E7BAA'), (22.0, '#5A9E3D'),
                      (40.0, '#C82020')]),
        dict(key='sleep', label='Sleep', unit='hr',
             series=full['sleep_hours'],
             # CF source: gradients.sleep_cycles — 0c #C00000 → 8c
             # #00B050, converted to hours (×1.5). Hues preserved from
             # the spreadsheet; brightness left at CF level so the band
             # matches Temp's #C82020 / #5A9E3D anchors rather than
             # reading muted next to them.
             anchors=[(0.0, '#C00000'), (12.0, '#00B050')]),
        dict(key='weight', label='Weight', unit='lbs',
             series=full['weight_interp'],
             # 3-stop orange-family gradient. Mid-stop at 156 lbs places
             # dramatic color shift across the typical envelope (152-160).
             anchors=[(145.0, '#2D1006'), (156.0, '#8C4F1F'),
                      (170.0, '#E89535')]),
    ]

    # Drop metrics with no data so an empty trend doesn't take up a subplot —
    # e.g. a watch-import profile has no sleep or weight to populate. Volume is
    # always present (rest days are 0, not null), so at least one row remains.
    metrics = [m for m in metrics if m['series'].notna().any()]
    dropped = {'sleep', 'temp', 'weight', 'volume'} - {m['key'] for m in metrics}
    if dropped:
        print(f"[trends] no data for {sorted(dropped)} — omitting those subplots")

    fig: go.Figure = make_subplots(
        rows=len(metrics), cols=1, shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=[
            f"{m['label']} ({MA_WINDOW[m['key']]}-day trend, min-max gradient)"
            for m in metrics
        ],
    )

    payload_lo = {}
    payload_hi = {}
    payload_ma = {}

    for row_i, m in enumerate(metrics, start=1):
        lo_raw, hi_raw = rolling_minmax_raw(m['series'], RANGE_WINDOW)
        ma = rolling_ma(m['series'], MA_WINDOW[m['key']])

        # Gap alignment: where the underlying value is null, force every
        # derived series to null so the trendline and envelope start/end
        # together at gap boundaries. Currently only weight has these gaps
        # (>WEIGHT_INTERP_MAX_GAP day stretches without a measurement);
        # volume has no nulls (rest days = 0) and temp's per-rest-day
        # nulls are filled by the rolling window's neighbors anyway.
        if m['key'] == 'weight':
            valid = m['series'].notna()
            lo_raw = lo_raw.where(valid, other=np.nan)
            hi_raw = hi_raw.where(valid, other=np.nan)
            ma = ma.where(valid, other=np.nan)

        lo_smooth = env_smooth(lo_raw)
        hi_smooth = env_smooth(hi_raw)

        # Re-mask post-smoothing — the rolling mean's min_periods=1 can
        # spread valid neighbours back into the gap.
        if m['key'] == 'weight':
            lo_smooth = lo_smooth.where(valid, other=np.nan)
            hi_smooth = hi_smooth.where(valid, other=np.nan)

        # Envelope as gradient strip stack. Short profiles render on a
        # sub-daily x-grid so strip band-crossings become diagonals, not
        # stacked verticals (see upsample_envelope); dense profiles stride.
        if short:
            ev_dates, ev_lo, ev_hi = upsample_envelope(
                dates, lo_smooth, hi_smooth, ENVELOPE_UPSAMPLE)
            ev_stride = 1
        else:
            ev_dates, ev_lo, ev_hi = dates, lo_smooth, hi_smooth
            ev_stride = x_stride
        add_envelope_strips(
            fig, ev_dates, ev_lo, ev_hi, m['anchors'],
            N_ENVELOPE_STRIPS, ev_stride, row_i, ENVELOPE_ALPHA)

        # Trendline (pure white for contrast against gradient envelope).
        add_trendline(fig, dates, ma, row_i)

        y_min_env = float(lo_smooth.min())
        y_max_env = float(hi_smooth.max())
        pad = (y_max_env - y_min_env) * 0.04 if y_max_env > y_min_env else 1.0
        fig.update_yaxes(
            range=[y_min_env - pad, y_max_env + pad],
            row=row_i, col=1,
            title=dict(text=m['unit'],
                       font=dict(color=FG_DIM, size=11)),
            gridcolor=GRID, zerolinecolor=GRID,
        )

        def _round_arr(s, n=2):
            return [None if (v is None
                             or (isinstance(v, float) and np.isnan(v)))
                    else round(float(v), n)
                    for v in s.values]
        payload_lo[m['key']] = _round_arr(lo_raw)
        payload_hi[m['key']] = _round_arr(hi_raw)
        payload_ma[m['key']] = _round_arr(ma)

    # Yearly gridlines — shared helper, standard tick styling.
    fig.update_xaxes(**yearly_x_axis_kwargs(start, str(dates.max().date())))

    apply_default_layout(
        fig,
        font=dict(color=FG, size=12),
        margin=dict(t=24, l=70, r=40, b=56),
        showlegend=False,
        hovermode=False,
    )
    fig.update_annotations(font=dict(color=FG, size=13))

    epoch = pd.Timestamp('1970-01-01')
    first_day = int((pd.Timestamp(start) - epoch).days)
    last_day = first_day + len(dates) - 1

    payload = {
        'first_day': first_day,
        'n_days': len(dates),
        'lo': payload_lo,
        'hi': payload_hi,
        'ma': payload_ma,
        'ma_window': MA_WINDOW,
        'range_window': RANGE_WINDOW,
        # Only the metrics actually plotted, in row order — the tooltip JS
        # iterates this rather than a fixed list so omitted trends don't break it.
        'metrics': [m['key'] for m in metrics],
    }

    shown = ', '.join(m['label'].lower() for m in metrics)

    out_path = os.path.join(args.out_dir, 'qualitative_trends.html')
    render_plot(
        fig, out_path,
        title_slug='qualitative_trends',
        page_title='Volume / temp / weight',
        title='Miscellaneous Trends',
        subtitle=f'{shown[:1].upper()}{shown[1:]}: moving-average trendlines '
                 'with 14-day rolling min-max envelopes',
        cursor_tooltip=CursorTooltip(
            payload=payload,
            build_js=_BUILD_JS,
            first_day=first_day,
            last_day=last_day,
            spike_full_plot=True,
        ),
    )
    print(f'wrote {out_path}')

    print('\nseries summaries (post-shaping):')
    print(f"  volume  rest-days={int((full['miles']==0).sum())}  "
          f"min={full['miles'].min():.1f}  max={full['miles'].max():.1f}  "
          f"median(running)={full['miles'][full['miles']>0].median():.1f}")
    t = full['temp_c'].dropna()
    print(f"  temp    n={len(t):>5d}  min={t.min():.1f}  "
          f"max={t.max():.1f}  median={t.median():.1f}")
    s = full['sleep_hours'].dropna()
    print(f"  sleep   n={len(s):>5d}  min={s.min():.1f}  "
          f"max={s.max():.1f}  median={s.median():.1f}")
    w = full['weight_interp'].dropna()
    raw_w = full['weight_lbs'].dropna()
    print(f"  weight  n={len(w):>5d}  ({len(raw_w)} raw, "
          f"{len(w) - len(raw_w)} interpolated)  "
          f"min={w.min():.1f}  max={w.max():.1f}  median={w.median():.1f}")
    print(f"\ntotal traces: {len(tuple(fig.data))}")


_BUILD_JS = r"""
function buildTooltip(day) {
  var P = window.__TT_DATA;
  var idx = day - P.first_day;
  if (idx < 0 || idx >= P.n_days) return '';

  var DOW = ['Sun','Mon','Tue','Wed','Thu','Fri','Sat'];
  // Only the metrics actually plotted (set by the builder); iterating a fixed
  // list would dereference P.ma[<omitted>] and throw.
  var metricKeys   = P.metrics || ['volume', 'temp', 'sleep', 'weight'];
  var metricLabels = {volume: 'Avg. volume', temp: 'Avg. temp',
                      sleep: 'Avg. sleep', weight: 'Avg. weight'};
  var metricUnits  = {volume: 'mi', temp: '°C', sleep: 'hr', weight: 'lbs'};

  function fmt(v, n) {
    if (v === null || v === undefined || isNaN(v)) return null;
    return Number(v).toFixed(n);
  }

  var dt = new Date(day * 86400000);
  var y = dt.getUTCFullYear();
  var mo = String(dt.getUTCMonth() + 1).padStart(2, '0');
  var dd = String(dt.getUTCDate()).padStart(2, '0');
  var dateStr = y + '-' + mo + '-' + dd + ' (' + DOW[dt.getUTCDay()] + ')';

  var html = '<div class="tt-date">' + dateStr + '</div>';
  var anyShown = false;
  for (var i = 0; i < metricKeys.length; i++) {
    var m = metricKeys[i];
    var maStr = fmt(P.ma[m][idx], 1);
    var loStr = fmt(P.lo[m][idx], 1);
    var hiStr = fmt(P.hi[m][idx], 1);
    if (maStr === null && loStr === null && hiStr === null) continue;
    anyShown = true;
    var maOut = (maStr === null) ? '—' : maStr;
    var rangeOut = (loStr === null || hiStr === null)
                    ? '—' : ('(' + loStr + ' to ' + hiStr + ')');
    html += '<div class="tt-row">'
          +   '<span>' + metricLabels[m] + '</span>'
          +   '<span>'
          +     '<b>' + maOut + '</b> '
          +     '<span class="tt-mute">' + metricUnits[m]
          +     ' ' + rangeOut + '</span>'
          +   '</span>'
          + '</div>';
  }
  return anyShown ? html : '';
}
"""


if __name__ == '__main__':
    main()
