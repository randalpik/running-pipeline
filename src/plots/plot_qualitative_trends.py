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
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.shared.paths import DATA_DIR, OUTPUT_DIR


DEFAULT_DAILY = str(DATA_DIR / 'daily.csv')
DEFAULT_OUT = str(OUTPUT_DIR)
START_DATE = '2016-01-01'

MA_WINDOW = {'volume': 56, 'temp': 28, 'weight': 56}

RANGE_WINDOW = 14
RANGE_SMOOTH = 7
WEIGHT_INTERP_MAX_GAP = 7

N_ENVELOPE_STRIPS = 40        # vertical gradient resolution per subplot
ENVELOPE_X_STRIDE = 4         # x stride for strip traces (smoothed data, no visual loss)
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
    end = df['date'].max()
    cal = pd.DataFrame({'date': pd.date_range(start_date, end, freq='D')})
    keep = df[['date', 'miles', 'temp_c', 'weight_lbs']]
    full = cal.merge(keep, on='date', how='left').set_index('date')
    full['miles'] = full['miles'].fillna(0.0)
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
    return s.rolling(window, min_periods=max(1, window // 4),
                     center=True).mean()


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
    ap.add_argument('--start', default=START_DATE)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    full = load_series(args.daily, args.start)
    dates = full.index
    miles_max = float(full['miles'].max())

    metrics = [
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
        dict(key='weight', label='Weight', unit='lbs',
             series=full['weight_interp'],
             # 3-stop orange-family gradient. Mid-stop at 156 lbs places
             # dramatic color shift across the typical envelope (152-160).
             anchors=[(145.0, '#2D1006'), (156.0, '#8C4F1F'),
                      (170.0, '#E89535')]),
    ]

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        vertical_spacing=0.05,
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

        lo_smooth = smooth_series(lo_raw, RANGE_SMOOTH)
        hi_smooth = smooth_series(hi_raw, RANGE_SMOOTH)

        # Re-mask post-smoothing — the rolling mean's min_periods=1 can
        # spread valid neighbours back into the gap.
        if m['key'] == 'weight':
            lo_smooth = lo_smooth.where(valid, other=np.nan)
            hi_smooth = hi_smooth.where(valid, other=np.nan)

        # Envelope as gradient strip stack.
        add_envelope_strips(
            fig, dates, lo_smooth, hi_smooth, m['anchors'],
            N_ENVELOPE_STRIPS, ENVELOPE_X_STRIDE, row_i, ENVELOPE_ALPHA)

        # Trendline (pure white for contrast against gradient envelope).
        add_trendline(fig, dates, ma, row_i)

        y_min_env = float(lo_smooth.min())
        y_max_env = float(hi_smooth.max())
        pad = (y_max_env - y_min_env) * 0.04 if y_max_env > y_min_env else 1.0
        fig.update_yaxes(
            range=[y_min_env - pad, y_max_env + pad],
            row=row_i, col=1,
            title=dict(text=m['unit'],
                       font=dict(color='#aaa', size=11)),
            gridcolor='#2a2a2a', zerolinecolor='#2a2a2a',
        )

        def _round_arr(s, n=2):
            return [None if (v is None
                             or (isinstance(v, float) and np.isnan(v)))
                    else round(float(v), n)
                    for v in s.values]
        payload_lo[m['key']] = _round_arr(lo_raw)
        payload_hi[m['key']] = _round_arr(hi_raw)
        payload_ma[m['key']] = _round_arr(ma)

    # Yearly gridlines.
    year_ticks = pd.date_range(args.start, dates.max(), freq='YS')
    fig.update_xaxes(
        range=[args.start, str(dates.max().date())],
        gridcolor='#2a2a2a', zerolinecolor='#2a2a2a',
        tickmode='array',
        tickvals=[d.strftime('%Y-%m-%d') for d in year_ticks],
        ticktext=[d.strftime('%Y') for d in year_ticks],
        tickfont=dict(color='#aaa', size=11),
        showgrid=True,
    )

    fig.update_layout(
        title=dict(
            text=('<b>Miscellaneous Trends</b>'
                  '<br><sub style="font-size:13px;color:#bbb">'
                  'Mileage, temperature, and weight: moving average '
                  'trendlines with 14-day rolling min-max envelopes</sub>'),
            x=0.01, xanchor='left',
            y=0.985, yanchor='top',
            font=dict(color='#eee'),
        ),
        template='plotly_dark',
        paper_bgcolor='#1a1a1a',
        plot_bgcolor='#1a1a1a',
        font=dict(color='#eee', size=12),
        margin=dict(t=80, l=70, r=40, b=70),
        autosize=True,
        showlegend=False,
        hovermode=False,
    )
    for ann in fig['layout']['annotations']:
        ann['font'] = dict(color='#eee', size=13)

    payload = {
        'start_date': args.start,
        'n_days': len(dates),
        'lo': payload_lo,
        'hi': payload_hi,
        'ma': payload_ma,
        'ma_window': MA_WINDOW,
        'range_window': RANGE_WINDOW,
    }

    out_path = os.path.join(args.out_dir, 'qualitative_trends.html')
    write_dark_html(fig, out_path, payload)
    print(f'wrote {out_path}')

    print('\nseries summaries (post-shaping):')
    print(f"  volume  rest-days={int((full['miles']==0).sum())}  "
          f"min={full['miles'].min():.1f}  max={full['miles'].max():.1f}  "
          f"median(running)={full['miles'][full['miles']>0].median():.1f}")
    t = full['temp_c'].dropna()
    print(f"  temp    n={len(t):>5d}  min={t.min():.1f}  "
          f"max={t.max():.1f}  median={t.median():.1f}")
    w = full['weight_interp'].dropna()
    raw_w = full['weight_lbs'].dropna()
    print(f"  weight  n={len(w):>5d}  ({len(raw_w)} raw, "
          f"{len(w) - len(raw_w)} interpolated)  "
          f"min={w.min():.1f}  max={w.max():.1f}  median={w.median():.1f}")
    print(f"\ntotal traces: {len(fig.data)}")


def write_dark_html(fig, path, payload):
    fig.write_html(path, include_plotlyjs=True, full_html=True,
                   config={'responsive': True})
    css = (
        '<style>'
        'html,body{margin:0;padding:0;width:100%;height:100%;'
        'background:#1a1a1a;color:#eee;'
        'font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif;}'
        '.plotly-graph-div,.js-plotly-plot{width:100%!important;height:100vh!important;}'
        '</style>'
    )
    overlay = build_hover_overlay(payload)
    with open(path, 'r') as f:
        html = f.read()
    html = html.replace('<head>', '<head>' + css, 1)
    html = html.replace('</body>', overlay + '</body>')
    with open(path, 'w') as f:
        f.write(html)


def build_hover_overlay(payload):
    payload_json = json.dumps(payload)
    js = r"""
<style>
#qt-tooltip {
  position: fixed; top: 0; left: 0;
  background: rgba(26,26,26,0.96);
  color: #eee;
  border: 1px solid #555;
  padding: 10px 13px;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
  font-size: 12px; line-height: 1.5;
  border-radius: 4px;
  pointer-events: none;
  z-index: 9999;
  min-width: 240px;
  display: none;
  box-shadow: 0 4px 12px rgba(0,0,0,0.4);
}
#qt-tooltip .qt-day {
  font-weight: 600; font-size: 13px; color: #fff;
  margin-bottom: 5px;
  border-bottom: 1px solid #333; padding-bottom: 3px;
}
#qt-tooltip .qt-row {
  display: flex; justify-content: space-between;
  gap: 14px; align-items: baseline;
}
#qt-tooltip .qt-label { color: #aaa; }
#qt-tooltip .qt-window { color: #666; font-size: 11px; }
#qt-tooltip .qt-val { color: #eee; font-variant-numeric: tabular-nums; }
#qt-tooltip .qt-unit { color: #888; }
#qt-tooltip .qt-range { color: #888; font-size: 11px;
                        font-variant-numeric: tabular-nums; }
#qt-spike {
  position: fixed; top: 0; left: 0;
  width: 1px; height: 100vh;
  background: rgba(255,255,255,0.3);
  pointer-events: none;
  z-index: 9998;
  display: none;
}
</style>
<div id="qt-tooltip"></div>
<div id="qt-spike"></div>
<script>
(function() {
  var P = __PAYLOAD__;
  var metricKeys   = ['volume', 'temp', 'weight'];
  var metricLabels = {volume: 'Avg. volume', temp: 'Avg. temp',
                       weight: 'Avg. weight'};
  var metricUnits  = {volume: 'mi',     temp: '°C',   weight: 'lbs'};
  var DOW = ['Sun','Mon','Tue','Wed','Thu','Fri','Sat'];

  var startMs = Date.parse(P.start_date + 'T00:00:00Z');

  function fmt(v, decimals) {
    if (v === null || v === undefined || isNaN(v)) return null;
    return Number(v).toFixed(decimals);
  }

  function dayLabel(idx) {
    var d = new Date(startMs + idx * 86400000);
    var y = d.getUTCFullYear();
    var m = String(d.getUTCMonth() + 1).padStart(2, '0');
    var dd = String(d.getUTCDate()).padStart(2, '0');
    return y + '-' + m + '-' + dd + ' (' + DOW[d.getUTCDay()] + ')';
  }

  function buildHtml(idx) {
    var html = '<div class="qt-day">' + dayLabel(idx) + '</div>';
    var anyShown = false;
    for (var i = 0; i < metricKeys.length; i++) {
      var m = metricKeys[i];
      var w = P.ma_window[m];
      var maStr = fmt(P.ma[m][idx], 1);
      var loStr = fmt(P.lo[m][idx], 1);
      var hiStr = fmt(P.hi[m][idx], 1);
      if (maStr === null && loStr === null && hiStr === null) continue;
      anyShown = true;
      var maOut = (maStr === null) ? '—' : maStr;
      var rangeOut = (loStr === null || hiStr === null)
                     ? '—' : ('(' + loStr + ' to ' + hiStr + ')');
      html += '<div class="qt-row">'
            + '<span class="qt-label">' + metricLabels[m] + '</span>'
            + '<span>'
            +   '<span class="qt-val">' + maOut + '</span> '
            +   '<span class="qt-unit">' + metricUnits[m] + '</span> '
            +   '<span class="qt-range">' + rangeOut + '</span>'
            + '</span>'
            + '</div>';
    }
    return anyShown ? html : '';
  }

  var tt = document.getElementById('qt-tooltip');
  var spike = document.getElementById('qt-spike');
  var lastContent = '', ttW = 0, ttH = 0;
  var rafScheduled = false;
  var pendingX = 0, pendingY = 0, pendingContent = '', pendingSpikeX = 0, pendingShow = false;

  function update() {
    rafScheduled = false;
    if (!pendingShow) {
      tt.style.display = 'none';
      spike.style.display = 'none';
      return;
    }
    if (pendingContent !== lastContent) {
      tt.innerHTML = pendingContent;
      lastContent = pendingContent;
      ttW = tt.offsetWidth;
      ttH = tt.offsetHeight;
    }
    var x = pendingX + 15, y = pendingY + 10;
    if (x + ttW > window.innerWidth)  x = pendingX - ttW - 15;
    if (y + ttH > window.innerHeight) y = pendingY - ttH - 10;
    tt.style.transform = 'translate(' + Math.max(0, x) + 'px,' + Math.max(0, y) + 'px)';
    tt.style.display = 'block';
    spike.style.transform = 'translateX(' + pendingSpikeX + 'px)';
    spike.style.display = 'block';
  }

  function bind() {
    var pdiv = document.querySelector('.plotly-graph-div');
    if (!pdiv || !pdiv._fullLayout) { setTimeout(bind, 100); return; }

    pdiv.addEventListener('mousemove', function(e) {
      var fl = pdiv._fullLayout;
      if (!fl) return;
      var xa = fl.xaxis;
      var rect = pdiv.getBoundingClientRect();
      var bg = fl._size;
      var pl = rect.left + bg.l;
      var pr = rect.left + bg.l + bg.w;
      var pt = rect.top + bg.t;
      var pb = rect.top + bg.t + bg.h;
      if (e.clientX < pl || e.clientX > pr || e.clientY < pt || e.clientY > pb) {
        pendingShow = false;
        if (!rafScheduled) { rafScheduled = true; requestAnimationFrame(update); }
        return;
      }
      var dataMs = xa.p2c(e.clientX - rect.left - bg.l);
      var idx = Math.round((dataMs - startMs) / 86400000);
      if (idx < 0) idx = 0;
      if (idx >= P.n_days) idx = P.n_days - 1;
      var html = buildHtml(idx);
      if (!html) {
        pendingShow = false;
        if (!rafScheduled) { rafScheduled = true; requestAnimationFrame(update); }
        return;
      }
      pendingContent = html;
      pendingX = e.clientX;
      pendingY = e.clientY;
      pendingSpikeX = e.clientX;
      pendingShow = true;
      if (!rafScheduled) { rafScheduled = true; requestAnimationFrame(update); }
    });
    pdiv.addEventListener('mouseleave', function() {
      pendingShow = false;
      if (!rafScheduled) { rafScheduled = true; requestAnimationFrame(update); }
    });
  }
  bind();
})();
</script>
""".replace('__PAYLOAD__', payload_json)
    return js


if __name__ == '__main__':
    main()
