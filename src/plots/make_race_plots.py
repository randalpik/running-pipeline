"""Event-focused race plots: every race projected to 5K-equivalent pace.

Two outputs (interactive, dark-themed, self-contained HTML):
  - race_pace_all.html         : every race in races.csv on one panel,
                                 with a right-sidebar checkbox UI to
                                 toggle distance bins
  - race_pace_by_distance.html : 8 distance-bin subplots (400m → marathon)

Each plot lays the CS-implied 5K curve (posterior median, no ribbons) under
the race markers as a fitness reference. The race projection uses the
hyperbolic CS model with the long-distance β_long un-bias APPLIED but the
XC pre-correction OMITTED — so XC 5Ks display their actual race pace, not
their flat-course equivalent.

NO outliers are pruned. Every row in races.csv with positive distance and
time is plotted, including fatigued (race_seq > 1) races — these aren't
visually de-emphasized; the hover tag flags them as "Nth race of the day."
Surface controls color (Track/Road/XC/Downhill/Unknown).

Workflow
--------
  python make_race_plots.py --tag v11

Default paths assume the script lives next to bayes_cs_summary_{tag}.csv,
bayes_cs_params_{tag}.csv, races.csv. Override with --in-dir, --races, --out-dir.

Dependencies: pandas, numpy, scipy, plotly.
"""
import argparse
import datetime as dt
import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Allow running from any directory while still importing the sibling module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cs_projection import (load_cs_outputs, project_races_to_5k_pace,
                            cs_line_at_anchor, cubic_at_anchor)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_IN_DIR = SCRIPT_DIR
DEFAULT_RACES  = os.path.join(SCRIPT_DIR, 'races.csv')
DEFAULT_OUT    = SCRIPT_DIR

# Plot-specific short-distance calibration. The CS+D' projection
# severely underestimates fitness from sub-mile races (peak-speed and
# anaerobic limits dominate, not sustained CS), so without a correction a
# fast 400m projects to a 5K-equiv pace far slower than the same day's
# CS line. β_short>0 inflates the effective race time for d<d_thresh_short
# by (1 + β_short·log(d_thresh_short/d)) before projection.
#
# Calibrated April 2026 against Max's 800m behavior. His 800ms (β-immune,
# at d=d_thresh_short the correction is exactly 1) sit ~+7 to +11 s/mi
# below the CS line — that's the natural floor for sustained sub-CS
# efforts at sub-mile distances. β_short=0.35 places his fastest 400m
# (59.30s on 2016-03-31) at +1.6 s/mi below the CS line of that date, so
# all 12 lifetime 400ms cleanly sit at-or-below CS without dominating
# the cross-distance PR sequence. A couple of focused-track-era 400ms
# remain near the CS line, which is appropriate.
#
# Tune β_short up to push 400m markers toward/above CS, down to push them
# below. d_thresh_short=800 is the boundary: 800m+ races receive no
# correction.
BETA_SHORT     = 0.35
D_THRESH_SHORT = 800.0


# ---------- distance grouping for the 8-panel plot (8% tolerance, may leave races unmatched) ----------
GROUPS = [
    ('400',      400),
    ('800',      800),
    ('Mile',     1609.344),
    ('2 Mile',   3218.688),
    ('5K',       5000),
    ('10K',      10000),
    ('HM',       21097.5),
    ('Marathon', 42195),
]
TOLERANCE = 0.08  # 8% of target on either side


# Display titles for the by-distance plot. β-calibration mentions are
# intentionally omitted — in this view, race datapoints sit at their actual
# times (β only rescales the CS line, not the points), so the calibration
# detail isn't critical to interpreting any data point. The chart subtitle
# still labels the gold line as "CS-derived".
SUBPLOT_DISPLAY = {
    '400':      lambda n: f'400m (n={n})',
    '800':      lambda n: f'800m (n={n})',
    'Mile':     lambda n: f'1500m (including Mile, n={n})',
    '2 Mile':   lambda n: f'3000m (including 2 Mile, n={n})',
    '5K':       lambda n: f'5K (n={n})',
    '10K':      lambda n: f'10K (n={n})',
    'HM':       lambda n: f'Half marathon (n={n})',
    'Marathon': lambda n: f'Marathon (n={n})',
}


# Preferred y-axis tick interval (seconds) per panel — tuned for visual
# density given each distance's typical race time spread. Values not in the
# dict fall through to auto_time_ticks.
PANEL_TICK_SEC = {
    '400':       1,
    '800':       5,
    'Mile':     10,
    '2 Mile':   60,
    '5K':      120,
    '10K':      30,
    'HM':      300,
    'Marathon': 300,
}


# ---------- distance bins for the all-races plot's checkbox filter ----------
# 10 bins, snap-to-nearest by absolute meter distance — every race gets a bin
# regardless of how unusual the distance is. The farthest natural snap is
# 4828m (3-mile XC) → 5K at ~3.4%. Metric distances (1500m, 3000m) are
# separate bins from their imperial cousins (Mile, 2 Mile) since paces can
# meaningfully differ and Max wants individual visibility control.
FILTER_BINS = [
    ('400m',     400),
    ('800m',     800),
    ('1500m',    1500),
    ('Mile',     1609.344),
    ('3000m',    3000),
    ('2 Mile',   3218.688),
    ('5K',       5000),
    ('10K',      10000),
    ('HM',       21097.5),
    ('Marathon', 42195),
]


def classify_filter_bin(dist_m):
    """Snap a race distance to the closest FILTER_BINS entry. No tolerance cap."""
    if dist_m is None or pd.isna(dist_m):
        return None
    return min(FILTER_BINS, key=lambda nt: abs(nt[1] - float(dist_m)))[0]


def classify_group(dist_m):
    if dist_m is None or pd.isna(dist_m):
        return None
    for name, tgt in GROUPS:
        lo, hi = tgt * (1 - TOLERANCE), tgt * (1 + TOLERANCE)
        if lo <= dist_m <= hi:
            return name
    return None


def friendly_distance(d):
    if d is None or pd.isna(d):
        return '—'
    d = float(d)
    if abs(d - 1609.344) < 5 or d == 1609:
        return '1 mile'
    if abs(d - 3218.688) < 5 or d == 3218:
        return '2 miles'
    if 19410 <= d <= 22785:
        return 'Half Marathon'
    if 38819 <= d <= 45570:
        return 'Marathon'
    return f'{d:.0f} m'


def sec_to_mss(s):
    """Format seconds as M:SS (or H:MM:SS for >1h). Used for tick labels."""
    if s is None or pd.isna(s):
        return ''
    total = int(round(float(s)))
    if total >= 3600:
        h = total // 3600
        m = (total % 3600) // 60
        ss = total % 60
        return f'{h}:{m:02d}:{ss:02d}'
    m = total // 60
    ss = total % 60
    return f'{m}:{ss:02d}'


def sec_to_mss_full(s):
    """Format seconds preserving sub-second precision for race times."""
    if s is None or pd.isna(s):
        return ''
    s = float(s)
    tenths_total = int(round(s * 10))
    whole = tenths_total // 10
    frac_tenths = tenths_total % 10
    if whole >= 3600:
        h = whole // 3600
        m = (whole % 3600) // 60
        ss = whole % 60
        body = f'{h}:{m:02d}:{ss:02d}'
    else:
        m = whole // 60
        ss = whole % 60
        body = f'{m}:{ss:02d}'
    if frac_tenths > 0:
        return f'{body}.{frac_tenths}'
    return body


def auto_time_ticks(t_min, t_max, *, target_count=6):
    """Pick sensible tickvals/ticktext on a time axis given a range in
    seconds. Aim for ~target_count ticks; pick from a fixed interval ladder
    (1, 2, 5, 10, 30, 60, 300, 600, 1800, 3600 sec). Labels via sec_to_mss
    so short ranges show M:SS and HM/marathon ranges show H:MM:SS.
    """
    import math
    if not np.isfinite(t_min) or not np.isfinite(t_max) or t_max <= t_min:
        return [int(round(t_min))], [sec_to_mss(t_min)]
    span = t_max - t_min
    intervals = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600]
    interval = intervals[-1]
    for iv in intervals:
        if span / iv <= target_count:
            interval = iv
            break
    return time_ticks_at_interval(t_min, t_max, interval)


def time_ticks_at_interval(t_min, t_max, interval):
    """Tickvals/ticktext at a fixed seconds interval, snapped to multiples
    of `interval`. Labels via sec_to_mss."""
    import math
    if not np.isfinite(t_min) or not np.isfinite(t_max) or t_max <= t_min:
        return [int(round(t_min))], [sec_to_mss(t_min)]
    tick_min = math.floor(t_min / interval) * interval
    tick_max = math.ceil(t_max / interval) * interval
    n = int(round((tick_max - tick_min) / interval)) + 1
    ticks = [tick_min + i * interval for i in range(n)]
    return ticks, [sec_to_mss(t) for t in ticks]


def thin_yearly_ticks(x_lo, x_hi, *, max_labels=6):
    """Yearly gridlines (Jan 1 each year), but only label every Nth year if
    there'd be more than max_labels. Empty-string labels still draw the
    tick + gridline, just without text — keeping visual rhythm consistent
    across panels with different x-spans."""
    y_lo = int(pd.Timestamp(x_lo).year)
    y_hi = int(pd.Timestamp(x_hi).year)
    years = list(range(y_lo, y_hi + 1))
    if not years:
        return [], []
    tickvals = [pd.Timestamp(f'{y}-01-01') for y in years]
    n = len(years)
    if n <= max_labels:
        ticktext = [str(y) for y in years]
    else:
        step = -(-n // max_labels)  # ceil division
        # Anchor labels to the LAST year (most recent), then walk back by `step`.
        ticktext = [str(y) if ((n - 1 - i) % step == 0) else ''
                    for i, y in enumerate(years)]
    return tickvals, ticktext


# ---------- visual encoding ----------
SURFACE_COLORS = {
    'Track':    '#ff5b4d',
    'Road':     '#4aa3ff',
    'XC':       '#4ade80',
    'Downhill': '#b87de9',
    'Unknown':  '#888888',
}
SURFACE_LEGEND_ORDER = ['Road', 'Track', 'XC', 'Downhill', 'Unknown']

# CS line styling (matches plot_training_quality.py and bayes_cs_plot.py)
CS_LINE_COLOR = 'rgb(255,180,80)'
CS_LINE_WIDTH = 2.5

# PR-overlay styling. PR points get a thin white diamond outline drawn
# ON TOP of the colored race marker (transparent fill so the underlying
# surface color shows through). Size is set to base_size + 1 so the ring
# sits flush against the marker — no halo gap.
PR_LINE_WIDTH    = 1.5      # px — thin outline, not a halo
PR_RING_PADDING  = 1        # ring size = base_size + this
PR_LINE_COLOR    = 'white'
PR_LEGEND_NAME   = 'PR effort'
PR_LEGEND_RANK   = 2100  # below CS-derived (2000) and Estimated (2001)


def pr_marker(base_size):
    """Plotly marker dict for the PR overlay (transparent fill, white ring).

    Plotly draws marker.line centered on the size boundary, so setting
    size = base_size + 1 with line_width=1.5 puts the ring's inner edge
    at radius (base_size+1)/2 - 0.75, which slightly overlaps the base
    marker's outer edge — eliminating any visible gap.
    """
    return dict(
        symbol='diamond',
        size=base_size + PR_RING_PADDING,
        color='rgba(0,0,0,0)',
        line=dict(width=PR_LINE_WIDTH, color=PR_LINE_COLOR),
    )


# Surfaces excluded from PR eligibility. Downhill races (net-downhill,
# course-aided) project to absurdly fast 5K-equivalents that, while still
# interesting to display, would dominate the running-min sequence and
# invalidate every subsequent on-the-flat PR. They're plotted normally;
# they just don't compete for the white PR ring.
PR_EXCLUDED_SURFACES = {'Downhill'}


def is_pr_eligible(surface):
    return str(surface) not in PR_EXCLUDED_SURFACES


def compute_pr_mask(df, *, value_col, date_col='date'):
    """Return a boolean array (aligned to df.index order) marking running-min
    PR points: chronologically-earliest entry, then each subsequent entry
    that beats the prior best on `value_col`. NaN values never count.

    Lower = better on `value_col` (so pace_norm_min and time_norm_sec both
    work). Sort is stable on date; ties go to the earlier index.
    """
    if len(df) == 0:
        return np.zeros(0, dtype=bool)
    order = df[date_col].argsort(kind='stable').values
    vals = df[value_col].values[order]
    is_pr_sorted = np.zeros(len(vals), dtype=bool)
    best = np.inf
    for i, v in enumerate(vals):
        if pd.notna(v) and v < best:
            best = v
            is_pr_sorted[i] = True
    is_pr = np.zeros(len(vals), dtype=bool)
    is_pr[order] = is_pr_sorted
    return is_pr


def ordinal(n):
    """Return n with English ordinal suffix: 1→'1st', 2→'2nd', 3→'3rd', 11→'11th'."""
    n = int(n)
    if 10 <= n % 100 <= 20:
        suffix = 'th'
    else:
        suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
    return f"{n}{suffix}"


def build_hover(row):
    parts = [f"<b>{row['date'].date()}</b>",
             f"Distance: {friendly_distance(row['distance_m'])}",
             f"Time: {sec_to_mss_full(row['time_sec_original'])}"]
    # Show the projected 5K-equivalent unless the race already IS a 5K
    is_5k = abs(float(row['distance_m']) - 5000.0) < 1.0
    if not is_5k:
        eq_sec = float(row['time_norm_sec'])
        eq_pace = float(row['pace_norm_sec'])
        parts.append(f"5K-equiv: {sec_to_mss(eq_sec)} ({sec_to_mss(eq_pace)}/mi)")
    parts.append(f"Surface: {row.get('surface') or '—'}")
    if row.get('location') and str(row['location']) != 'nan':
        parts.append(f"Location: {row['location']}")
    if row.get('fatigued'):
        parts.append(f"<i>Fatigued ({ordinal(row.get('race_seq', 0))} race of the day)</i>")
    if row.get('temp_c') is not None and not pd.isna(row.get('temp_c')):
        parts.append(f"Temp: {row['temp_c']:.0f}°C")
    if row.get('event') and str(row['event']) != 'nan':
        parts.append(f"Event: {row['event']}")
    if row.get('note') and str(row['note']) != 'nan':
        parts.append(f"Note: {row['note']}")
    return '<br>'.join(parts)


def build_hover_anchored(row, anchor_m):
    """Hover string for the by-distance plot. Includes projection to the
    panel's anchor distance and Δ vs CS expectation on that date."""
    parts = [f"<b>{row['date'].date()}</b>",
             f"Distance: {friendly_distance(row['distance_m'])}",
             f"Time: {sec_to_mss_full(row['time_sec_original'])}"]
    is_at_anchor = abs(float(row['distance_m']) - float(anchor_m)) < 1.0
    if not is_at_anchor:
        proj_sec = float(row['time_norm_sec'])
        parts.append(f"Projected to {friendly_distance(anchor_m)}: "
                     f"{sec_to_mss_full(proj_sec)}")
    cs_sec = row.get('cs_pred_sec', np.nan)
    if cs_sec is not None and not pd.isna(cs_sec):
        cs_sec = float(cs_sec)
        delta = float(row['time_norm_sec']) - cs_sec
        side = 'under (faster)' if delta < 0 else 'over (slower)'
        parts.append(f"CS expected: {sec_to_mss_full(cs_sec)}")
        parts.append(f"Δ vs CS: {delta:+.1f}s {side}")
    parts.append(f"Surface: {row.get('surface') or '—'}")
    if row.get('location') and str(row['location']) != 'nan':
        parts.append(f"Location: {row['location']}")
    if row.get('fatigued'):
        parts.append(f"<i>Fatigued ({ordinal(row.get('race_seq', 0))} race of the day)</i>")
    if row.get('temp_c') is not None and not pd.isna(row.get('temp_c')):
        parts.append(f"Temp: {row['temp_c']:.0f}°C")
    if row.get('event') and str(row['event']) != 'nan':
        parts.append(f"Event: {row['event']}")
    if row.get('note') and str(row['note']) != 'nan':
        parts.append(f"Note: {row['note']}")
    return '<br>'.join(parts)


# ---------- output (dark theme, full viewport, matching CS plot) ----------
def write_dark_html(fig, path, *, extra_html=''):
    fig.write_html(path, include_plotlyjs=True, full_html=True,
                   config={'responsive': True})
    css = (
        '<style>'
        'html,body{margin:0;padding:0;width:100%;height:100%;'
        'background:#1a1a1a;color:#eee;'
        'font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif;}'
        '.plotly-graph-div,.js-plotly-plot{width:100%!important;height:100vh!important;}'
        '</style>')
    with open(path, 'r') as f:
        html = f.read()
    html = html.replace('<head>', '<head>' + css, 1)
    if extra_html:
        html = html.replace('</body>', extra_html + '</body>')
    with open(path, 'w') as f:
        f.write(html)


def build_distance_filter_ui(bin_names):
    """HTML/CSS/JS for the right-sidebar distance-filter checkboxes.

    Each checkbox toggles the visibility of all traces tagged with
    meta.filter_bin matching the box's data-bin attribute. Sentinel traces
    (without meta.filter_bin) are left alone, which is what keeps each
    surface's legend entry alive even when all its bins are unchecked.
    """
    cb_html = '\n'.join(
        f'  <label class="bf-row"><input type="checkbox" data-bin="{b}" checked> {b}</label>'
        for b in bin_names)
    return f"""
<style>
#bin-filter {{
  position: fixed; right: 12px; bottom: 20px;
  background: rgba(26,26,26,0.92);
  border: 1px solid #444;
  padding: 12px 14px;
  border-radius: 4px;
  color: #eee;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
  font-size: 13px;
  z-index: 100;
  min-width: 110px;
  user-select: none;
}}
#bin-filter .bf-title {{ font-weight: 500; margin-bottom: 8px; color: #eee; }}
#bin-filter .bf-buttons {{ margin-bottom: 8px; display: flex; gap: 4px; }}
#bin-filter .bf-buttons button {{
  background: transparent; border: 1px solid #555; color: #eee;
  padding: 2px 8px; cursor: pointer; font-size: 12px; border-radius: 3px;
  flex: 1;
}}
#bin-filter .bf-buttons button:hover {{ background: #2a2a2a; }}
#bin-filter .bf-row {{ display: block; margin: 3px 0; cursor: pointer; line-height: 1.6; }}
#bin-filter input[type=checkbox] {{ margin-right: 6px; vertical-align: middle; cursor: pointer; accent-color: #4aa3ff; }}
</style>
<div id="bin-filter">
  <div class="bf-title">Distance</div>
  <div class="bf-buttons">
    <button id="bf-all">All</button>
    <button id="bf-none">None</button>
  </div>
{cb_html}
</div>
<script>
(function() {{
  function findPlot() {{
    return document.querySelector('.plotly-graph-div');
  }}
  // Plotly may store numeric arrays in a binary-packed typedarray-spec
  // format ({{dtype, bdata, _inputArray}}) for compactness — t.y[i] then
  // returns undefined and t.y.length is undefined. The original numbers
  // are on _inputArray, which is typically a Float64Array (a TypedArray,
  // NOT a plain Array — Array.isArray() returns false on it). Both
  // TypedArrays and plain Arrays support [i] indexing and .length, so
  // checking for length is enough.
  function asArray(v) {{
    if (v == null) return null;
    if (Array.isArray(v)) return v;
    if (v._inputArray && typeof v._inputArray.length === 'number') return v._inputArray;
    if (typeof v.length === 'number') return v;
    return null;
  }}
  // Walk every trace tagged meta.filter_bin and gather (date, y) tuples
  // from those currently visible AND PR-eligible (Downhill races are
  // excluded). Respects bin checkboxes AND surface legend toggles, both
  // of which mutate trace.visible.
  function gatherVisibleRaces(plot) {{
    var pts = [];
    plot.data.forEach(function(t) {{
      if (!t.meta || !t.meta.filter_bin) return;
      if (t.meta.pr_eligible === false) return;
      var v = t.visible;
      if (v === false || v === 'legendonly') return;
      var xs = asArray(t.x);
      var ys = asArray(t.y);
      if (!xs || !ys) return;
      for (var i = 0; i < xs.length; i++) {{
        var y = ys[i];
        if (y == null || isNaN(y)) continue;
        pts.push({{x: xs[i], y: y, ts: new Date(xs[i]).getTime()}});
      }}
    }});
    return pts;
  }}
  function computePRs(pts) {{
    pts.sort(function(a, b) {{ return a.ts - b.ts; }});
    var best = Infinity, prX = [], prY = [];
    for (var i = 0; i < pts.length; i++) {{
      if (pts[i].y < best) {{
        best = pts[i].y;
        prX.push(pts[i].x);
        prY.push(pts[i].y);
      }}
    }}
    return {{x: prX, y: prY}};
  }}
  function findOverlayIdx(plot) {{
    for (var i = 0; i < plot.data.length; i++) {{
      var t = plot.data[i];
      if (t.meta && t.meta.is_pr_overlay) return i;
    }}
    return -1;
  }}
  function recomputePRs() {{
    var plot = findPlot();
    if (!plot || !plot.data || !window.Plotly) return;
    var idx = findOverlayIdx(plot);
    if (idx < 0) return;
    var pts = gatherVisibleRaces(plot);
    var pr  = computePRs(pts);
    Plotly.restyle(plot, {{x: [pr.x], y: [pr.y]}}, [idx]);
  }}
  function update() {{
    var plot = findPlot();
    if (!plot || !plot.data || !window.Plotly) {{ setTimeout(update, 100); return; }}
    var checked = new Set();
    document.querySelectorAll('#bin-filter input[type=checkbox]').forEach(function(cb) {{
      if (cb.checked) checked.add(cb.dataset.bin);
    }});
    var updates = plot.data.map(function(t) {{
      var bin = t.meta && t.meta.filter_bin;
      if (!bin) return true;          // CS lines, sentinels, PR overlay — leave alone
      return checked.has(bin);
    }});
    // Restyle visibility — the plotly_restyle event listener will trigger
    // recomputePRs once the change is applied.
    Plotly.restyle(plot, {{'visible': updates}});
  }}
  function attachLegendInterceptor() {{
    var plot = findPlot();
    if (!plot || !plot.data || !window.Plotly) {{ setTimeout(attachLegendInterceptor, 100); return; }}
    // Cancel clicks on the PR-legend sentinel (returning false stops Plotly
    // from toggling visibility AND prevents the legend marker from dimming).
    plot.on('plotly_legendclick', function(ev) {{
      var t = plot.data[ev.curveNumber];
      if (t && t.meta && t.meta.is_pr_legend_sentinel) return false;
    }});
    plot.on('plotly_legenddoubleclick', function(ev) {{
      var t = plot.data[ev.curveNumber];
      if (t && t.meta && t.meta.is_pr_legend_sentinel) return false;
    }});
    // plotly_restyle fires AFTER the visibility change is applied — both
    // for our own checkbox-driven restyles AND for Plotly's internal
    // legend-click toggles. To avoid recursion when recomputePRs itself
    // calls Plotly.restyle on the overlay, we identify our own restyles
    // by checking the event payload: a restyle scoped to ONLY the PR
    // overlay's trace index is ours, so skip. Anything else is a real
    // visibility change and we recompute.
    plot.on('plotly_restyle', function(eventData) {{
      var indices = (eventData && eventData[1]) || null;
      var prIdx = findOverlayIdx(plot);
      if (indices && indices.length === 1 && indices[0] === prIdx) return;
      // Defer slightly so any in-flight Plotly state updates settle.
      setTimeout(recomputePRs, 0);
    }});
  }}
  document.querySelectorAll('#bin-filter input[type=checkbox]').forEach(function(cb) {{
    cb.addEventListener('change', update);
  }});
  document.getElementById('bf-all').addEventListener('click', function() {{
    document.querySelectorAll('#bin-filter input[type=checkbox]').forEach(function(cb) {{ cb.checked = true; }});
    update();
  }});
  document.getElementById('bf-none').addEventListener('click', function() {{
    document.querySelectorAll('#bin-filter input[type=checkbox]').forEach(function(cb) {{ cb.checked = false; }});
    update();
  }});
  attachLegendInterceptor();
}})();
</script>
"""


# ---------- shared y-axis & x-axis layout ----------
# y-range chosen to accommodate Max's full historical range, from his 2024
# downhill mile (~4:40/mi 5K-equiv) to his 2008 Nike 5K For Kids (~9:54).
# Reversed so faster paces appear higher.
Y_MIN, Y_MAX = 4.30, 10.00
YTICK_VALS = [4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0]
YTICK_TXT  = [sec_to_mss(v * 60) for v in YTICK_VALS]


def add_cs_line(fig, daily_summary, *, row=None, col=None, show_legend=True,
                legend_name='CS-derived 5K pace'):
    """Drop the posterior-median CS line (no ribbons) onto fig."""
    trace = go.Scatter(
        x=daily_summary['date'], y=daily_summary['p5k_implied_min'],
        mode='lines', name=legend_name,
        line=dict(color=CS_LINE_COLOR, width=CS_LINE_WIDTH),
        hoverinfo='skip',
        showlegend=show_legend,
        legendgroup='cs',
        legendrank=2000)  # push below default-rank (1000) surface entries
    if row is not None and col is not None:
        fig.add_trace(trace, row=row, col=col)
    else:
        fig.add_trace(trace)


def fit_handdrawn_cubic(daily_summary, *,
                         handdrawn_start=pd.Timestamp('2008-04-01'),
                         handdrawn_end=pd.Timestamp('2013-05-26'),
                         anchors=None):
    """Fit the calibrated pre-2013 cubic in 5K-equiv pace space.

    Returns (coefs, t0, hd_start, hd_end). The cubic interpolates 4 anchors:
      - (2008-04-01, 9:00) beginner fitness
      - (2009-07-01, 8:20) peak before stopping running
      - (2012-01-01, 8:40) slight regression after years off
      - (2013-05-26, model value at that date) — meets the real curve

    The end date was chosen so the cubic's natural end-slope matches the
    real curve's slope (~-0.33 s/mile/day) — tangent join, no kink. Used by
    both the all-races plot (rendered directly) and the by-distance plot
    (re-projected to each panel's anchor distance).
    """
    p5k_med = daily_summary['p5k_implied_min'].values
    daily_dates_pd = pd.to_datetime(daily_summary['date']).values
    if anchors is None:
        anchors = [
            (handdrawn_start,             9.000),
            (pd.Timestamp('2009-07-01'),  8.333),
            (pd.Timestamp('2012-01-01'),  8.667),
            (handdrawn_end,               None),
        ]
    resolved = []
    for d, p in anchors:
        if p is None:
            idx = int(np.argmin(np.abs(daily_dates_pd - np.datetime64(d))))
            p = float(p5k_med[idx])
        resolved.append((pd.Timestamp(d), float(p)))
    t0 = resolved[0][0]
    ts = np.array([(d - t0).days for d, _ in resolved], dtype=float)
    ys = np.array([p for _, p in resolved], dtype=float)
    coefs = np.polyfit(ts, ys, len(resolved) - 1)
    return coefs, t0, handdrawn_start, handdrawn_end


def add_cs_line_blended(fig, daily_summary, race_dates, *,
                         handdrawn_start=pd.Timestamp('2008-04-01'),
                         handdrawn_end=pd.Timestamp('2013-05-26'),
                         anchors=None):
    """All-races-plot CS line: dotted hand-drawn cubic from handdrawn_start
    to handdrawn_end, then the real model line (solid) from there onward.

    Why hand-drawn through pre-2013: the GP isn't really estimating CS in
    that period (sparse boundary races, multi-year training gaps), so the
    GP's smooth fit misleadingly looks like steady fitness gain. The cubic
    captures the actual training arc.

    The race_dates argument is unused (kept for API symmetry).
    """
    coefs, t0, hd_start, hd_end = fit_handdrawn_cubic(
        daily_summary, handdrawn_start=handdrawn_start,
        handdrawn_end=handdrawn_end, anchors=anchors)

    p5k_med = daily_summary['p5k_implied_min'].values
    daily_dates_pd = pd.to_datetime(daily_summary['date']).values

    dotted_dates = pd.date_range(hd_start, hd_end, freq='D')
    dotted_ts = (dotted_dates - t0).days.values.astype(float)
    dotted_ys = np.polyval(coefs, dotted_ts)

    mask_solid = daily_dates_pd >= np.datetime64(hd_end)
    y_solid = np.where(mask_solid, p5k_med, np.nan)

    fig.add_trace(go.Scatter(
        x=daily_summary['date'], y=y_solid,
        mode='lines', name='CS-derived 5K pace',
        line=dict(color=CS_LINE_COLOR, width=CS_LINE_WIDTH),
        hoverinfo='skip', showlegend=True, legendgroup='cs',
        legendrank=2000))
    fig.add_trace(go.Scatter(
        x=dotted_dates, y=dotted_ys,
        mode='lines', name='Estimated 5K pace',
        line=dict(color=CS_LINE_COLOR, width=CS_LINE_WIDTH, dash='dot'),
        hoverinfo='skip', showlegend=True, legendgroup='cs',
        legendrank=2001))


def add_race_traces_filterable(fig, df, *, marker_size=9):
    """All-races-plot variant: emits one trace per (surface, bin) so the JS
    checkbox UI can toggle visibility per bin. Also emits one sentinel trace
    per surface carrying the legend entry — so the legend stays stable when
    the user unchecks bins.

    The sentinel parks a single marker at year 1900 (outside the explicit
    x-axis range, so it never renders on plot). Empty-data sentinels turn
    out to suppress the legend entry in some plotly versions; an off-range
    point keeps the legend marker reliably rendered.

    Each bin trace tags itself with meta.filter_bin so the JS can find it.
    Sentinels and other traces (CS lines) leave meta.filter_bin unset and
    are ignored by the filter.

    Fatigued races are NOT visually distinguished — the plot's purpose is
    to show absolute race-time performance regardless of context. Fatigue
    info is preserved in the hover text.
    """
    bin_names = [b[0] for b in FILTER_BINS]
    sentinel_x = [pd.Timestamp('1900-01-01')]
    sentinel_y = [6.0]  # arbitrary; clipped by axis range
    for surf in SURFACE_LEGEND_ORDER:
        sub = df[df['surface_plot'] == surf]
        if len(sub) == 0:
            continue
        color = SURFACE_COLORS.get(surf, '#888888')
        common_marker = dict(
            color=color, size=marker_size, symbol='diamond',
            opacity=0.85,
            line=dict(width=0.5, color='white'))
        # Off-range sentinel: owns the legend entry, never renders on plot.
        fig.add_trace(go.Scatter(
            x=sentinel_x, y=sentinel_y, mode='markers', name=surf,
            marker=common_marker,
            legendgroup=surf, showlegend=True, hoverinfo='skip'))
        # Per-bin data traces (no legend; tagged for the filter JS).
        for bin_name in bin_names:
            s2 = sub[sub['filter_bin'] == bin_name]
            if len(s2) == 0:
                continue
            fig.add_trace(go.Scatter(
                x=s2['date'], y=s2['pace_norm_min'],
                mode='markers', name=surf,
                marker=common_marker,
                hovertemplate='%{customdata}<extra></extra>',
                customdata=s2['hover'],
                legendgroup=surf, showlegend=False,
                meta={'filter_bin': bin_name,
                      'pr_eligible': is_pr_eligible(surf)}))


def add_pr_overlay_filterable(fig, df, *, value_col='pace_norm_min',
                                date_col='date'):
    """All-races-plot PR overlay: one trace carries the actual PR markers
    (showlegend=False, dynamically rewritten by the filter JS), plus an
    off-range sentinel that owns the 'PR effort' legend entry.

    Initial PR set is computed against the full pool (every checkbox starts
    checked). The JS recomputes on every visibility change.
    """
    # PR pool excludes surfaces in PR_EXCLUDED_SURFACES (currently Downhill).
    # The race markers stay visible on the plot — they just don't compete.
    eligible = df[df['surface_plot'].apply(is_pr_eligible)] if 'surface_plot' in df.columns \
               else df[df['surface'].apply(is_pr_eligible)]
    is_pr = compute_pr_mask(eligible, value_col=value_col, date_col=date_col)
    pr_df = eligible[is_pr].sort_values(date_col)
    # Actual PR markers — meta.is_pr_overlay flags this trace for the JS
    # so it can find/restyle it. hoverinfo='skip' delegates hover to the
    # underlying race marker beneath (same x/y, drawn earlier).
    fig.add_trace(go.Scatter(
        x=pr_df[date_col], y=pr_df[value_col],
        mode='markers', name=PR_LEGEND_NAME,
        marker=pr_marker(base_size=9),
        hoverinfo='skip',
        showlegend=False,
        legendgroup='pr',
        meta={'is_pr_overlay': True}))
    # Off-range sentinel owning the legend entry. Non-clickable: the
    # JS legendclick handler returns false on traces with
    # meta.is_pr_legend_sentinel.
    fig.add_trace(go.Scatter(
        x=[pd.Timestamp('1900-01-01')], y=[6.0],
        mode='markers', name=PR_LEGEND_NAME,
        marker=pr_marker(base_size=9),
        showlegend=True, hoverinfo='skip',
        legendgroup='pr', legendrank=PR_LEGEND_RANK,
        meta={'is_pr_legend_sentinel': True}))


def build_pr_nonclick_js():
    """Standalone JS that makes the PR-effort legend entry non-clickable.

    Used by the by-distance plot, which doesn't have the filter UI's JS.
    Returning false from plotly_legendclick cancels the toggle AND prevents
    the legend marker from dimming.
    """
    return """
<script>
(function() {
  function attach() {
    var plot = document.querySelector('.plotly-graph-div');
    if (!plot || !plot.data || !window.Plotly) { setTimeout(attach, 100); return; }
    plot.on('plotly_legendclick', function(ev) {
      var t = plot.data[ev.curveNumber];
      if (t && t.meta && t.meta.is_pr_legend_sentinel) return false;
    });
    plot.on('plotly_legenddoubleclick', function(ev) {
      var t = plot.data[ev.curveNumber];
      if (t && t.meta && t.meta.is_pr_legend_sentinel) return false;
    });
  }
  attach();
})();
</script>
"""


def yearly_x_axis(**kwargs):
    """Yearly gridline + label config for the x-axis (matches CS plot)."""
    base = dict(showgrid=True, gridcolor='#333',
                dtick='M12', tickformat='%Y')
    base.update(kwargs)
    return base


def reversed_pace_y_axis(**kwargs):
    base = dict(title='5K-equivalent pace (min/mi)',
                range=[Y_MAX, Y_MIN],
                tickmode='array', tickvals=YTICK_VALS, ticktext=YTICK_TXT,
                showgrid=True, gridcolor='#333')
    base.update(kwargs)
    return base


# ---------- main ----------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--in-dir', default=DEFAULT_IN_DIR,
                   help='Directory containing bayes_cs_summary_{tag}.csv and '
                        'bayes_cs_params_{tag}.csv (default: script dir).')
    p.add_argument('--tag', default='v11',
                   help='Fit tag suffix (default: v11).')
    p.add_argument('--races', default=DEFAULT_RACES,
                   help='Path to races.csv (default: script-dir/races.csv).')
    p.add_argument('--out-dir', default=DEFAULT_OUT,
                   help='Output directory (default: script dir).')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # ---------- load ----------
    daily_summary, beta_long, d_thresh, xc_correction = load_cs_outputs(
        args.in_dir, args.tag)
    print(f'Loaded CS outputs: {len(daily_summary)} daily points '
          f'({daily_summary["date"].iloc[0].date()} → '
          f'{daily_summary["date"].iloc[-1].date()})')
    print(f'  β_long={beta_long:.4f}, d_thresh={d_thresh:.0f}m, '
          f'xc_correction={xc_correction:.4f} (XC correction NOT applied here)')

    if not os.path.exists(args.races):
        sys.exit(f'ERROR: races file not found: {args.races}')
    races = pd.read_csv(args.races, parse_dates=['date'])
    print(f'Loaded {len(races)} races')

    if 'fatigued' not in races.columns:
        races['fatigued'] = False
    else:
        races['fatigued'] = races['fatigued'].fillna(False).astype(bool)
    if 'surface' not in races.columns:
        races['surface'] = 'Unknown'

    # NO filtering. Per the design intent: every row with positive distance
    # and time gets projected. Even fatigued / Downhill / cs-excluded rows.
    elig = races[(races['distance_m'].fillna(0) > 0)
                 & (races['time_sec'].fillna(0) > 0)].copy().sort_values('date')
    print(f'Plotting {len(elig)} races (no exclusions)')

    elig = project_races_to_5k_pace(
        elig, daily_summary, beta_long, d_thresh,
        apply_xc_correction=False,  # ← key difference vs CS plot
        beta_short=BETA_SHORT, d_thresh_short=D_THRESH_SHORT)

    n_proj = elig['pace_norm_min'].notna().sum()
    print(f'Projection succeeded for {n_proj}/{len(elig)} races')
    elig = elig[elig['pace_norm_min'].notna()].copy()

    elig['surface_plot'] = elig['surface'].fillna('Unknown')
    elig['group']        = elig['distance_m'].apply(classify_group)
    elig['filter_bin']   = elig['distance_m'].apply(classify_filter_bin)
    elig['hover']        = elig.apply(build_hover, axis=1)

    # X-axis range: 2008-04-01 (just before earliest race 2008-04-26) → today
    x_lo = pd.Timestamp('2008-04-01')
    x_hi = pd.Timestamp(dt.date.today())

    # ---------- plot 1: all races, single panel, with distance-filter checkboxes ----------
    fig1 = go.Figure()
    add_cs_line_blended(fig1, daily_summary, elig['date'])
    add_race_traces_filterable(fig1, elig, marker_size=9)
    add_pr_overlay_filterable(fig1, elig, value_col='pace_norm_min')
    fig1.update_layout(
        title=dict(text='Lifetime races: 5K-equivalent pace<br>'
                        '<sub style="font-size:13px;color:#bbb">'
                        'Hyperbolic CS projection with corrections applied for short and long distances'
                        '</sub>',
                   y=0.97, yanchor='top'),
        template='plotly_dark',
        paper_bgcolor='#1a1a1a', plot_bgcolor='#1a1a1a',
        font=dict(color='#eee'),
        hovermode='closest',
        autosize=True,
        margin=dict(t=130, l=70, r=200, b=60),
        legend=dict(yanchor='top', y=0.99, xanchor='left', x=1.02),
        xaxis=yearly_x_axis(title='Date', range=[x_lo, x_hi]),
        yaxis=reversed_pace_y_axis())

    out1 = os.path.join(args.out_dir, 'race_pace_all.html')
    filter_ui = build_distance_filter_ui([b[0] for b in FILTER_BINS])
    write_dark_html(fig1, out1, extra_html=filter_ui)
    print(f'Wrote {out1}')

    # ---------- plot 2: 8 distance-bin subplots ----------
    # Each panel is rendered in its own time-at-anchor coordinates:
    # the anchor is the MODE distance for that bin (so e.g. the Mile panel's
    # anchor is whichever of 1500/1600/1609 you raced most often). The CS
    # line is the model's predicted time at that anchor on each date —
    # rescaled but same shape as the 5K-equiv curve. The hand-drawn cubic
    # is converted to time-at-anchor too (only visible on the 5K panel
    # because no other bin has races during 2008-2013). Each marker's
    # hover includes Δ vs CS — how much that race fell above/below what
    # the model thought you were capable of at that distance on that date.
    group_names = [g[0] for g in GROUPS]
    nominal_lookup = dict(GROUPS)

    # Per-bin anchor = mode distance among races in that bin (fall back to
    # the nominal target if the bin is empty).
    bin_anchors = {}
    bin_counts = {}
    for name in group_names:
        sub = elig[elig['group'] == name]
        bin_counts[name] = len(sub)
        if len(sub):
            bin_anchors[name] = float(sub['distance_m'].mode().iloc[0])
        else:
            bin_anchors[name] = float(nominal_lookup[name])

    # Subplot title: clean per-bin display name + count. β-calibration not
    # mentioned — see SUBPLOT_DISPLAY docstring.
    def subplot_title(name):
        n = bin_counts[name]
        fn = SUBPLOT_DISPLAY.get(name, lambda n: f"{name} (n={n})")
        return fn(n)

    fig2 = make_subplots(rows=2, cols=4,
                         subplot_titles=[subplot_title(n) for n in group_names],
                         shared_yaxes=False, shared_xaxes=False,
                         horizontal_spacing=0.07,
                         vertical_spacing=0.16)

    # Cubic, computed once and reused for every panel
    cubic_coefs, cubic_t0, hd_start, hd_end = fit_handdrawn_cubic(daily_summary)

    surfaces_seen = set()
    cs_legend_drawn = False
    pr_legend_drawn = False

    for i, name in enumerate(group_names):
        r = i // 4 + 1
        c = i % 4 + 1
        sub = elig[elig['group'] == name]
        if len(sub) == 0:
            continue
        anchor = bin_anchors[name]

        # 1. Project races to this anchor (β_short and β_long applied
        #    symmetrically). For races already at the anchor, this is identity.
        sub_proj = project_races_to_5k_pace(
            sub, daily_summary, beta_long, d_thresh,
            apply_xc_correction=False, norm_dist_m=anchor,
            beta_short=BETA_SHORT, d_thresh_short=D_THRESH_SHORT)
        sub_proj = sub_proj[sub_proj['time_norm_sec'].notna()].copy()
        sub_proj['surface_plot'] = sub_proj['surface'].fillna('Unknown')
        if len(sub_proj) == 0:
            continue

        # 2. CS-predicted time at this anchor for every date in summary
        cs_times = cs_line_at_anchor(daily_summary, anchor, beta_long, d_thresh,
                                      beta_short=BETA_SHORT,
                                      d_thresh_short=D_THRESH_SHORT)
        # 3. Hand-drawn cubic re-projected to this anchor (gives a
        #    parallel time series for the 2008-04-26 → 2013-05-26 window)
        dotted_dates, dotted_times = cubic_at_anchor(
            daily_summary, cubic_coefs, cubic_t0, hd_start, hd_end, anchor,
            beta_long, d_thresh,
            beta_short=BETA_SHORT, d_thresh_short=D_THRESH_SHORT)

        # 4. CS-prediction lookup per race date (cubic in handdrawn window,
        #    real CS line elsewhere)
        cs_by_date = pd.Series(cs_times, index=daily_summary['date'].dt.date.values)
        cubic_by_date = (pd.Series(dotted_times,
                                   index=pd.DatetimeIndex(dotted_dates).date)
                         if len(dotted_dates) else pd.Series(dtype=float))
        hd_end_date = hd_end.date()

        def _cs_at(d):
            d_date = d.date() if hasattr(d, 'date') else d
            if d_date <= hd_end_date and d_date in cubic_by_date.index:
                return float(cubic_by_date.loc[d_date])
            v = cs_by_date.get(d_date, np.nan)
            return float(v) if v is not None and not pd.isna(v) else np.nan

        sub_proj['cs_pred_sec'] = sub_proj['date'].apply(_cs_at)
        sub_proj['hover'] = sub_proj.apply(
            lambda row: build_hover_anchored(row, anchor), axis=1)

        # 5. CS solid trace (post-handdrawn) and dotted (handdrawn).
        #    Only the first subplot puts the entries in the legend.
        daily_dates = daily_summary['date'].values
        mask_solid = daily_dates >= np.datetime64(hd_end)
        fig2.add_trace(go.Scatter(
            x=daily_dates[mask_solid],
            y=cs_times[mask_solid],
            mode='lines', name='CS-derived',
            line=dict(color=CS_LINE_COLOR, width=CS_LINE_WIDTH),
            hoverinfo='skip', legendgroup='cs',
            showlegend=(not cs_legend_drawn),
            legendrank=2000),
            row=r, col=c)
        if len(dotted_dates):
            fig2.add_trace(go.Scatter(
                x=dotted_dates, y=dotted_times,
                mode='lines', name='Estimated',
                line=dict(color=CS_LINE_COLOR, width=CS_LINE_WIDTH, dash='dot'),
                hoverinfo='skip', legendgroup='cs',
                showlegend=(not cs_legend_drawn),
                legendrank=2001),
                row=r, col=c)
        cs_legend_drawn = True

        # 6. Race markers, organized by surface. Fatigued races are not
        #    visually distinguished — fatigue info lives in the hover.
        for surf in SURFACE_LEGEND_ORDER:
            s2 = sub_proj[sub_proj['surface_plot'] == surf]
            if len(s2) == 0:
                continue
            color = SURFACE_COLORS.get(surf, '#888888')
            show_legend = surf not in surfaces_seen
            surfaces_seen.add(surf)
            fig2.add_trace(go.Scatter(
                x=s2['date'], y=s2['time_norm_sec'],
                mode='markers', name=surf,
                marker=dict(color=color, size=8, symbol='diamond',
                            opacity=0.85,
                            line=dict(width=0.5, color='white')),
                hovertemplate='%{customdata}<extra></extra>',
                customdata=s2['hover'],
                legendgroup=surf, showlegend=show_legend),
                row=r, col=c)

        # 6b. Within-panel PR overlay. Running min on time_norm_sec
        #     (= time at this panel's anchor distance), computed across all
        #     PR-eligible races in this panel — Downhill races are excluded
        #     from PR competition (they still display as colored markers,
        #     they just don't earn the white ring).
        eligible = sub_proj[sub_proj['surface_plot'].apply(is_pr_eligible)]
        is_pr_panel = compute_pr_mask(eligible, value_col='time_norm_sec')
        pr_panel = eligible[is_pr_panel].sort_values('date')
        if len(pr_panel) > 0:
            owns_legend = not pr_legend_drawn
            fig2.add_trace(go.Scatter(
                x=pr_panel['date'], y=pr_panel['time_norm_sec'],
                mode='markers', name=PR_LEGEND_NAME,
                marker=pr_marker(base_size=8),
                hoverinfo='skip',
                legendgroup='pr',
                showlegend=owns_legend,
                legendrank=PR_LEGEND_RANK,
                # Only the legend-owning trace flags as sentinel — that's
                # the one whose click event the JS handler intercepts.
                meta=({'is_pr_legend_sentinel': True} if owns_legend else None)),
                row=r, col=c)
            pr_legend_drawn = True

        # 7. Per-subplot dynamic axes
        x_data_min = sub_proj['date'].min()
        x_data_max = sub_proj['date'].max()
        x_pad_days = max(int((x_data_max - x_data_min).days * 0.05), 30)
        x_lo_sub = x_data_min - pd.Timedelta(days=x_pad_days)
        x_hi_sub = x_data_max + pd.Timedelta(days=x_pad_days)

        # Y-range pulls from data, plus CS values within this subplot's x-range
        in_range = ((daily_summary['date'] >= x_lo_sub) &
                    (daily_summary['date'] <= x_hi_sub)).values
        cs_in = cs_times[in_range]
        cs_in = cs_in[~np.isnan(cs_in)]
        if len(dotted_dates):
            dot_mask = ((dotted_dates >= x_lo_sub) &
                        (dotted_dates <= x_hi_sub))
            dot_in = dotted_times[dot_mask]
            dot_in = dot_in[~np.isnan(dot_in)]
        else:
            dot_in = np.array([])
        y_candidates = np.concatenate([
            sub_proj['time_norm_sec'].values, cs_in, dot_in])
        y_min = float(np.nanmin(y_candidates))
        y_max = float(np.nanmax(y_candidates))
        y_span = y_max - y_min
        y_pad = y_span * 0.05 if y_span > 0 else max(y_max * 0.05, 1.0)
        y_lo_sub = y_min - y_pad
        y_hi_sub = y_max + y_pad

        ticks, labels = (
            time_ticks_at_interval(y_lo_sub, y_hi_sub, PANEL_TICK_SEC[name])
            if name in PANEL_TICK_SEC
            else auto_time_ticks(y_lo_sub, y_hi_sub, target_count=7))
        xticks, xlabels = thin_yearly_ticks(x_lo_sub, x_hi_sub, max_labels=5)
        fig2.update_xaxes(
            range=[x_lo_sub, x_hi_sub],
            tickmode='array', tickvals=xticks, ticktext=xlabels,
            showgrid=True, gridcolor='#333',
            row=r, col=c)
        fig2.update_yaxes(
            range=[y_hi_sub, y_lo_sub],   # reversed: faster up
            tickmode='array', tickvals=ticks, ticktext=labels,
            showgrid=True, gridcolor='#333',
            row=r, col=c)

    fig2.update_layout(
        title=dict(text='Lifetime races normalized by distance<br>'
                        '<sub style="font-size:13px;color:#bbb">'
                        'Hyperbolic CS projection with time prediction lines'
                        '</sub>',
                   y=0.97, yanchor='top'),
        template='plotly_dark',
        paper_bgcolor='#1a1a1a', plot_bgcolor='#1a1a1a',
        font=dict(color='#eee'),
        autosize=True,
        hovermode='closest',
        margin=dict(t=130, l=70, r=200, b=60),
        legend=dict(yanchor='top', y=0.99, xanchor='left', x=1.02))
    fig2.update_xaxes(title_text='Date', row=2)

    out2 = os.path.join(args.out_dir, 'race_pace_by_distance.html')
    write_dark_html(fig2, out2, extra_html=build_pr_nonclick_js())
    print(f'Wrote {out2}')

    # (The dedicated 5K plot was replaced by the distance-filter checkboxes
    # on the all-races plot — uncheck everything except 5K to recreate it.)

    # ---------- summary tables ----------
    print('\n========================================================================')
    print('RACE COUNT BY DISTANCE GROUP × SURFACE')
    print('========================================================================')
    summary = (elig.groupby(['group', 'surface_plot']).size()
                   .unstack(fill_value=0))
    ordered = [g for g in group_names if g in summary.index]
    print(summary.reindex(ordered).to_string())
    n_unmatched = elig['group'].isna().sum()
    if n_unmatched:
        print(f'\n{n_unmatched} races did not match any distance group at '
              f'{TOLERANCE*100:.0f}% tolerance (still on the all-races plot, '
              f'absent from the 8-bin grid).')
    print(f'\nFatigued (race_seq>1) races plotted: {int(elig["fatigued"].sum())}')


if __name__ == '__main__':
    main()
