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
  python src/plots/make_race_plots.py

Defaults read CS fit outputs and races.csv from data/ and write the HTML
into output/. Override with --in-dir, --races, --out-dir.

Dependencies: pandas, numpy, scipy, plotly.
"""
import argparse
import datetime as dt
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.shared.paths import DATA_DIR, OUTPUT_DIR
from src.shared.cs_projection import (load_cs_outputs, project_races_to_5k_pace,
                                       cs_line_at_anchor, cubic_at_anchor)
from src.plotting import (render_plot, CursorTooltip, apply_default_layout,
                            right_margin_for_anchored_box,
                            sec_to_mss, sec_to_mss_full,
                            SURFACES, CS_LINE, CS_LINE_WIDTH, GRID,
                            pr_marker, is_pr_eligible,
                            PR_LEGEND_NAME, PR_LEGEND_RANK,
                            yearly_x_axis_kwargs)
from src.plotting import widgets

_PLOTS_DIR = Path(__file__).resolve().parent
_FILTER_JS = _PLOTS_DIR / 'make_race_plots_filter.js'
_PANEL_PRS_JS = _PLOTS_DIR / 'make_race_plots_panel_prs.js'

# Width of the distance-filter box (#bin-filter); also used to size margin.r.
BIN_FILTER_WIDTH = 150


DEFAULT_IN_DIR = str(DATA_DIR)
DEFAULT_RACES  = str(DATA_DIR / 'races.csv')
DEFAULT_OUT    = str(OUTPUT_DIR)

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
    ('3000m',    3000),
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
    'Mile':     lambda n: f'Mile (including 1500m, n={n})',
    '3000m':   lambda n: f'3000m (including 2 Mile, n={n})',
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
    '3000m':   60,
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


# ---------- visual encoding ----------
# Surface palette + legend order. Colors come from src.plotting.tokens.SURFACES
# (canonical hex). pr_marker / is_pr_eligible / PR_LEGEND_* live in
# src.plotting.markers.
SURFACE_LEGEND_ORDER = ['Road', 'Track', 'XC', 'Downhill', 'Unknown']


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
    # The smart-spikeline scaffold prepends the date itself in smooth
    # mode, so this content focuses on what's race-specific.
    parts = [f"Distance: {friendly_distance(row['distance_m'])}",
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
    parts = [f"Distance: {friendly_distance(row['distance_m'])}",
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


def build_distance_filter_ui(bin_names):
    """HTML for the right-sidebar distance-filter checkboxes.

    Behavior — checkbox toggles trace visibility by meta.filter_bin,
    plus PR-overlay recompute on any visibility change — lives in
    make_race_plots_filter.js (loaded via overlay_js_files).
    """
    body = (
        widgets.title('Distance')
        + '\n'
        + widgets.button_row([('bf-all', 'All'), ('bf-none', 'None')])
        + '\n'
        + widgets.checkbox_rows(
            [(b, b) for b in bin_names], data_attr='bin'
        )
    )
    return widgets.sidebar(
        'bin-filter', body=body, width_px=BIN_FILTER_WIDTH
    )


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
        line=dict(color=CS_LINE, width=CS_LINE_WIDTH),
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
        line=dict(color=CS_LINE, width=CS_LINE_WIDTH),
        hoverinfo='skip', showlegend=True, legendgroup='cs',
        legendrank=2000))
    fig.add_trace(go.Scatter(
        x=dotted_dates, y=dotted_ys,
        mode='lines', name='Estimated 5K pace',
        line=dict(color=CS_LINE, width=CS_LINE_WIDTH, dash='dot'),
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
        color = SURFACES.get(surf, '#888888')
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
        # snap_eligible = True so the smart spikeline scaffold treats each
        # race marker as a snap target and reads customdata for snap content.
        for bin_name in bin_names:
            s2 = sub[sub['filter_bin'] == bin_name]
            if len(s2) == 0:
                continue
            fig.add_trace(go.Scatter(
                x=s2['date'], y=s2['pace_norm_min'],
                mode='markers', name=surf,
                marker=common_marker,
                hoverinfo='skip',
                customdata=s2['hover'],
                legendgroup=surf, showlegend=False,
                meta={'filter_bin': bin_name,
                      'pr_eligible': is_pr_eligible(surf),
                      'snap_eligible': True}))


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


def yearly_x_axis(x_lo, x_hi, **kwargs):
    """Yearly gridline + label config for the x-axis (matches CS plot)."""
    return yearly_x_axis_kwargs(x_lo, x_hi, **kwargs)


def reversed_pace_y_axis(**kwargs):
    base = dict(title='5K-equivalent pace (min/mi)',
                range=[Y_MAX, Y_MIN],
                tickmode='array', tickvals=YTICK_VALS, ticktext=YTICK_TXT,
                showgrid=True, gridcolor=GRID)
    base.update(kwargs)
    return base


# ---------- main ----------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--in-dir', default=DEFAULT_IN_DIR,
                   help=f'Directory containing bayes_cs_summary{{tag}}.csv and '
                        f'bayes_cs_params{{tag}}.csv (default: {DEFAULT_IN_DIR}).')
    p.add_argument('--tag', default='',
                   help='Fit tag suffix (default: empty / unversioned).')
    p.add_argument('--races', default=DEFAULT_RACES,
                   help=f'Path to races.csv (default: {DEFAULT_RACES}).')
    p.add_argument('--out-dir', default=DEFAULT_OUT,
                   help=f'Output directory (default: {DEFAULT_OUT}).')
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
    apply_default_layout(
        fig1,
        hovermode='closest',
        margin=dict(t=20, l=70,
                    r=right_margin_for_anchored_box(BIN_FILTER_WIDTH, legend_min_px=200),
                    b=60),
        legend=dict(yanchor='top', y=0.99, xanchor='left', x=1.02),
        xaxis=yearly_x_axis(x_lo, x_hi, title='Date'),
        yaxis=reversed_pace_y_axis())

    out1 = os.path.join(args.out_dir, 'race_pace_all.html')
    filter_ui = build_distance_filter_ui([b[0] for b in FILTER_BINS])

    # Smooth-mode tooltip payload: per-day CS pace + sorted race list, so
    # the cursor scaffold can show "CS pace at this date + nearest race
    # within ±N days" when hovering empty space. Snap mode is per-marker
    # via customdata (already populated by add_race_traces_filterable).
    js_epoch = pd.Timestamp('1970-01-01')
    cs_dates = pd.to_datetime(daily_summary['date'])
    cs_first_day = int((cs_dates.iloc[0] - js_epoch).days)
    cs_last_day  = int((cs_dates.iloc[-1] - js_epoch).days)
    cs_pace_per_day = [round(float(v), 4)
                       for v in daily_summary['p5k_implied_min'].values]

    sessions_all = []
    for _, r in elig.iterrows():
        sessions_all.append({'day': int((r['date'] - js_epoch).days),
                             'html': r['hover']})
    sessions_all.sort(key=lambda s: s['day'])

    payload_all = {
        'first_day': cs_first_day,
        'cs_pace':   cs_pace_per_day,
        'sessions':  sessions_all,
        'nearest_window_days': 60,
    }
    smooth_build_js_all = r"""
function buildTooltip(day, isSnap, pointHtml) {
  var P = window.__TT_DATA;
  var idx = day - P.first_day;
  if (idx < 0 || idx >= P.cs_pace.length) return '';

  var DOW = ['Sun','Mon','Tue','Wed','Thu','Fri','Sat'];
  function paceMSS(min) {
    if (min == null || isNaN(min)) return '—';
    var s = Math.round(min * 60);
    var mn = Math.floor(s / 60), sc = s % 60;
    return mn + ':' + (sc < 10 ? '0' : '') + sc;
  }
  function dateLabel(d) {
    var dt = new Date(d * 86400000);
    var y = dt.getUTCFullYear();
    var m = String(dt.getUTCMonth() + 1).padStart(2, '0');
    var dd = String(dt.getUTCDate()).padStart(2, '0');
    return y + '-' + m + '-' + dd + ' (' + DOW[dt.getUTCDay()] + ')';
  }

  var trendLabel = new Date(day) >= new Date('2013-05-26').getTime() / 86400000 ? 'CS-derived pace' : 'Estimated pace';

  var html = '';
  html += '<div class="tt-date">' + dateLabel(day) + '</div>';

  // Section 1: trend info — CS-derived 5K pace at this date.
  html += '<div class="tt-section">';
  html += '<div class="tt-row"><span>' + trendLabel + '</span><b>' + paceMSS(P.cs_pace[idx]) + '/mi</b></div>';
  html += '</div>';

  // Section 2: race details. Smooth = nearest race within window.
  var run = null;
  var s = P.sessions;
  if (isSnap) {
    var lo = 0, hi = s.length - 1;
    while (lo < hi) {
      var mid = (lo + hi) >> 1;
      if (s[mid].day < day) lo = mid + 1; else hi = mid;
    }
    if (s[lo] && s[lo].day === day) run = s[lo];
  } else if (s.length) {
    var lo = 0, hi = s.length - 1;
    while (lo < hi) {
      var mid = (lo + hi) >> 1;
      if (s[mid].day < day) lo = mid + 1; else hi = mid;
    }
    var cands = [s[lo]];
    if (lo > 0) cands.push(s[lo - 1]);
    var best = null, bestAbs = 9999;
    for (var k = 0; k < cands.length; k++) {
      var ad = Math.abs(cands[k].day - day);
      if (ad < bestAbs) { bestAbs = ad; best = cands[k]; }
    }
    if (best && bestAbs <= P.nearest_window_days) run = best;
  }

  if (run || (isSnap && pointHtml)) {
    html += '<div class="tt-section">';
    if (!isSnap && run) {
      var dd2 = run.day - day;
      var lbl = dd2 === 0 ? 'same day'
              : (dd2 > 0 ? '+' + dd2 + ' day' + (dd2 === 1 ? '' : 's')
                         :  dd2 + ' day' + (dd2 === -1 ? '' : 's'));
      html += '<div class="tt-section-title">Nearest race [' + lbl + ']</div>';
    }
    html += (isSnap && pointHtml ? pointHtml : run.html);
    html += '</div>';
  }
  return html;
}
"""

    render_plot(
        fig1, out1,
        title_slug='race_pace_all',
        page_title='Races',
        title='Lifetime races: 5K-equivalent pace',
        subtitle='Hyperbolic CS projection with corrections applied for short and long distances',
        cursor_tooltip=CursorTooltip(
            payload=payload_all,
            build_js=smooth_build_js_all,
            first_day=cs_first_day,
            last_day=cs_last_day,
        ),
        overlay_html=filter_ui,
        overlay_js_files=[_FILTER_JS],
    )
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

    # Per-panel data for the smart-spikeline buildTooltip — keyed by the
    # subplot's xaxis id (Plotly assigns 'x', 'x2', ... left-to-right /
    # top-to-bottom). The scaffold passes the active xaxisId to
    # buildTooltip, which then looks up the right panel's CS prediction
    # array and race list.
    js_epoch = pd.Timestamp('1970-01-01')
    panel_payload = {}
    panel_first_day = int((daily_summary['date'].iloc[0] - js_epoch).days)
    panel_last_day  = int((daily_summary['date'].iloc[-1] - js_epoch).days)
    daily_dates_idx = ((daily_summary['date'] - js_epoch).dt.days).astype(int).values

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

        # Build the per-panel tooltip payload. cs_for_day uses the cubic
        # in the hand-drawn era and the real CS line afterwards (same
        # rule as _cs_at above). round() keeps the JSON small.
        cs_for_day = []
        for d_idx, d_val in zip(daily_dates_idx, daily_summary['date']):
            d_date = d_val.date()
            v = (cubic_by_date.loc[d_date] if d_date <= hd_end_date
                 and d_date in cubic_by_date.index else cs_by_date.get(d_date, np.nan))
            cs_for_day.append(None if v is None or pd.isna(v) else round(float(v), 1))
        panel_sessions = []
        for _, r2 in sub_proj.iterrows():
            panel_sessions.append({
                'day':  int((r2['date'] - js_epoch).days),
                'time_norm_sec': (None if pd.isna(r2.get('time_norm_sec'))
                                  else round(float(r2['time_norm_sec']), 1)),
                'cs_pred_sec':   (None if pd.isna(r2.get('cs_pred_sec'))
                                  else round(float(r2['cs_pred_sec']), 1)),
                'html':  r2['hover'],
            })
        panel_sessions.sort(key=lambda s: s['day'])

        # Plotly 'x' for the first subplot, 'x2' onward for subsequent.
        # The make_subplots layout ordering puts row=1 col=1 at index 1,
        # row=1 col=2 at index 2, etc. (i.e. left-to-right, top-to-bottom).
        sp_idx = i + 1
        xa_id = 'x' if sp_idx == 1 else f'x{sp_idx}'
        panel_payload[xa_id] = {
            'name':        name,
            'anchor_m':    anchor,
            'cs_pred_sec': cs_for_day,
            'sessions':    panel_sessions,
        }

        # 5. CS solid trace (post-handdrawn) and dotted (handdrawn).
        #    Only the first subplot puts the entries in the legend.
        daily_dates = daily_summary['date'].values
        mask_solid = daily_dates >= np.datetime64(hd_end)
        fig2.add_trace(go.Scatter(
            x=daily_dates[mask_solid],
            y=cs_times[mask_solid],
            mode='lines', name='CS-derived',
            line=dict(color=CS_LINE, width=CS_LINE_WIDTH),
            hoverinfo='skip', legendgroup='cs',
            showlegend=(not cs_legend_drawn),
            legendrank=2000),
            row=r, col=c)
        if len(dotted_dates):
            fig2.add_trace(go.Scatter(
                x=dotted_dates, y=dotted_times,
                mode='lines', name='Estimated',
                line=dict(color=CS_LINE, width=CS_LINE_WIDTH, dash='dot'),
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
            color = SURFACES.get(surf, '#888888')
            show_legend = surf not in surfaces_seen
            surfaces_seen.add(surf)
            fig2.add_trace(go.Scatter(
                x=s2['date'], y=s2['time_norm_sec'],
                mode='markers', name=surf,
                marker=dict(color=color, size=8, symbol='diamond',
                            opacity=0.85,
                            line=dict(width=0.5, color='white')),
                hoverinfo='skip',
                customdata=s2['hover'],
                legendgroup=surf, showlegend=show_legend,
                meta={'panel_name': name,
                      'pr_eligible': bool(is_pr_eligible(surf)),
                      'snap_eligible': True}),
                row=r, col=c)

        # 6b. Within-panel PR overlay. Running min on time_norm_sec
        #     (= time at this panel's anchor distance), computed across all
        #     PR-eligible races in this panel — Downhill races are excluded
        #     from PR competition (they still display as colored markers,
        #     they just don't earn the white ring). Tagged with panel_name
        #     so the legend-toggle handler can recompute it per panel.
        #
        #     The legend entry for "PR effort" lives on a dedicated
        #     off-range sentinel trace added once after this loop; that
        #     way the legend item never disappears when a panel's overlay
        #     goes empty (e.g. user hides the only surface in that bin).
        eligible = sub_proj[sub_proj['surface_plot'].apply(is_pr_eligible)]
        is_pr_panel = compute_pr_mask(eligible, value_col='time_norm_sec')
        pr_panel = eligible[is_pr_panel].sort_values('date')
        if len(pr_panel) > 0:
            fig2.add_trace(go.Scatter(
                x=pr_panel['date'], y=pr_panel['time_norm_sec'],
                mode='markers', name=PR_LEGEND_NAME,
                marker=pr_marker(base_size=8),
                hoverinfo='skip',
                legendgroup='pr',
                showlegend=False,
                meta={'panel_name': name, 'is_pr_overlay': True}),
                row=r, col=c)

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
        fig2.update_xaxes(
            **yearly_x_axis_kwargs(x_lo_sub, x_hi_sub, max_labels=5),
            row=r, col=c)
        fig2.update_yaxes(
            range=[y_hi_sub, y_lo_sub],   # reversed: faster up
            tickmode='array', tickvals=ticks, ticktext=labels,
            showgrid=True, gridcolor=GRID,
            row=r, col=c)

    # Off-range PR-legend sentinel — a single point at (1900-01-01, 6.0)
    # outside every subplot's x-range. Hosts the legend entry that
    # represents PR effort across all panels; never goes empty even when
    # a panel's overlay is restyled to []. Click is suppressed by the
    # plotly_legendclick handler (returns false on this trace).
    fig2.add_trace(go.Scatter(
        x=[pd.Timestamp('1900-01-01')], y=[6.0],
        mode='markers', name=PR_LEGEND_NAME,
        marker=pr_marker(base_size=8),
        hoverinfo='skip',
        legendgroup='pr',
        showlegend=True,
        legendrank=PR_LEGEND_RANK,
        meta={'is_pr_legend_sentinel': True}),
        row=1, col=1)

    apply_default_layout(
        fig2,
        hovermode='closest',
        margin=dict(t=40, l=70, r=200, b=60),
        legend=dict(yanchor='top', y=0.99, xanchor='left', x=1.02))
    fig2.update_xaxes(title_text='Date', row=2)

    out2 = os.path.join(args.out_dir, 'race_pace_by_distance.html')

    # Smooth + snap behaviour per panel. The scaffold passes the cursor's
    # active xaxisId; buildTooltip indexes panel_payload by that ID to
    # show the panel's anchor + CS prediction + nearest race in time.
    payload_by_dist = {
        'first_day': panel_first_day,
        'panels':    panel_payload,
        'nearest_window_days': 60,
    }
    smooth_build_js_by_dist = r"""
function buildTooltip(day, isSnap, pointHtml, ctx) {
  var P = window.__TT_DATA;
  var xaId = (ctx && ctx.xaxisId) || 'x';
  var panel = P.panels[xaId];
  if (!panel) return '';
  var idx = day - P.first_day;
  if (idx < 0 || idx >= panel.cs_pred_sec.length) return '';

  var DOW = ['Sun','Mon','Tue','Wed','Thu','Fri','Sat'];
  function timeFmt(s) {
    if (s == null) return '—';
    var x = Math.round(s);
    if (x >= 3600) {
      var h = Math.floor(x / 3600), m = Math.floor((x % 3600) / 60), sc = x % 60;
      return h + ':' + (m < 10 ? '0' : '') + m + ':' + (sc < 10 ? '0' : '') + sc;
    }
    var mn = Math.floor(x / 60), sc = x % 60;
    return mn + ':' + (sc < 10 ? '0' : '') + sc;
  }
  function distLabel(m) {
    if (Math.abs(m - 1609.344) < 5) return '1 mi';
    if (Math.abs(m - 3218.688) < 5) return '2 mi';
    if (m >= 19410 && m <= 22785) return 'half marathon';
    if (m >= 38819 && m <= 45570) return 'marathon';
    return Math.round(m) + 'm';
  }
  function dateLabel(d) {
    var dt = new Date(d * 86400000);
    var y = dt.getUTCFullYear();
    var mo = String(dt.getUTCMonth() + 1).padStart(2, '0');
    var dd = String(dt.getUTCDate()).padStart(2, '0');
    return y + '-' + mo + '-' + dd + ' (' + DOW[dt.getUTCDay()] + ')';
  }

  var html = '';
  html += '<div class="tt-date">' + dateLabel(day) + '</div>';

  // Section 1: per-panel CS prediction at the panel's anchor distance.
  html += '<div class="tt-section">';
  html += '<div class="tt-row"><span>CS-predicted ' + distLabel(panel.anchor_m) + '</span><b>'
        + timeFmt(panel.cs_pred_sec[idx]) + '</b></div>';
  html += '</div>';

  // Section 2: race details. Smooth = nearest within window in this panel.
  var run = null;
  var s = panel.sessions;
  if (isSnap) {
    var lo = 0, hi = s.length - 1;
    while (lo < hi) {
      var mid = (lo + hi) >> 1;
      if (s[mid].day < day) lo = mid + 1; else hi = mid;
    }
    if (s[lo] && s[lo].day === day) run = s[lo];
  } else if (s.length) {
    var lo = 0, hi = s.length - 1;
    while (lo < hi) {
      var mid = (lo + hi) >> 1;
      if (s[mid].day < day) lo = mid + 1; else hi = mid;
    }
    var cands = [s[lo]];
    if (lo > 0) cands.push(s[lo - 1]);
    var best = null, bestAbs = 9999;
    for (var k = 0; k < cands.length; k++) {
      var ad = Math.abs(cands[k].day - day);
      if (ad < bestAbs) { bestAbs = ad; best = cands[k]; }
    }
    if (best && bestAbs <= P.nearest_window_days) run = best;
  }

  if (run || (isSnap && pointHtml)) {
    html += '<div class="tt-section">';
    if (!isSnap && run) {
      var dd2 = run.day - day;
      var lbl = dd2 === 0 ? 'same day'
              : (dd2 > 0 ? '+' + dd2 + ' day' + (dd2 === 1 ? '' : 's')
                         :  dd2 + ' day' + (dd2 === -1 ? '' : 's'));
      html += '<div class="tt-section-title">Nearest race [' + lbl + ']</div>';
    }
    html += (isSnap && pointHtml ? pointHtml : run.html);
    html += '</div>';
  }
  return html;
}
"""

    render_plot(
        fig2, out2,
        title_slug='race_pace_by_distance',
        page_title='Races by distance',
        title='Lifetime races normalized by distance',
        subtitle='Hyperbolic CS projection with time prediction lines',
        cursor_tooltip=CursorTooltip(
            payload=payload_by_dist,
            build_js=smooth_build_js_by_dist,
            first_day=panel_first_day,
            last_day=panel_last_day,
        ),
        overlay_js_files=[_PANEL_PRS_JS],
    )
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
