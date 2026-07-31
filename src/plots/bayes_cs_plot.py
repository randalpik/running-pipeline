"""Plot the Bayesian CS timeline with credible-interval ribbons.

Reads the outputs of bayes_cs_fit.py (summary CSV, params CSV) and renders an
interactive HTML chart. Race diamonds are projected to 5K-equivalent pace
via the hyperbolic CS model: for each race at (d_race, t_race), the implied
CS is (d_race - D')/t_race, anchored to the model's posterior-median D' at
the race's date. The 5K-equivalent then evaluates as
t_5K = (5000 - D') * t_race / (d_race - D'). This preserves race-to-race
deviations (fast races project fast, slow races project slow) while handling
short distances correctly via D' rather than Riegel's piecewise exponent.

5K is chosen as the anchor distance because the vast majority of Max's races
are 5Ks (track 5Ks, XC 5Ks, road 5Ks across all eras).

XC race times are pre-corrected by dividing by (1 + xc_correction) before
projection — same correction the fit script applies before fitting — so XC
diamonds visually align with the CS line.

Default behavior: reads CS fit outputs from data/ (the bayes fit's default
output directory) and writes the HTML to output/.

Usage:
  python src/plots/bayes_cs_plot.py
  python src/plots/bayes_cs_plot.py --tag v9
  python src/plots/bayes_cs_plot.py --in-dir /path/to/outputs --tag v9

Required dependencies:
  pip install pandas numpy scipy plotly
"""
import argparse
import os
import sys
import datetime as dt
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.shared.paths import DATA_DIR, OUTPUT_DIR
from src.shared.units import METERS_PER_MILE
from src.shared.plot_window import data_span, first_race_date, axis_pad_entry
from src.shared.cs_projection import load_cs_outputs, project_races_to_5k_pace
from src.shared.performance_frontier import standard_demos, build_frontier_band
from src.plotting import (render_plot, CursorTooltip, apply_default_layout,
                            sec_to_mss, sec_to_mss_prec, time_decimals,
                            SURFACES, rgba, GRID,
                            FRONTIER_LINE, CAT_COLORS, CAT_LABEL, yearly_x_axis_kwargs,
                            nice_time_ticks, marker_half_px, tt_title)


DEFAULT_IN_DIR = str(DATA_DIR)
DEFAULT_RACES  = str(DATA_DIR / 'races.csv')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--in-dir', default=DEFAULT_IN_DIR,
                   help=f'Directory containing fit outputs (default {DEFAULT_IN_DIR})')
    p.add_argument('--tag', default='',
                   help='Tag suffix matching the fit run (e.g. "v9")')
    p.add_argument('--races', default=DEFAULT_RACES,
                   help='Path to races.csv used by the fit')
    p.add_argument('--out', default=None,
                   help=f'Output HTML path (default: cs_timeline{{tag}}.html in {OUTPUT_DIR})')
    return p.parse_args()


def main():
    args = parse_args()
    suffix = f'_{args.tag}' if args.tag else ''

    # Load summary (already daily-interpolated) and bias parameters via shared module.
    daily_summary, beta_long_med, d_thresh_long, xc_correction = load_cs_outputs(
        args.in_dir, args.tag)
    print(f"Bias parameters: β_long={beta_long_med:.4f}, "
          f"xc_correction={xc_correction:.4f}, d_thresh={d_thresh_long:.0f}m")
    print(f"Summary: {len(daily_summary)} daily points, "
          f"{daily_summary['date'].iloc[0].date()} to {daily_summary['date'].iloc[-1].date()}")

    # ---------- load and filter races ----------
    if not os.path.exists(args.races):
        sys.exit(f"ERROR: races file not found: {args.races}")
    races = pd.read_csv(args.races, parse_dates=['date'])
    races['date_d'] = races['date'].dt.date

    if 'fatigued' not in races.columns:
        races['fatigued'] = False
    if 'surface' not in races.columns:
        races['surface'] = 'Unknown'

    elig = races[
        (~races['fatigued'].astype(bool)) &
        (races['surface'] != 'Downhill') &
        (races['time_sec'] >= 120)
    ].copy().sort_values('date')
    print(f"Hard-eligible races: {len(elig)}")

    # Apply the same auto-exclusions the fit applied. Match the fit's composite
    # key (date, distance_m) — see bayes_cs_fit.py.
    excl_path = os.path.join(args.in_dir, f'bayes_cs_auto_exclusions{suffix}.csv')
    if os.path.exists(excl_path):
        try:
            excl_df = pd.read_csv(excl_path, parse_dates=['date'])
        except pd.errors.EmptyDataError:
            # The fit writes a header-less file when there are 0 exclusions
            # (e.g. a profile with very few races) — nothing to exclude.
            excl_df = pd.DataFrame()
        if len(excl_df):
            excl_keys = set(zip(excl_df['date'].dt.date,
                                excl_df['distance_m'].astype(int)))
            elig['_key'] = list(zip(elig['date'].dt.date,
                                    elig['distance_m'].astype(int)))
            n_before = len(elig)
            elig = elig[~elig['_key'].isin(excl_keys)].drop(columns=['_key'])
            print(f"Applied {len(excl_df)} auto-exclusions: "
                  f"{n_before} -> {len(elig)} races")
    else:
        print(f"WARNING: no auto-exclusions file at {excl_path} — "
              f"plot will show all hard-eligible races")

    # ---------- truncate to plotting window + project races to 5K-equivalent ----------
    # Left bound: the Max profile is a hardcoded exception — pinned at
    # 2013-06-01 (his pre-2013 races are sparse with huge uncertainty). Every
    # other profile starts at its first race. project_races_to_5k_pace handles
    # XC pre-correction (matches the fit) and the β_long un-bias for distances
    # > d_thresh internally; see cs_projection.py for the rationale and formulas.
    if os.environ.get('RP_PROFILE', 'max') == 'max':
        window_start = pd.Timestamp('2013-06-01')
    else:
        fr = first_race_date()
        window_start = fr if fr is not None else pd.Timestamp(daily_summary['date'].min())
    summary_plot = (daily_summary[daily_summary['date'] >= window_start]
                    .copy().reset_index(drop=True))
    elig_plot = elig[elig['date'] >= window_start].copy()
    print(f'Plot window: {len(summary_plot)} daily points (from {window_start.date()})')

    # X-axis bounds: left at the (possibly hardcoded) window_start, right at the
    # latest logged run (data_span union), so the axis tracks current data — not
    # the last race, not the margin-extended fit grid. The CS line/band is drawn
    # over the full margin-extended summary and clipped here, so it reaches the
    # right edge; the pixel gutter (half a race diamond) is added by axis_pad.js.
    x_lo = window_start
    _, x_hi = data_span()

    # Project twice from the SAME race rows so each race shows the physical
    # route correction at its OWN distance, THEN converted to 5K (Max, June
    # 2026 — corrected-then-converted is the informative order, not convert-
    # then-adjust). `pace_norm_min` is the corrected diamond; `pace_norm_min_unc`
    # is the same race with the grade/footing/altitude correction OFF — the
    # "before correction" reference. The XC pre-correction stays on in both so
    # the only difference between them is the §B physical route correction.
    elig_proj = project_races_to_5k_pace(
        elig_plot, summary_plot, beta_long_med, d_thresh_long,
        apply_xc_correction=True, xc_correction=xc_correction)
    # "Before correction" baseline: BOTH corrections off (physical route AND the
    # categorical XC ×1.08) so the connector shows the TOTAL correction applied —
    # the §B physical correction on watch races AND the XC terrain factor on
    # pre-watch XC races (which have no watch data, so the categorical is what
    # corrects them).
    elig_unc = project_races_to_5k_pace(
        elig_plot, summary_plot, beta_long_med, d_thresh_long,
        apply_xc_correction=False, apply_physical_correction=False)
    elig_proj['pace_norm_min_unc'] = elig_unc['pace_norm_min'].to_numpy()
    elig_plot = elig_proj[elig_proj['pace_norm_min'].notna()].copy()
    print(f'Hyperbolic projection (5K-equiv): {len(elig_plot)} race diamonds')

    # ---------- performance frontier (red line) ----------
    # Demonstrated-5K-capability envelope over race 5K-equivalents + the kept
    # TQ corpus (see src/shared/performance_frontier.py for semantics).
    # standard_demos is the shared canonical set every tab builds from;
    # Fitness is the line's home.
    demos = standard_demos(daily_summary, beta_long_med, d_thresh_long,
                           xc_correction,
                           races_path=Path(args.races),
                           exclusions_path=Path(excl_path))
    frontier, front_lo, front_hi, demos = build_frontier_band(
        demos, pd.DatetimeIndex(summary_plot['date']), summary_plot)
    n_front = int(frontier['frontier_pace_min'].notna().sum())
    # Every non-race workout faster than the asymptotic CS line is shown
    # ("Frontier workouts" — context belongs on the chart even when a bigger
    # neighbor overshadows the point at its own date, or when the point can't
    # bind the frontier at all). The CS line is always slower than the
    # 5K-fitness floor used for `excess`/binding, so this is a superset of the
    # frontier-eligible (`excess > 0`) set; `binding` still flags only the
    # workouts that actually define the envelope (a hover annotation, not a
    # display filter).
    cs_asym_at = np.interp(
        demos['date'].to_numpy('datetime64[D]').astype(float),
        summary_plot['date'].to_numpy('datetime64[D]').astype(float),
        summary_plot['cs_pace_med'].to_numpy(float))
    above_cs = demos['pace_min'].to_numpy(float) < cs_asym_at
    front_workouts = demos[above_cs & (demos['src'] != 'race')].copy()
    print(f'Frontier: {int(above_cs.sum())} demos above the CS line '
          f'({len(front_workouts)} non-race, '
          f'{int(front_workouts["binding"].sum())} binding the frontier, '
          f'{int((front_workouts["excess"] > 0).sum())} above the 5K-fitness floor)')

    # ---------- figure ----------
    fig = go.Figure()

    # Posterior median CS — demoted to a faint reference (June 2026): the
    # graph's purpose is now 5K prediction, so the asymptotic CS line and
    # its CrI ribbons no longer dominate. The frontier carries the band.
    fig.add_trace(go.Scatter(
        x=summary_plot['date'], y=summary_plot['cs_pace_med'],
        mode='lines', line=dict(color=rgba('#ffb450', 0.55), width=2.0),
        name='Projected Critical Speed', hoverinfo='skip', showlegend=True,
        legendrank=4))
    # Frontier-swept 95% prediction band: the frontier recomputed with the
    # floor at the CS lo95/hi95 5K predictions. Collapses onto the line
    # where a demonstration binds (proof pins the prediction); equals the
    # CS CrI on the floor. Purple so band and line read as one object.
    fig.add_trace(go.Scatter(
        x=(summary_plot['date'].tolist()
           + summary_plot['date'].tolist()[::-1]),
        y=(front_lo['frontier_pace_min'].tolist()
           + front_hi['frontier_pace_min'].tolist()[::-1]),
        fill='toself', fillcolor=rgba(FRONTIER_LINE, 0.14),
        line=dict(width=0), mode='lines',
        name='95% prediction band', hoverinfo='skip', showlegend=True,
        legendrank=2))
    # CS-implied 5K prediction — the frontier's floor, BRIGHT gold (the
    # graph's primary gold object; the asymptotic CS median above is the
    # faint one). Same 5K-equiv space as the diamonds.
    fig.add_trace(go.Scatter(
        x=summary_plot['date'], y=summary_plot['p5k_implied_min'],
        mode='lines', line=dict(color='rgb(255,180,80)', width=2.5),
        name='5K fitness', hoverinfo='skip', showlegend=True,
        legendrank=3))
    # Performance frontier — demonstrated 5K capability (5K-equiv pace space,
    # same space as the race diamonds; the gold CS line is asymptotic pace).
    if n_front:
        fig.add_trace(go.Scatter(
            x=frontier['date'], y=frontier['frontier_pace_min'],
            mode='lines', line=dict(color=FRONTIER_LINE, width=2),
            connectgaps=False,
            name='Frontier 5K pace', hoverinfo='skip',
            showlegend=True, legendrank=1))
    # Race diamonds (bias-corrected). Colors derived from canonical SURFACES
    # hex tokens with 0.7 alpha so a re-skin only edits one place. Each
    # diamond carries a per-race snap-mode tooltip via customdata, and the
    # trace is tagged snap_eligible so the smart-spikeline scaffold treats
    # it as a snap target.
    surf_colors = {k: rgba(SURFACES[k], 0.7) for k in ('Track', 'Road', 'XC')}

    def _race_inner(row):
        ev = row.get('event') or '(no event)'
        if pd.isna(ev): ev = '(no event)'
        dist = float(row['distance_m'])
        dist_mi = dist / METERS_PER_MILE
        t_orig = float(row.get('time_sec_original', row['time_sec']))
        p_orig = row.get('pace_sec_per_mi_original',
                         row.get('pace_sec_per_mi'))
        pace_raw = (sec_to_mss(p_orig)
                    if p_orig is not None and not pd.isna(p_orig) else '')
        is_5k = abs(dist - 5000.0) < 1.0
        # Entered precision (races.csv time_dec) — the original time displays
        # exactly as logged, and the course-corrected time inherits the same
        # decimals (no more, no less). Value-inference is only a fallback.
        td = row.get('time_dec')
        td = int(td) if td is not None and not pd.isna(td) else time_decimals(t_orig)

        # Course correction (§B): the race's time/pace at its OWN distance after
        # the physical route correction (+ XC categorical, whatever applied) —
        # i.e. corrected, NOT yet converted to 5K. row['time_sec'] is exactly
        # that (project mutates it; time_sec_original is the raw race time).
        # Shown only when a correction actually moved the time.
        t_corr = float(row['time_sec'])
        has_corr = abs(t_corr - t_orig) >= 1.0
        corr_line = ''
        if has_corr:
            corr_line = (f"<div>Course correction: <b>{sec_to_mss_prec(t_corr, td)}</b> "
                         f"<span class='tt-mute'>"
                         f"({sec_to_mss(t_corr / dist_mi)}/mi)</span></div>")

        # 5K-equiv (the diamond's y). Suppressed for every 5K: the projection
        # is exact identity at 5000m, so the line would only echo the actual
        # time (uncorrected 5K) or the course-correction line (corrected 5K).
        equiv_pace_sec = float(row['pace_norm_min']) * 60
        equiv_time_sec = equiv_pace_sec * 5000.0 / METERS_PER_MILE
        equiv_line = ''
        if not is_5k:
            equiv_line = (f"<div>5K equivalent: <b>{sec_to_mss(equiv_time_sec)}</b> "
                          f"<span class='tt-mute'>"
                          f"({sec_to_mss(equiv_pace_sec)}/mi)</span></div>")
        return (f"<div><b>{ev}</b> <span class='tt-mute'>({row['surface']})</span></div>"
                f"<div>{int(dist)}m in "
                f"<b>{sec_to_mss_prec(t_orig, td)}</b> "
                f"<span class='tt-mute'>({pace_raw}/mi)</span></div>"
                f"{corr_line}{equiv_line}")

    # "Before correction" reference: for races a correction moved materially
    # (>1 s/mi at 5K-equiv), show an open diamond at the UNCORRECTED 5K-equiv
    # plus a connector to the corrected diamond — so the per-race effect (actual
    # time corrected at its own distance, then converted) is visible. Covers BOTH
    # the §B physical route correction (watch races) and the categorical XC
    # ×1.08 (pre-watch XC races). Toggleable via its own legend entry; default-on.
    # Added BEFORE the race diamonds so the solid diamonds draw ON TOP of these
    # connectors/ghosts (Plotly draws later traces above earlier ones).
    moved = elig_plot[(elig_plot['pace_norm_min_unc'].notna()) &
                      ((elig_plot['pace_norm_min_unc'] - elig_plot['pace_norm_min']).abs()
                       * 60 > 1.0)].copy()
    if len(moved):
        cx, cy = [], []
        for _, r in moved.iterrows():
            cx += [r['date'], r['date'], None]
            cy += [r['pace_norm_min_unc'], r['pace_norm_min'], None]
        fig.add_trace(go.Scatter(
            x=cx, y=cy, mode='lines', name='Before correction',
            line=dict(color='rgba(180,180,180,0.65)', width=1.2),
            legendgroup='precorr', legendgrouptitle_text='Correction',
            hoverinfo='skip'))
        fig.add_trace(go.Scatter(
            x=moved['date'], y=moved['pace_norm_min_unc'],
            mode='markers', name='Before correction', showlegend=False,
            marker=dict(color='rgba(185,185,185,0.0)', size=8, symbol='diamond-open',
                        line=dict(width=1.2, color='rgba(185,185,185,0.83)')),
            legendgroup='precorr', hoverinfo='skip',
            meta={'snap_eligible': False}))
        n_xc_moved = int((moved['surface'].astype(str).str.upper() == 'XC').sum())
        print(f'Before-correction reference: {len(moved)} races moved >1 s/mi '
              f'({n_xc_moved} XC)')

    for surf, col in surf_colors.items():
        sub = elig_plot[elig_plot['surface'] == surf]
        if len(sub) == 0: continue
        # Per-race inner HTML for snap mode. The scaffold prepends the
        # trend section (CS median + 50/95% intervals) and the date
        # header itself, so this is just the race-specific content.
        inner_html = sub.apply(_race_inner, axis=1).tolist()
        fig.add_trace(go.Scatter(
            x=sub['date'], y=sub['pace_norm_min'],
            mode='markers', name=surf,
            marker=dict(color=col, size=8, symbol='diamond',
                        line=dict(width=0.3, color='white')),
            customdata=inner_html,
            hoverinfo='skip',
            legendgroup='races',
            legendgrouptitle_text='Race pace (5K equivalent)',
            meta={'snap_eligible': True}))

    # Frontier workouts: EVERY non-race demonstration faster than the CS line,
    # rendered exactly as Training renders its session markers (per-category
    # colors/sizes, matching legend entries) under a "Frontier workouts"
    # legend group — small dots alongside the larger race diamonds. Binding
    # status (defines the envelope somewhere) is noted in the hover.
    # Legend labels + order: shared canonical labels, Tempo above Intervals.
    FRONTIER_CATS = [('tempo', CAT_LABEL['tempo']),
                     ('interval', CAT_LABEL['interval']),
                     ('rep', CAT_LABEL['rep']),
                     ('continuous_fartlek', CAT_LABEL['continuous_fartlek']),
                     ('long', CAT_LABEL['long']),
                     ('hill_cont', CAT_LABEL['hill_cont'])]

    def _disp_cat(row):
        if row['src'] == 'long_run':
            return 'long'
        if row['src'] == 'hill':
            return 'hill_cont'
        return row['category']

    def _frontier_title(row):
        # Tooltip title matches the dedicated tab: "Long run" / "Intervals" /
        # "Continuous hills" / "Hill repeats" + the route/location parenthetical.
        dc = _disp_cat(row)
        if dc == 'long':
            label = 'Long run'
        elif row['src'] == 'hill_rep':
            label = CAT_LABEL['hill_rep']
        else:
            label = CAT_LABEL.get(dc, str(row['src']).title())
        return tt_title(label, row.get('display_name'), row.get('city_state'))

    def _frontier_inner(row):
        pace_sec = float(row['pace_min']) * 60
        t5k_sec = pace_sec * 5000.0 / METERS_PER_MILE
        return (f"<div>{_frontier_title(row)}</div>"
                f"<div>{row['detail']}</div>"
                f"<div>5K equivalent: <b>{sec_to_mss(t5k_sec)}</b> "
                f"<span class='tt-mute'>({sec_to_mss(pace_sec)}/mi)</span></div>")

    if len(front_workouts):
        front_workouts['disp_cat'] = front_workouts.apply(_disp_cat, axis=1)
        for cat, label in FRONTIER_CATS:
            sub = front_workouts[front_workouts['disp_cat'] == cat]
            if sub.empty:
                continue
            fig.add_trace(go.Scatter(
                x=sub['date'], y=sub['pace_min'],
                mode='markers',
                name=f'{label} (n={len(sub)})',
                marker=dict(color=CAT_COLORS[cat], size=7,
                            line=dict(color='rgba(255,255,255,0.4)', width=0.5),
                            opacity=0.85),
                customdata=[_frontier_inner(r) for _, r in sub.iterrows()],
                hoverinfo='skip',
                legendgroup='frontier_workouts',
                legendgrouptitle_text='Frontier workouts',
                meta={'snap_eligible': True}))

    # ---------- layout ----------
    # Y-bounds + ticks enclosing the plotted data — the CS median line and the
    # race diamonds (the point estimates the axis is sized for; the 95% band
    # may extend past the edges for sparse fits). nice_time_ticks adapts the
    # interval to the data span; target=7 reproduces the former 30s spacing
    # over Max's ~4:30-8:00 range and adapts to any profile's range.
    _ys = np.concatenate([
        summary_plot['cs_pace_med'].to_numpy(dtype=float),
        elig_plot['pace_norm_min'].to_numpy(dtype=float),
        frontier['frontier_pace_min'].to_numpy(dtype=float),
        front_lo['frontier_pace_min'].to_numpy(dtype=float),
        front_hi['frontier_pace_min'].to_numpy(dtype=float),
        front_workouts['pace_min'].to_numpy(dtype=float),
    ])
    _ys = _ys[np.isfinite(_ys)]
    _lo, _hi = (float(_ys.min()), float(_ys.max())) if len(_ys) else (4.50, 8.00)
    _ticks_sec, ytick_txt = nice_time_ticks(_lo * 60, _hi * 60, target=7)
    ytick_vals = [t / 60.0 for t in _ticks_sec]
    y_min, y_max = ytick_vals[0], ytick_vals[-1]

    apply_default_layout(
        fig,
        margin=dict(t=20, l=70, r=200, b=28),
        legend=dict(yanchor='top', y=0.99, xanchor='left', x=1.02, groupclick='toggleitem'),
        xaxis=yearly_x_axis_kwargs(x_lo, x_hi),
        yaxis=dict(title='Pace (min/mi)', range=[y_max, y_min],
                   tickmode='array', tickvals=ytick_vals, ticktext=ytick_txt,
                   showgrid=True, gridcolor=GRID))

    # ---------- spikeline tooltip payload ----------
    # Per-day arrays for the trend section (CS median + 50/95% intervals)
    # plus a sorted session list (races + frontier workouts). Smooth mode
    # shows the trend section + nearest session within ±60d; snap mode shows
    # the trend section + the snapped session's details (no date, no
    # "Nearest event" header).
    epoch = pd.Timestamp('1970-01-01')

    def _round_pace(v):
        return round(float(v) * 60) if pd.notna(v) else None

    cs_pace_med  = [_round_pace(v) for v in summary_plot['cs_pace_med']]
    p5k_fit = [_round_pace(v) for v in summary_plot['p5k_implied_min']]
    fr_lo = [_round_pace(v) for v in front_lo['frontier_pace_min']]
    fr_hi = [_round_pace(v) for v in front_hi['frontier_pace_min']]

    sessions = []
    for _, r in elig_plot.iterrows():
        sessions.append({'day':  int((r['date'] - epoch).days),
                         'html': _race_inner(r)})
    # Frontier-setting workouts join the snap list so hovering near an open
    # red circle resolves to its session in smooth mode too.
    for _, r in front_workouts.iterrows():
        sessions.append({'day':  int((r['date'] - epoch).days),
                         'html': _frontier_inner(r)})
    sessions.sort(key=lambda s: s['day'])

    # Per-day frontier pace (sec/mi, null in gap breaks) for the trend section.
    frontier_sec = [_round_pace(v) for v in frontier['frontier_pace_min']]

    first_day = int((summary_plot['date'].iloc[0]  - epoch).days)
    last_day  = int((summary_plot['date'].iloc[-1] - epoch).days)

    payload = {
        'first_day': first_day,
        'cs_med':    cs_pace_med,
        'p5k':       p5k_fit,
        'fr_lo':     fr_lo,
        'fr_hi':     fr_hi,
        'frontier':  frontier_sec,
        'sessions':  sessions,
        'nearest_window_days': 60,
    }

    build_js = r"""
function buildTooltip(day, isSnap, pointHtml) {
  var P = window.__TT_DATA;
  var idx = day - P.first_day;
  if (idx < 0 || idx >= P.cs_med.length) return '';

  var DOW = ['Sun','Mon','Tue','Wed','Thu','Fri','Sat'];
  function paceMSS(s) {
    if (s == null) return '—';
    var x = Math.round(s);
    var mn = Math.floor(x / 60), sc = x % 60;
    return mn + ':' + (sc < 10 ? '0' : '') + sc;
  }
  function dateLabel(d) {
    var dt = new Date(d * 86400000);
    var y = dt.getUTCFullYear();
    var m = String(dt.getUTCMonth() + 1).padStart(2, '0');
    var dd = String(dt.getUTCDate()).padStart(2, '0');
    return y + '-' + m + '-' + dd + ' (' + DOW[dt.getUTCDay()] + ')';
  }

  var html = '';
  // Date header in both modes — see scaffold note in cursor_tooltip.js.
  html += '<div class="tt-date">' + dateLabel(day) + '</div>';

  // Section 1: per-day prediction summary — every right-justified value bold,
  // ordered to match the legend: performance frontier (the graph's primary
  // object), its 95% band, the CS-implied 5K fitness, then the asymptotic CS.
  html += '<div class="tt-section">';
  if (P.frontier && P.frontier[idx] != null) {
    html += '<div class="tt-row"><span>Frontier 5K pace</span><b>' + paceMSS(P.frontier[idx]) + '/mi</b></div>';
  }
  if (P.fr_lo && P.fr_lo[idx] != null && P.fr_hi && P.fr_hi[idx] != null) {
    html += '<div class="tt-row"><span>95% band</span><b>' + paceMSS(P.fr_lo[idx]) + '–' + paceMSS(P.fr_hi[idx]) + '/mi</b></div>';
  }
  if (P.p5k && P.p5k[idx] != null) {
    html += '<div class="tt-row"><span>5K fitness</span><b>' + paceMSS(P.p5k[idx]) + '/mi</b></div>';
  }
  html += '<div class="tt-row"><span>Projected Critical Speed</span><b>' + paceMSS(P.cs_med[idx]) + '/mi</b></div>';
  html += '</div>';

  // Section 2: session details (race or frontier workout). Smooth = nearest within window.
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
      html += '<div class="tt-section-title">Nearest event [' + lbl + ']</div>';
    }
    html += (isSnap && pointHtml ? pointHtml : run.html);
    html += '</div>';
  }
  return html;
}
"""

    out_html = args.out or str(OUTPUT_DIR / f'cs_timeline{suffix}.html')
    render_plot(
        fig, out_html,
        title_slug=f'cs_timeline{suffix}',
        page_title='5K fitness',
        title='5K fitness over time',
        subtitle='Posterior-median 5K fitness from a Bayesian latent-process model, compared to actual race performance',
        cursor_tooltip=CursorTooltip(
            payload=payload,
            build_js=build_js,
            first_day=first_day,
            last_day=last_day,
        ),
        # Race diamonds here are not PR-ringed (only the Races plot rings them);
        # size the gutter for the bare size-8 diamond and its 0.3px outline.
        axis_pad=[axis_pad_entry(x_lo, x_hi, marker_half_px(8, line_width=0.3))],
    )
    print(f"Wrote {out_html}")


if __name__ == '__main__':
    main()
