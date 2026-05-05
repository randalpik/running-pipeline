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
from src.shared.cs_projection import load_cs_outputs, project_races_to_5k_pace
from src.plotting import (render_plot, CursorTooltip, apply_default_layout,
                            sec_to_mss, sec_to_mss_full, SURFACES, rgba, GRID,
                            yearly_x_axis_kwargs)


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
        excl_df = pd.read_csv(excl_path, parse_dates=['date'])
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
    # The CS plot truncates to >= 2013-06-01; pre-2013 has very few datapoints
    # and huge uncertainty. project_races_to_5k_pace handles XC pre-correction
    # (matches the fit) and the β_long un-bias for distances > d_thresh internally;
    # see cs_projection.py for the rationale and formulas.
    summary_plot = (daily_summary[daily_summary['date'] >= '2013-06-01']
                    .copy().reset_index(drop=True))
    elig_plot = elig[elig['date'] >= '2013-06-01'].copy()
    print(f'Plot window: {len(summary_plot)} daily points')

    elig_plot = project_races_to_5k_pace(
        elig_plot, summary_plot, beta_long_med, d_thresh_long,
        apply_xc_correction=True, xc_correction=xc_correction)
    elig_plot = elig_plot[elig_plot['pace_norm_min'].notna()].copy()
    print(f'Hyperbolic projection (5K-equiv): {len(elig_plot)} race diamonds')

    # ---------- figure ----------
    fig = go.Figure()

    # 95% ribbon
    fig.add_trace(go.Scatter(
        x=summary_plot['date'].tolist() + summary_plot['date'].tolist()[::-1],
        y=summary_plot['cs_pace_lo95'].tolist() + summary_plot['cs_pace_hi95'].tolist()[::-1],
        fill='toself', fillcolor='rgba(255,180,80,0.10)',
        line=dict(width=0), mode='lines',
        name='95% credible interval', hoverinfo='skip', showlegend=True))
    # 50% ribbon
    fig.add_trace(go.Scatter(
        x=summary_plot['date'].tolist() + summary_plot['date'].tolist()[::-1],
        y=summary_plot['cs_pace_lo50'].tolist() + summary_plot['cs_pace_hi50'].tolist()[::-1],
        fill='toself', fillcolor='rgba(255,180,80,0.25)',
        line=dict(width=0), mode='lines',
        name='50% credible interval', hoverinfo='skip', showlegend=True))
    # Posterior median CS
    fig.add_trace(go.Scatter(
        x=summary_plot['date'], y=summary_plot['cs_pace_med'],
        mode='lines', line=dict(color='rgb(255,180,80)', width=2.5),
        name='Posterior median CS', hoverinfo='skip', showlegend=True))
    # Race diamonds (bias-corrected). Colors derived from canonical SURFACES
    # hex tokens with 0.7 alpha so a re-skin only edits one place. Each
    # diamond carries a per-race snap-mode tooltip via customdata, and the
    # trace is tagged snap_eligible so the smart-spikeline scaffold treats
    # it as a snap target.
    surf_colors = {k: rgba(SURFACES[k], 0.7) for k in ('Track', 'Road', 'XC')}

    def _race_inner(row):
        ev = row.get('event') or '(no event)'
        if pd.isna(ev): ev = '(no event)'
        t_orig = row.get('time_sec_original', row['time_sec'])
        p_orig = row.get('pace_sec_per_mi_original',
                         row.get('pace_sec_per_mi'))
        pace_raw = (sec_to_mss(p_orig)
                    if p_orig is not None and not pd.isna(p_orig) else '')
        is_xc = str(row.get('surface', '')).upper() == 'XC'
        is_5k = abs(float(row['distance_m']) - 5000.0) < 1.0

        equiv_pace_sec = float(row['pace_norm_min']) * 60
        equiv_time_sec = equiv_pace_sec * 5000.0 / 1609.344
        if is_5k and not is_xc:
            equiv_line = ''
        else:
            xc_color = SURFACES['XC']
            if is_xc and is_5k:
                label = f'<span style="color:{xc_color}">XC-corrected</span>'
            elif is_xc:
                label = f'5K-equiv <span style="color:{xc_color}">(XC-corrected)</span>'
            else:
                label = '5K-equiv'
            equiv_line = (f"<div>{label}: <b>{sec_to_mss(equiv_time_sec)}</b> "
                          f"<span class='tt-mute'>"
                          f"({sec_to_mss(equiv_pace_sec)}/mi)</span></div>")
        return (f"<div>{ev} <span class='tt-mute'>({row['surface']})</span></div>"
                f"<div>{int(row['distance_m'])}m in "
                f"<b>{sec_to_mss_full(t_orig)}</b> "
                f"<span class='tt-mute'>({pace_raw}/mi)</span></div>"
                f"{equiv_line}")

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
            legendgrouptitle_text='Race pace (5K-equiv)',
            meta={'snap_eligible': True}))

    # ---------- layout ----------
    y_min, y_max = 4.50, 8.00
    ytick_vals = [4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0]
    ytick_txt = [sec_to_mss(v * 60) for v in ytick_vals]

    apply_default_layout(
        fig,
        margin=dict(t=20, l=70, r=200, b=60),
        legend=dict(yanchor='top', y=0.99, xanchor='left', x=1.02, groupclick='toggleitem'),
        xaxis=yearly_x_axis_kwargs(
            pd.Timestamp('2013-06-01'),
            summary_plot['date'].iloc[-1],
            title='Date',
        ),
        yaxis=dict(title='Pace (min/mi)', range=[y_max, y_min],
                   tickmode='array', tickvals=ytick_vals, ticktext=ytick_txt,
                   showgrid=True, gridcolor=GRID))

    # ---------- spikeline tooltip payload ----------
    # Per-day arrays for the trend section (CS median + 50/95% intervals)
    # plus a sorted race-session list. Smooth mode shows the trend section
    # + nearest race within ±60d; snap mode shows the trend section + the
    # snapped race's details (no date, no "Nearest race" header).
    epoch = pd.Timestamp('1970-01-01')

    def _round_pace(v):
        return round(float(v) * 60) if pd.notna(v) else None

    cs_pace_med  = [_round_pace(v) for v in summary_plot['cs_pace_med']]
    cs_pace_lo50 = [_round_pace(v) for v in summary_plot['cs_pace_lo50']]
    cs_pace_hi50 = [_round_pace(v) for v in summary_plot['cs_pace_hi50']]
    cs_pace_lo95 = [_round_pace(v) for v in summary_plot['cs_pace_lo95']]
    cs_pace_hi95 = [_round_pace(v) for v in summary_plot['cs_pace_hi95']]

    sessions = []
    for _, r in elig_plot.iterrows():
        sessions.append({'day':  int((r['date'] - epoch).days),
                         'html': _race_inner(r)})
    sessions.sort(key=lambda s: s['day'])

    first_day = int((summary_plot['date'].iloc[0]  - epoch).days)
    last_day  = int((summary_plot['date'].iloc[-1] - epoch).days)

    payload = {
        'first_day': first_day,
        'cs_med':    cs_pace_med,
        'cs_lo50':   cs_pace_lo50,
        'cs_hi50':   cs_pace_hi50,
        'cs_lo95':   cs_pace_lo95,
        'cs_hi95':   cs_pace_hi95,
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

  // Section 1: per-day CS posterior summary.
  html += '<div class="tt-section">';
  html += '<div class="tt-row"><span>CS median</span><b>' + paceMSS(P.cs_med[idx]) + '/mi</b></div>';
  html += '<div class="tt-row"><span>50% interval</span>' + paceMSS(P.cs_lo50[idx]) + '–' + paceMSS(P.cs_hi50[idx]) + '/mi</div>';
  html += '<div class="tt-row"><span>95% interval</span>' + paceMSS(P.cs_lo95[idx]) + '–' + paceMSS(P.cs_hi95[idx]) + '/mi</div>';
  html += '</div>';

  // Section 2: race details. Smooth = nearest within window.
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

    out_html = args.out or str(OUTPUT_DIR / f'cs_timeline{suffix}.html')
    render_plot(
        fig, out_html,
        title_slug=f'cs_timeline{suffix}',
        page_title='CS fitness',
        title='Fitness trend as Critical Speed over time',
        subtitle='Posterior median and credible-interval ribbons from Bayesian latent-process model analysis',
        cursor_tooltip=CursorTooltip(
            payload=payload,
            build_js=build_js,
            first_day=first_day,
            last_day=last_day,
        ),
    )
    print(f"Wrote {out_html}")


if __name__ == '__main__':
    main()
