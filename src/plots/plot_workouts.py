"""
plot_workouts.py — Qualitative "every workout" plot at 5K-equivalent pace.

Shows every quality workout (intervals, tempo, repetitions, continuous
fartlek, continuous hills, hill repeats) — including sessions the TQ model
prunes — at 5K-equivalent pace, with a gold CS reference curve.

Mirrors the Races/Fitness relationship: this plot is to the Training plot
what the Races plot is to the Fitness (CS-timeline) plot.

Hill repeats lack quality data needed for a CS+D' projection, so they're
positioned at the persisted TQ smoother track on each session date
(data/training_quality_track.csv). Their elevation gained per session is
shown in hover: rep_time × rep_count × elev_per_min from the hills snapshot.

Repetitions are projected via the same hyperbolic CS+D' formula as
intervals.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.shared.paths import DATA_DIR, OUTPUT_DIR
from src.shared.workouts import (
    load_cs,
    project_workouts, project_hill_continuous, project_hill_reps,
)
from src.plotting import (render_plot, CursorTooltip, apply_default_layout,
                            sec_to_mss, fmt_min, CAT_COLORS, GRID, CS_LINE)


OUTPUT_DIR.mkdir(exist_ok=True)
OUT_HTML    = str(OUTPUT_DIR / 'workouts.html')
TRACK_CSV   = DATA_DIR / 'training_quality_track.csv'
OFFSETS_CSV = DATA_DIR / 'training_quality_offsets.csv'


CAT_LABEL = {
    'interval':           'Intervals',
    'tempo':              'Tempo',
    'rep':                'Repetitions',
    'continuous_fartlek': 'Cont. fartlek',
    'hill_cont':          'Cont. hills',
    'hill_rep':           'Hill repeats',
}

HILL_CONT_COLOR = CAT_COLORS['hill_lc']
HILL_REP_COLOR  = CAT_COLORS['hill_rep']


def _route_paren(display_name, city_state):
    parts = [str(x).strip() for x in (display_name, city_state)
             if pd.notna(x) and str(x).strip()]
    return f' ({", ".join(parts)})' if parts else ''


def _y_safe(arr):
    return [None if (v is None or (isinstance(v, float) and np.isnan(v)))
            else float(v) for v in arr]


def workout_hover(r):
    cat = r['category']
    title = f"<b>{CAT_LABEL.get(cat, cat)}</b>"
    title += _route_paren(r.get('display_name'), r.get('city_state'))
    xc_note = ' <i style="color:#9cf">[XC-corrected -6%]</i>' if r.get('xc_corrected') else ''
    rep_count = int(r['rep_count'])
    rep_dist = int(r['rep_dist'])
    if cat == 'continuous_fartlek' and rep_count == 1:
        body = f"{rep_dist}m @ {sec_to_mss(r['pace_per_mile'])}/mi"
    else:
        body = (f"{rep_count} × {rep_dist}m @ "
                f"{sec_to_mss(r['pace_per_mile'])}/mi")
    if pd.notna(r.get('rest_per_mile')) and r['rest_per_mile'] > 0:
        body += f", rest {sec_to_mss(r['rest_per_mile'])}/mi"

    p5k_disp = r.get('p5k_display_min', r['p5k_min'])
    p5k_line = (f"<b>5K-equiv:</b> {fmt_min(p5k_disp)}/mi   "
                f"<b>CS 5K:</b> {fmt_min(r['p5k_cs_min'])}/mi")
    temp_line = f"<b>Temp:</b> {r['temp_c']:.0f}°C"
    parts = [f"{title}{xc_note}", body, temp_line, p5k_line]
    return "<br>".join(parts)


def hill_cont_hover(r):
    title = (f"<b>Continuous hills</b>"
             f"{_route_paren(r.get('loop_display_name'), r.get('loop_city_state'))}")
    nreps = int(r['nreps'])
    loops_word = 'loop' if nreps == 1 else 'loops'
    ft_gained = int(round(float(r.get('ft_gained') or 0)))
    parts = [
        title,
        f"{nreps} {loops_word}, {int(r['session_min'])} min total"
        + (f", {ft_gained} ft gained" if ft_gained else ''),
        f"<b>Temp:</b> {r['temp_c']:.0f}°C",
        f"<b>Actual pace:</b> {sec_to_mss(r['actual_pace_s'])}/mi",
        f"<b>5K-equiv:</b> {fmt_min(r['p5k_min_elev_corr'])}/mi   "
        f"<b>CS 5K:</b> {fmt_min(r['p5k_cs_min'])}/mi",
    ]
    return "<br>".join(parts)


def hill_rep_hover(r):
    title = (f"<b>{CAT_LABEL['hill_rep']}</b>"
             f"{_route_paren(r.get('loop_display_name'), r.get('loop_city_state'))}")
    rep_count = int(r['rep_count'])
    reps_word = 'rep' if rep_count == 1 else 'reps'
    rt = float(r['rep_time_min'])
    if rt == int(rt):
        rt_str = f"{int(rt)} min"
    else:
        mm = int(rt)
        ss = int(round((rt - mm) * 60))
        rt_str = f"{mm}:{ss:02d}"
    body = f"{rep_count} {reps_word} × {rt_str}"
    elev = r.get('total_elev_ft')
    if pd.notna(elev):
        body += f", {int(round(float(elev)))} ft gained"
    temp_line = f"<b>Temp:</b> {r['temp_c']:.0f}°C"
    parts = [title, body, temp_line]
    return "<br>".join(parts)


def main():
    cs, epoch = load_cs()

    workouts = project_workouts(cs, epoch)
    hills_c  = project_hill_continuous(cs, epoch)
    hills_r  = project_hill_reps()

    # Load TQ's final per-category offsets so each category's markers sit at
    # the same per-category baseline TQ uses. Categories not in the CSV (rep,
    # hc_loop_other, etc.) get 0 offset.
    if not OFFSETS_CSV.exists():
        raise SystemExit(f'Missing {OFFSETS_CSV} — run plot_training_quality.py first.')
    offsets_df = pd.read_csv(OFFSETS_CSV)
    cat_offset = dict(zip(offsets_df['category'], offsets_df['offset_sec_per_mi']))

    def _apply_offset(df, offset_col='category'):
        off_sec = df[offset_col].map(cat_offset).fillna(0.0)
        return df['p5k_min'] - off_sec / 60.0

    workouts['p5k_display_min'] = _apply_offset(workouts)
    # Reps are excluded from the TQ model so cat_offset has no entry; bolt
    # the manual rep-anaerobic offset on top.
    is_rep = workouts['category'] == 'rep'
    workouts.loc[is_rep, 'p5k_display_min'] = workouts.loc[is_rep, 'p5k_display_min']

    # Hills use the route-agnostic elevation-corrected projection from the
    # shared module (HILL_ELEV_COST_SEC_PER_FT × ft_climbed). NO per-loop
    # offset is applied — those are TQ-only. Same formula for every loop,
    # parameterized by amount climbed.
    hills_c['p5k_display_min'] = hills_c['p5k_min_elev_corr']

    # Position hill_rep sessions at the TQ smoother track on their date.
    if not TRACK_CSV.exists():
        raise SystemExit(f'Missing {TRACK_CSV} — run plot_training_quality.py first.')
    track = pd.read_csv(TRACK_CSV, parse_dates=['date'])
    track_days = (track['date'] - epoch).dt.days.astype(float).values
    track_vals = track['p5k_track_min'].values

    if not hills_r.empty:
        hr_days = (hills_r['date'] - epoch).dt.days.astype(float).values
        hr_track = np.interp(hr_days, track_days, track_vals,
                              left=np.nan, right=np.nan)
        # Force NaN where the underlying track value at the bracketing day is
        # NaN (the labrum gap). np.interp would otherwise interpolate across.
        for i, d in enumerate(hr_days):
            j = int(np.searchsorted(track_days, d))
            if j == 0 or j >= len(track_days):
                hr_track[i] = np.nan
                continue
            if np.isnan(track_vals[j-1]) or np.isnan(track_vals[j]):
                hr_track[i] = np.nan
        hills_r = hills_r.copy()
        hills_r['p5k_track_min'] = hr_track

    # ---------- figure ----------
    fig = go.Figure()

    # Gold CS-implied 5K reference curve (same color/style as other tabs).
    cs_plot = cs[cs['date'] >= pd.Timestamp('2016-01-01')]
    fig.add_trace(go.Scatter(
        x=cs_plot['date'], y=_y_safe(cs_plot['p5k_implied_min'].values),
        mode='lines', name='CS-implied 5K',
        line=dict(color=CS_LINE, width=2),
        hoverinfo='skip',
    ))

    # Workouts: one trace per category. Identical styling regardless of
    # excluded_reason — pruned sessions are visually indistinguishable from
    # in-scope sessions.
    for cat in ['interval', 'tempo', 'rep', 'continuous_fartlek']:
        sub = workouts[workouts['category'] == cat]
        if sub.empty:
            continue
        cd = [workout_hover(r) for _, r in sub.iterrows()]
        fig.add_trace(go.Scatter(
            x=sub['date'], y=_y_safe(sub['p5k_display_min'].values),
            mode='markers',
            name=f'{CAT_LABEL[cat]} (n={len(sub)})',
            marker=dict(color=CAT_COLORS[cat], size=7,
                        line=dict(color='rgba(255,255,255,0.4)', width=0.5),
                        opacity=0.85),
            customdata=cd,
            hoverinfo='skip',
            legendgroup='workouts', legendgrouptitle_text='Workouts',
            meta={'snap_eligible': True},
        ))

    # Continuous hills (one trace, all loops together).
    if len(hills_c):
        cd = [hill_cont_hover(r) for _, r in hills_c.iterrows()]
        fig.add_trace(go.Scatter(
            x=hills_c['date'], y=_y_safe(hills_c['p5k_display_min'].values),
            mode='markers',
            name=f"{CAT_LABEL['hill_cont']} (n={len(hills_c)})",
            marker=dict(color=HILL_CONT_COLOR, size=7,
                        line=dict(color='rgba(255,255,255,0.4)', width=0.5),
                        opacity=0.85),
            customdata=cd,
            hoverinfo='skip',
            legendgroup='hills', legendgrouptitle_text='Hills',
            meta={'snap_eligible': True},
        ))

    # Hill repeats: positioned at TQ smoother track on each session date.
    if not hills_r.empty:
        plottable = hills_r.dropna(subset=['p5k_track_min'])
        if len(plottable):
            cd = [hill_rep_hover(r) for _, r in plottable.iterrows()]
            fig.add_trace(go.Scatter(
                x=plottable['date'], y=_y_safe(plottable['p5k_track_min'].values),
                mode='markers',
                name=f"{CAT_LABEL['hill_rep']} (n={len(plottable)})",
                marker=dict(color=HILL_REP_COLOR, size=7,
                            line=dict(color='rgba(255,255,255,0.4)', width=0.5),
                            opacity=0.85),
                customdata=cd,
                hoverinfo='skip',
                legendgroup='hills',
                meta={'snap_eligible': True},
            ))

    # ---------- layout ----------
    # 5K-equivalent pace axis range derived from the actual data so the
    # slowest workouts aren't clipped. Includes CS line, all marker positions,
    # and the TQ track. 10 sec/mi padding on each side; ticks every 10 sec/mi.
    y_candidates = []
    y_candidates.extend(cs_plot['p5k_implied_min'].dropna().tolist())
    y_candidates.extend(workouts['p5k_display_min'].dropna().tolist())
    if len(hills_c):
        y_candidates.extend(hills_c['p5k_display_min'].dropna().tolist())
    if not hills_r.empty:
        y_candidates.extend(hills_r['p5k_track_min'].dropna().tolist())
    pad_min = 10.0 / 60  # 10 sec/mi padding
    y_min = float(np.floor((min(y_candidates) - pad_min) * 6.0)) / 6.0  # round down to 10 sec
    y_max = float(np.ceil((max(y_candidates) + pad_min) * 6.0)) / 6.0   # round up to 10 sec
    tickvals = [y_min + i * (10/60) for i in range(int(round((y_max - y_min) * 6)) + 1)]
    ticktext = [sec_to_mss(v * 60) for v in tickvals]

    apply_default_layout(
        fig,
        margin=dict(t=20, l=70, r=200, b=60),
        hovermode=False,
        legend=dict(yanchor='top', y=0.99, xanchor='left', x=1.02,
                    groupclick='toggleitem', font=dict(size=11)),
        xaxis=dict(title='Date', showgrid=True, gridcolor=GRID,
                   tick0='2016-01-01', dtick='M12',
                   range=[pd.Timestamp('2016-01-01'),
                          pd.Timestamp(workouts['date'].max()) + pd.Timedelta(days=30)]),
        yaxis=dict(title='5K-equivalent pace (min/mi)',
                   range=[y_max, y_min],
                   tickmode='array', tickvals=tickvals, ticktext=ticktext,
                   showgrid=True, gridcolor=GRID, zeroline=False),
    )

    # ---------- cursor-tooltip payload ----------
    # Smooth mode shows date + CS pace; snap mode shows the session details.
    js_epoch = pd.Timestamp('1970-01-01')
    plot_start = pd.Timestamp('2016-01-01')
    plot_end   = pd.Timestamp(workouts['date'].max()) + pd.Timedelta(days=30)
    all_days   = pd.date_range(plot_start, plot_end, freq='D')

    days_2016 = (all_days - epoch).days.astype(float).values
    cs_pace_per_day = np.interp(days_2016, cs['day'].values, cs['p5k_implied_min'].values)

    sessions = []
    for _, r in workouts.iterrows():
        sessions.append({'day': int((r['date'] - js_epoch).days),
                         'html': workout_hover(r)})
    for _, r in hills_c.iterrows():
        sessions.append({'day': int((r['date'] - js_epoch).days),
                         'html': hill_cont_hover(r)})
    if not hills_r.empty:
        for _, r in hills_r.dropna(subset=['p5k_track_min']).iterrows():
            sessions.append({'day': int((r['date'] - js_epoch).days),
                             'html': hill_rep_hover(r)})
    sessions.sort(key=lambda s: s['day'])

    first_day = int((all_days[0]  - js_epoch).days)
    last_day  = int((all_days[-1] - js_epoch).days)

    payload = {
        'first_day': first_day,
        'cs_pace':   [round(float(v), 4) for v in cs_pace_per_day],
        'sessions':  sessions,
        'nearest_window_days': 60,
    }

    build_js = r"""
function buildTooltip(day, isSnap, pointHtml) {
  var P = window.__TT_DATA;
  var idx = day - P.first_day;
  if (idx < 0 || idx >= P.cs_pace.length) return '';

  var DOW = ['Sun','Mon','Tue','Wed','Thu','Fri','Sat'];
  function fmtMin(v) {
    if (v === null || v === undefined || isNaN(v)) return '—';
    var s = Math.round(v * 60);
    var m = Math.floor(s / 60), r = s % 60;
    return m + ':' + (r < 10 ? '0' : '') + r;
  }
  function dateLabel(d) {
    var dt = new Date(d * 86400000);
    var y = dt.getUTCFullYear();
    var m = String(dt.getUTCMonth() + 1).padStart(2, '0');
    var dd = String(dt.getUTCDate()).padStart(2, '0');
    return y + '-' + m + '-' + dd + ' (' + DOW[dt.getUTCDay()] + ')';
  }

  var html = '';
  html += '<div class="tt-date">' + dateLabel(day) + '</div>';
  html += '<div class="tt-section">';
  html += '<div class="tt-row"><span>CS 5K pace</span><b>' + fmtMin(P.cs_pace[idx]) + '/mi</b></div>';
  html += '</div>';

  // Session section: snap shows the marker's own html; smooth shows the
  // nearest session within the window.
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
      html += '<div class="tt-section-title">Nearest session [' + lbl + ']</div>';
    }
    html += (isSnap && pointHtml ? pointHtml : run.html);
    html += '</div>';
  }
  return html;
}
"""

    render_plot(
        fig, OUT_HTML,
        title_slug='workouts',
        page_title='Workouts',
        title='All workouts at 5K-equivalent pace',
        subtitle='Corrections applied per category',
        cursor_tooltip=CursorTooltip(
            payload=payload,
            build_js=build_js,
            first_day=first_day,
            last_day=last_day,
        ),
    )
    print(f'Wrote {OUT_HTML}  '
          f'({len(workouts)} workouts, {len(hills_c)} cont. hills, '
          f'{len(hills_r)} hill reps)')


if __name__ == '__main__':
    main()
