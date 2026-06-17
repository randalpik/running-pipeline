"""
plot_workouts.py — Qualitative "every workout" plot at 5K-equivalent pace.

Shows every quality workout (intervals, tempo, repetitions, continuous
fartlek, continuous hills, hill repeats) — including sessions the TQ model
prunes — at 5K-equivalent pace, with a gold CS reference curve.

Mirrors the Races/Fitness relationship: this plot is to the Training plot
what the Races plot is to the Fitness (CS-timeline) plot.

Hill repeats lack quality data needed for a CS+D' projection, so they're
positioned at the persisted TQ smoother track on each session date
(data/training_quality_track.csv). Watch-era sessions show real per-rep
distance + elevation and an average grade in hover (workout_measured.csv,
hillrep-exact); pre-watch sessions fall back to the parser estimate
(rep_time × rep_count × elev_per_min from the hills snapshot).

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
from src.shared.units import FT_PER_M
from src.shared.plot_window import daily_floor, clip_to_daily_floor
from src.shared.workouts import (
    load_cs, watch_log_demotions,
    project_workouts, project_hill_continuous, project_hill_reps,
)
from src.shared.cs_projection import load_cs_outputs
from src.shared.performance_frontier import standard_demos, build_frontier
from src.plotting import (render_plot, CursorTooltip, apply_default_layout,
                            sec_to_mss, fmt_min, route_paren, CAT_COLORS, GRID,
                            CS_LINE, FRONTIER_LINE, TAG_COLORS,
                            yearly_x_axis_kwargs, nice_time_ticks)


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_HTML    = str(OUTPUT_DIR / 'workouts.html')
TRACK_CSV   = DATA_DIR / 'training_quality_track.csv'
HILL_MODEL_CSV = DATA_DIR / 'hill_model.csv'


CAT_LABEL = {
    'interval':           'Intervals',
    'tempo':              'Tempo',
    'rep':                'Repetitions',
    'continuous_fartlek': 'Fartlek',
    'hill_cont':          'Cont. hills',
    'hill_rep':           'Hill repeats',
}

HILL_CONT_COLOR = CAT_COLORS['hill_cont']
HILL_REP_COLOR  = CAT_COLORS['hill_rep']

# Condition-tag rings: tagged sessions get a colored halo ring (TAG_COLORS)
# drawn as a per-tag overlay trace — same construction as the race plots'
# PR ring, but sized to float OUTSIDE the 7px marker (visible gap) so it
# pops on the dark background. Each tag gets its own legend entry. Ring
# color matches the tag's tooltip text color.
# Exactly a 1px halo gap: marker outer edge = 7/2 + 0.5/2 outline = 3.75px;
# the 1px ring stroke straddles the size boundary, so inner stroke edge
# 3.75 + 1 = 4.75px → size = 2 * (4.75 + 0.5) = 10.5.
TAG_RING_SIZE  = 10.5
TAG_RING_WIDTH = 1     # the halo offset does the work; thin stroke suffices
TAG_LEGEND = {
    'uncertain accuracy': 'Uncertain accuracy',
    'snow':               'Snow',
    'outlier':            'Slow outlier',
    'xc':                 'XC-corrected',
    'enriched':           'Watch-enriched',
}


def session_tag(r):
    """TAG_COLORS key for a session, or None when untagged. Exclusion tags
    win over the XC ring — XC green marks sessions that are XC-corrected
    AND kept in Training. The white enriched ring is informational and
    lowest-priority: a non-excluded session whose watch enrichment succeeded
    (failures are visible by its ABSENCE on watch-era quality days)."""
    er = r.get('excluded_reason')
    if er == 'uncertain accuracy':
        return 'uncertain accuracy'
    if er == 'snow':
        return 'snow'
    if r.get('tq_outlier'):
        return 'outlier'
    if r.get('xc_corrected') and not isinstance(er, str):
        return 'xc'
    if not isinstance(er, str) and (
            isinstance(r.get('measured_line'), str) or r.get('watch_measured')):
        return 'enriched'
    return None


def collect_ring_points(ring_pts, df, ycol):
    """Append each tagged session's (date, y) to its tag's ring-point list."""
    for _, r in df.iterrows():
        tag = session_tag(r)
        if tag and pd.notna(r[ycol]):
            ring_pts[tag][0].append(r['date'])
            ring_pts[tag][1].append(float(r[ycol]))


def _y_safe(arr):
    return [None if (v is None or (isinstance(v, float) and np.isnan(v)))
            else float(v) for v in arr]


def workout_hover(r, single_type=False):
    cat = r['category']
    # When the dataset has a single workout type (e.g. a watch profile's
    # continuous fartleks), present it generically as "Workout".
    label = 'Workout' if single_type else CAT_LABEL.get(cat, cat)
    title = f"<b>{label}</b>"
    title += route_paren(r.get('display_name'), r.get('city_state'))
    xc_note = f' <span style="color:{TAG_COLORS["xc"]}">(XC-corrected)</span>' if r.get('xc_corrected') else ''
    rep_count = int(r['rep_count'])
    rep_dist = int(r['rep_dist'])
    # pace_per_mile is log-owned end-to-end: enriched days are normalized to
    # the logged quality pace in parse_workouts (watch never overrides it).
    pace = r['pace_per_mile']
    structure = r.get('structure')
    if isinstance(structure, str) and structure:
        # Watch-enriched: actual measured rep layout, not the effective rep.
        body = f"{structure} @ {sec_to_mss(pace)}/mi"
    elif rep_count == 1:
        # A single continuous effort (fartlek, tempo, ...) — no '1 ×'.
        body = f"{rep_dist}m @ {sec_to_mss(pace)}/mi"
    else:
        body = f"{rep_count} × {rep_dist}m @ {sec_to_mss(pace)}/mi"
    if pd.notna(r.get('rest_per_mile')) and r['rest_per_mile'] > 0:
        body += f", rest {sec_to_mss(r['rest_per_mile'])}/mi"
    # Wrap the decomposition so a long measured-rep `structure` string wraps at
    # the tooltip width instead of being clipped by the nowrap/overflow:hidden
    # tooltip (the appended Watch / Watch-adj sub-lines carry their own wrap).
    body = f'<span class="tt-wrap">{body}</span>'
    measured = r.get('measured_line')
    if isinstance(measured, str) and measured:
        body += f"<br>{measured}"
        # How much the watch reps above were scaled to match the log pace
        # (negative = GPS read short, watch times compressed to fit). Compare
        # the NORMALIZED pace actually applied (pace_per_mile) to the raw watch
        # pace — so watch-only days, which aren't normalized (the watch is the
        # source of truth), show no adjustment. Equals the logged quality pace
        # on hand-log days, so those are unchanged.
        wpr, qp = r.get('watch_pace_raw'), r.get('pace_per_mile')
        if pd.notna(wpr) and pd.notna(qp) and wpr > 0:
            pct = (qp / wpr - 1.0) * 100.0
            if abs(pct) >= 0.05:
                body += f"<br><b>Watch adj:</b> {pct:+.1f}%"

    p5k_disp = r.get('p5k_display_min', r['p5k_min'])
    p5k_line = (f"<b>5K-equiv:</b> {fmt_min(p5k_disp)}/mi   "
                f"<b>CS 5K:</b> {fmt_min(r['p5k_cs_min'])}/mi")
    temp_line = (f"<b>Temp:</b> {r['temp_c']:.0f}°C"
                 if pd.notna(r.get('temp_c')) else '')
    parts = [f"{title}{xc_note}", body, temp_line, p5k_line]
    excl = r.get('tq_excluded_line')
    if isinstance(excl, str) and excl:
        parts.append(excl)
    return "<br>".join(p for p in parts if p)


def excluded_line(note, reason=None):
    """Italic 'Excluded from Training' tooltip tag. Ringed tags (uncertain
    accuracy, snow) take their ring color; other reasons stay amber."""
    color = TAG_COLORS.get(reason, '#E8A33C')
    return f'<i style="color:{color}">Excluded from Training: {note}</i>'


HILL_EXCL_NOTE = {
    'snow':             'snow',
    'hc_rep_hybrid':    'hc/rep hybrid',
    'hc_no_covariates': 'no elevation/terrain data for loop',
}


def hill_cont_hover(r):
    title = (f"<b>Continuous hills</b>"
             f"{route_paren(r.get('loop_display_name'), r.get('loop_city_state'))}")
    nreps = int(r['nreps'])
    loops_word = 'loop' if nreps == 1 else 'loops'
    ft_gained = int(round(float(r.get('ft_gained') or 0)))
    # Watch-measured days replace the hand log's whole-minute estimate with
    # the exact moving total (the projection already uses it via t_eff).
    time_part = (f"{sec_to_mss(r['t_eff'])} total" if r.get('watch_measured')
                 else f"{int(r['session_min'])} min total")
    body = (f"{nreps} {loops_word}, {time_part}"
            + (f", {ft_gained} ft gained" if ft_gained else ''))
    measured = r.get('hill_measured_line')
    if isinstance(measured, str) and measured:
        body += f"<br>{measured}"
    temp_line = (f"<b>Temp:</b> {r['temp_c']:.0f}°C"
                 if pd.notna(r.get('temp_c')) else '')
    parts = [
        p for p in [
            title,
            body,
            temp_line,
            f"<b>Actual pace:</b> {sec_to_mss(r['actual_pace_s'])}/mi",
            f"<b>5K-equiv:</b> {fmt_min(r['p5k_display_min'])}/mi   "
            f"<b>CS 5K:</b> {fmt_min(r['p5k_cs_min'])}/mi",
        ] if p
    ]
    # Exclusion tags, same convention as flat workouts: category flags come
    # straight off excluded_reason; Training's prunes (outlier / easy
    # outlier) arrive via the persisted exclusions CSV.
    er = r.get('excluded_reason')
    if isinstance(er, str) and er:
        parts.append(excluded_line(HILL_EXCL_NOTE.get(er, er), er))
    else:
        excl = r.get('tq_excluded_line')
        if isinstance(excl, str) and excl:
            parts.append(excl)
    return "<br>".join(parts)


def hill_rep_hover(r, measured=None):
    title = (f"<b>{CAT_LABEL['hill_rep']}</b>"
             f"{route_paren(r.get('loop_display_name'), r.get('loop_city_state'))}")
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
    # Watch-measured days carry a real total elevation; only fall back to the
    # parser estimate (rep_time × rep_count × elev_per_min) when unmeasured.
    md = measured.get(str(r['date'].date())) if measured else None
    elev = r.get('total_elev_ft')
    if md:
        body += f", {int(round(md['total_gain_ft']))} ft gained"
    elif pd.notna(elev):
        body += f", {int(round(float(elev)))} ft gained"
    temp_line = (f"<b>Temp:</b> {r['temp_c']:.0f}°C"
                 if pd.notna(r.get('temp_c')) else '')
    parts = [p for p in [title, body, temp_line] if p]
    if md:
        parts.append(f"<b>Avg grade:</b> {md['avg_grade']:.1f}%")
        parts.append(md['reps_html'])
    # Hill reps never feed Training (no CS projection), so the snow tag is a
    # plain condition note rather than an exclusion line.
    if r.get('excluded_reason') == 'snow':
        parts.append(f'<i style="color:{TAG_COLORS["snow"]}">Snow</i>')
    return "<br>".join(parts)


def _rep_time(t):
    """Absolute rep time for the Watch line: 'M:SS', or raw seconds under
    100 (400@68, not 400@1:08)."""
    return f"{int(round(t))}" if t < 100 else sec_to_mss(t)


def measured_lines():
    """Per-date watch-measured rep decomposition for hover (workout_measured
    .csv, written by src/coros/reps.py). Only trusted statuses appear;
    mismatch-demoted days are skipped (their watch data was rejected — the
    hover shows the parser estimate like any non-enriched day). Reps show
    absolute time, not pace; the .tt-wrap span wraps at tooltip width."""
    path = DATA_DIR / 'workout_measured.csv'
    if not path.exists():
        return {}
    m = pd.read_csv(path)
    m = m[(m['rep_idx'] > 0) & (m['status'].isin(['exact', 'watch-only']))
          & ~m['date'].isin(watch_log_demotions())]
    lines = {}
    for date, day in m.groupby('date'):
        parts = [f"{int(r['dist_m'])}{'cf' if r['kind'] == 'cf' else ''}"
                 f"@{_rep_time(r['time_s'])}"
                 for _, r in day.iterrows()]
        lines[date] = ('<b>Watch:</b> <span class="tt-wrap">'
                       + ' · '.join(parts) + '</span>')
    return lines


def hill_measured_lines():
    """Per-date watch hover line for hill blocks (workout_measured.csv).
    The measured moving total is merged into the headline (hill_cont_hover
    shows it as 'xx:xx total'), so the Watch line only carries what the
    headline can't: per-loop splits on hill-exact days."""
    path = DATA_DIR / 'workout_measured.csv'
    if not path.exists():
        return {}
    m = pd.read_csv(path)
    m = m[(m['rep_idx'] > 0)
          & (m['status'].isin(['hill-exact', 'hill-total']))]
    lines = {}
    for date, day in m.groupby('date'):
        if (day['status'] == 'hill-exact').all() and len(day) > 1:
            splits = ' · '.join(sec_to_mss(t) for t in day['time_s'])
            lines[date] = (f'<b>Watch:</b> <span class="tt-wrap">'
                           f'loops {splits}</span>')
    return lines


def hill_rep_measured_lines():
    """Per-date watch-measured hill-rep detail (workout_measured.csv,
    hillrep-exact only). The watch gives what the log never held: per-rep
    distance + elevation, and a true average grade (total vertical / total
    horizontal). Returns {date: {total_gain_ft, avg_grade, reps_html}};
    mismatch / no-block days are absent and fall back to the parser estimate."""
    path = DATA_DIR / 'workout_measured.csv'
    if not path.exists():
        return {}
    m = pd.read_csv(path, dtype={'date': str})
    m = m[(m['rep_idx'] > 0) & (m['status'] == 'hillrep-exact')]
    lines = {}
    for date, day in m.groupby('date'):
        total_gain = float(day['gain_ft'].sum())
        total_dist = float(day['dist_m'].sum())
        # average grade over the whole block = total rise / total run; gain is
        # feet, distance metres, so rise back to metres via FT_PER_M (0.3048).
        avg_grade = (total_gain * FT_PER_M / total_dist * 100) if total_dist else 0.0
        parts = [f"{int(round(r['dist_m']))}m/+{int(round(r['gain_ft']))}ft"
                 for _, r in day.iterrows()]
        lines[date] = {
            'total_gain_ft': total_gain, 'avg_grade': avg_grade,
            'reps_html': ('<b>Watch:</b> <span class="tt-wrap">'
                          + ' · '.join(parts) + '</span>'),
        }
    return lines


OUTLIER_REASONS = {'outlier', 'easy outlier'}


def load_tq_exclusions(src):
    """Rows of training_quality_exclusions.csv (written by
    plot_training_quality.py) for one src ('workout' / 'hill')."""
    path = DATA_DIR / 'training_quality_exclusions.csv'
    cols = ['date', 'reason', 'resid', 'cutoff', 'src']
    if not path.exists():
        return pd.DataFrame(columns=cols)
    e = pd.read_csv(path)
    if 'src' not in e.columns:
        e['src'] = 'workout'
    return e[e['src'] == src]


def tq_exclusion_lines(src='workout'):
    """Per-date hover note for sessions Training excluded. Flags WHY, so the
    slow sessions that survive only on the Workouts plot are explained.
    Workout residual outliers show the residual against the prune cutoff;
    hill outliers (track prune or the hill model's easy-day prune) read
    'slow outlier' — the marker position already shows how slow."""
    lines = {}
    for _, r in load_tq_exclusions(src).iterrows():
        if r['reason'] in OUTLIER_REASONS and src == 'hill':
            note, color_key = 'slow outlier', 'outlier'
        elif r['reason'] == 'outlier' and pd.notna(r.get('resid')):
            note = (f"residual {r['resid']:+.1f} s/mi > +{r['cutoff']:.1f} cutoff")
            color_key = str(r['reason'])
        else:
            note, color_key = str(r['reason']), str(r['reason'])
        lines[str(r['date'])] = excluded_line(note, color_key)
    return lines


def main():
    cs, epoch = load_cs()

    workouts = project_workouts(cs, epoch)
    hills_c  = project_hill_continuous(cs, epoch)
    hills_r  = project_hill_reps(cs, epoch)

    lines = measured_lines()
    if lines:
        workouts['measured_line'] = (
            workouts['date'].dt.date.astype(str).map(lines))
        print(f'Watch decomposition on {workouts["measured_line"].notna().sum()} '
              f'workout hovers')

    hlines = hill_measured_lines()
    if hlines:
        hills_c['hill_measured_line'] = (
            hills_c['date'].dt.date.astype(str).map(hlines))
        print(f'Watch hill block on {hills_c["hill_measured_line"].notna().sum()} '
              f'hill hovers')

    hr_measured = hill_rep_measured_lines()
    if hr_measured and not hills_r.empty:
        # Mark enriched hill-rep sessions so they get the white watch-enriched
        # ring (session_tag -> 'enriched'), same as enriched workout markers.
        hills_r['watch_measured'] = (
            hills_r['date'].dt.date.astype(str).isin(hr_measured))
        print(f'Watch hill-rep detail on {len(hr_measured)} hill-rep hovers')

    excl = tq_exclusion_lines()
    if excl:
        workouts['tq_excluded_line'] = (
            workouts['date'].dt.date.astype(str).map(excl))
        print(f'Training-exclusion note on {workouts["tq_excluded_line"].notna().sum()} '
              f'workout hovers')

    h_excl = tq_exclusion_lines('hill')
    if h_excl and len(hills_c):
        hills_c['tq_excluded_line'] = (
            hills_c['date'].dt.date.astype(str).map(h_excl))
        print(f'Training-exclusion note on {hills_c["tq_excluded_line"].notna().sum()} '
              f'hill hovers')
    # Slow-outlier ring for sessions Training pruned as outliers — both
    # workouts (track-relative prune) and hills (track prune + the hill
    # model's easy-day prune). Workouts were missing this ring until June
    # 2026 (only the hover note flagged them).
    we = load_tq_exclusions('workout')
    w_out = set(we.loc[we['reason'].isin(OUTLIER_REASONS), 'date'].astype(str))
    workouts['tq_outlier'] = (
        workouts['date'].dt.strftime('%Y-%m-%d').isin(w_out))
    if len(hills_c):
        he = load_tq_exclusions('hill')
        out_dates = set(he.loc[he['reason'].isin(OUTLIER_REASONS), 'date']
                        .astype(str))
        hills_c['tq_outlier'] = (
            hills_c['date'].dt.strftime('%Y-%m-%d').isin(out_dates))

    # One shared CS predictor, no per-category offsets (June 2026): every
    # workout displays its raw 5K-equivalent projection — intent and era
    # effort policy stay visible instead of being subtracted by label.
    workouts['p5k_display_min'] = workouts['p5k_min']

    # Hills display: pinned Minetti net gain cost (p5k_min_hillcorr, from
    # the shared projection) minus the fitted trail term (hill_model.csv,
    # written by plot_training_quality). No centering — the hill-class
    # effort gap stays visible, exactly like tempos display their raw
    # sub-max level.
    hills_c['p5k_display_min'] = hills_c['p5k_min_hillcorr'].fillna(
        hills_c['p5k_min'])
    if HILL_MODEL_CSV.exists() and len(hills_c):
        hm = pd.read_csv(HILL_MODEL_CSV).set_index('term')['coef']
        trail_sec = hm.get('is_trail', 0.0) * hills_c['is_trail'].fillna(0.0)
        hills_c['p5k_display_min'] = hills_c['p5k_display_min'] - trail_sec / 60.0

    # Position hill_rep sessions at the TQ smoother track on their date.
    if not TRACK_CSV.exists():
        raise SystemExit(f'Missing {TRACK_CSV} — run plot_training_quality.py first.')
    track = pd.read_csv(TRACK_CSV, parse_dates=['date'])
    track_days = (track['date'] - epoch).dt.days.astype(float).to_numpy()
    track_vals = track['p5k_track_min'].to_numpy()

    if not hills_r.empty:
        hr_days = (hills_r['date'] - epoch).dt.days.astype(float).to_numpy()
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
        # Watch-era reps sit at their real projected p5k (project_hill_reps);
        # pre-watch reps stay at the TQ smoother track (display-only).
        wm = hills_r.get('watch_measured', pd.Series(False, index=hills_r.index)).fillna(False)
        hills_r['p5k_plot'] = np.where(wm & hills_r['p5k_min'].notna(),
                                       hills_r['p5k_min'], hills_r['p5k_track_min'])

    # ---------- figure ----------
    fig = go.Figure()

    # Gold CS-implied 5K reference curve (same color/style as other tabs).
    cs_plot = clip_to_daily_floor(cs)
    fig.add_trace(go.Scatter(
        x=cs_plot['date'], y=_y_safe(cs_plot['p5k_implied_min'].values),
        mode='lines', name='CS-implied 5K',
        line=dict(color=CS_LINE, width=2),
        hoverinfo='skip',
    ))

    # Performance frontier (red): demonstrated 5K capability, same canonical
    # construction as Fitness (src/shared/performance_frontier.py; corpus
    # artifact written by plot_training_quality, which runs earlier).
    daily_summary, _beta_long, _d_thresh, _xc = load_cs_outputs(str(DATA_DIR))
    front_plot = clip_to_daily_floor(daily_summary).copy()
    front_demos = standard_demos(daily_summary, _beta_long, _d_thresh, _xc)
    frontier, _ = build_frontier(front_demos, pd.DatetimeIndex(front_plot['date']),
                                 front_plot['p5k_implied_min'])
    fig.add_trace(go.Scatter(
        x=front_plot['date'], y=_y_safe(frontier['frontier_pace_min'].values),
        mode='lines', name='Performance frontier (5K)',
        line=dict(color=FRONTIER_LINE, width=2),
        hoverinfo='skip',
    ))

    # Workouts: one trace per category. Condition tags (uncertain accuracy /
    # snow / XC-corrected-and-kept) are drawn as halo-ring overlay traces
    # after the marker traces — ring points are collected here per trace so
    # they exactly track what's plotted.
    #
    # Single-type collapse: when the dataset has exactly one workout/hill
    # category (the watch continuous-fartlek case), present it generically as
    # one "Workout" legend line with no group header. The underlying category
    # (and its CS analysis) is unchanged — only the display label.
    present_cats = [c for c in ['interval', 'tempo', 'rep', 'continuous_fartlek']
                    if not workouts[workouts['category'] == c].empty]
    n_legend = (len(present_cats)
                + (1 if len(hills_c) else 0)
                + (1 if (not hills_r.empty
                         and len(hills_r.dropna(subset=['p5k_plot']))) else 0))
    single_type = n_legend == 1
    ring_pts = {tag: ([], []) for tag in TAG_LEGEND}
    for cat in ['interval', 'tempo', 'rep', 'continuous_fartlek']:
        sub = workouts[workouts['category'] == cat]
        if sub.empty:
            continue
        cd = [workout_hover(r, single_type) for _, r in sub.iterrows()]
        collect_ring_points(ring_pts, sub, 'p5k_display_min')
        fig.add_trace(go.Scatter(
            x=sub['date'], y=_y_safe(sub['p5k_display_min'].values),
            mode='markers',
            name=(f'Workout (n={len(sub)})' if single_type
                  else f'{CAT_LABEL[cat]} (n={len(sub)})'),
            marker=dict(color=CAT_COLORS[cat], size=7,
                        line=dict(color='rgba(255,255,255,0.4)', width=0.5),
                        opacity=0.85),
            customdata=cd,
            hoverinfo='skip',
            legendgroup='workouts',
            legendgrouptitle_text='' if single_type else 'Workouts',
            meta={'snap_eligible': True},
        ))

    # Continuous hills (one trace, all loops together).
    if len(hills_c):
        cd = [hill_cont_hover(r) for _, r in hills_c.iterrows()]
        collect_ring_points(ring_pts, hills_c, 'p5k_display_min')
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

    # Hill repeats: watch-era reps at their real projected p5k, pre-watch reps
    # at the TQ smoother track (display-only).
    if not hills_r.empty:
        plottable = hills_r.dropna(subset=['p5k_plot'])
        if len(plottable):
            cd = [hill_rep_hover(r, hr_measured) for _, r in plottable.iterrows()]
            collect_ring_points(ring_pts, plottable, 'p5k_plot')
            fig.add_trace(go.Scatter(
                x=plottable['date'], y=_y_safe(plottable['p5k_plot'].values),
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

    # Condition-tag halo rings, one overlay trace (and legend entry) per tag.
    # Transparent fill so the underlying marker stays hover-snappable; drawn
    # after the marker traces so the rings sit on top.
    for tag, label in TAG_LEGEND.items():
        xs, ys = ring_pts[tag]
        if not xs:
            continue
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode='markers',
            name=f'{label} (n={len(xs)})',
            marker=dict(symbol='circle', size=TAG_RING_SIZE,
                        color='rgba(0,0,0,0)',
                        line=dict(width=TAG_RING_WIDTH, color=TAG_COLORS[tag])),
            hoverinfo='skip',
            legendgroup='tags', legendgrouptitle_text='Tags',
        ))

    # ---------- layout ----------
    # 5K-equivalent pace axis range derived from the actual data so the
    # slowest workouts aren't clipped. Includes CS line, all marker positions,
    # and the TQ track. target=14 reproduces the former 10 s/mi spacing over
    # Max's ~2:15 span and adapts the interval to any profile's range.
    y_candidates = []
    y_candidates.extend(cs_plot['p5k_implied_min'].dropna().tolist())
    y_candidates.extend(workouts['p5k_display_min'].dropna().tolist())
    if len(hills_c):
        y_candidates.extend(hills_c['p5k_display_min'].dropna().tolist())
    if not hills_r.empty:
        y_candidates.extend(hills_r['p5k_plot'].dropna().tolist())
    _ticks_sec, ticktext = nice_time_ticks(
        min(y_candidates) * 60, max(y_candidates) * 60, target=14)
    tickvals = [t / 60.0 for t in _ticks_sec]
    y_min, y_max = tickvals[0], tickvals[-1]

    apply_default_layout(
        fig,
        margin=dict(t=20, l=70, r=200, b=28),
        hovermode=False,
        legend=dict(yanchor='top', y=0.99, xanchor='left', x=1.02,
                    groupclick='toggleitem', font=dict(size=11)),
        xaxis=yearly_x_axis_kwargs(
            daily_floor(),
            pd.Timestamp(workouts['date'].max()) + pd.Timedelta(days=30),
        ),
        yaxis=dict(title='5K-equivalent pace (min/mi)',
                   range=[y_max, y_min],
                   tickmode='array', tickvals=tickvals, ticktext=ticktext,
                   showgrid=True, gridcolor=GRID, zeroline=False),
    )

    # ---------- cursor-tooltip payload ----------
    # Smooth mode shows date + CS pace; snap mode shows the session details.
    js_epoch = pd.Timestamp('1970-01-01')
    plot_start = daily_floor()
    plot_end   = pd.Timestamp(workouts['date'].max()) + pd.Timedelta(days=30)
    all_days   = pd.date_range(plot_start, plot_end, freq='D')

    days_2016 = (all_days - epoch).days.astype(float).to_numpy()
    cs_pace_per_day = np.interp(days_2016, cs['day'].to_numpy(), cs['p5k_implied_min'].to_numpy())

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
                             'html': hill_rep_hover(r, hr_measured)})
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
