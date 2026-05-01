"""
plot_training_quality.py — Interactive Plotly version of the training-quality
visualization.

Shows every workout (from workout_decomposed.csv) and qualifying long run
plotted at CS_implied + corrected residual, with the adaptive-Gaussian smoother
track laid on top of the CS-implied 5K curve.

Each point's hover shows the original log string plus all derived fields,
intended for pruning / outlier evaluation.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.shared.paths import DATA_DIR, OUTPUT_DIR
from src.plotting import (render_plot, CursorTooltip, apply_default_layout,
                            title_block, TITLE_MARGIN_TOP,
                            sec_to_mss, fmt_min, CAT_COLORS, GRID,
                            GAP_BREAK_DAYS, adaptive_gauss_smoother)


# ---------- paths ----------
WORKOUTS_PATH = DATA_DIR / 'workout_decomposed.csv'
DAILY_PATH    = DATA_DIR / 'daily.csv'
CS_PATH       = DATA_DIR / 'bayes_cs_summary.csv'
OUTPUT_DIR.mkdir(exist_ok=True)
OUT_HTML = str(OUTPUT_DIR / 'training_quality.html')

for _required in (WORKOUTS_PATH, DAILY_PATH, CS_PATH):
    if not _required.exists():
        raise SystemExit(f'Could not find {_required}')

# ---------- pipeline parameters (match training_unified_pipeline.py) ----------
TAU = 210.0
LR_MILES_MIN = 20         # long run filter, low end (no upper bound)
LR_BIN_SPLIT = 23         # split point for two-bin long run classification

# ---------- smoother parameters (match training_quality_track.py) ----------
GAUSS_BASE_BW_DAYS = 30
GAUSS_TARGET_ESS   = 12
GAUSS_MAX_BW_DAYS  = 400
GRID_FREQ          = '7D'

# ---------- visual config ----------
CAT_LABEL = {
    'interval': 'Interval', 'tempo': 'Tempo', 'rep': 'Rep',
    'continuous_fartlek': 'Cont. fartlek',
    'lr_20-22.9': 'Long 20–22.9', 'lr_23+': 'Long 23+',
    'hill_lc': 'Hill (lc)', 'hill_rc': 'Hill (rc)', 'hill_pwr1': 'Hill (pwr1)',
}


# ---------- pipeline ----------
def load_cs():
    cs = pd.read_csv(CS_PATH, parse_dates=['date']).sort_values('date').reset_index(drop=True)
    cs['t5k_pred_sec'] = (5000.0 - cs['dp_med']) / cs['cs_mps_med']
    cs['p5k_implied_min'] = 1609.344 * cs['t5k_pred_sec'] / 5000.0 / 60.0
    epoch = cs['date'].min()
    cs['day'] = (cs['date'] - epoch).dt.days.astype(float)
    return cs, epoch


def add_cs(df, cs, epoch):
    df = df.copy()
    df['day'] = (df['date'] - epoch).dt.days.astype(float)
    df['p5k_cs_min'] = np.interp(df['day'], cs['day'].values, cs['p5k_implied_min'].values)
    df['dp_t']       = np.interp(df['day'], cs['day'].values, cs['dp_med'].values)
    df['year']       = df['date'].dt.year
    return df


def project_workouts(cs, epoch):
    w = pd.read_csv(WORKOUTS_PATH, parse_dates=['date'])
    # bring along workout_raw + conditions + qd for filters and hover
    daily = pd.read_csv(DAILY_PATH, parse_dates=['date'])
    w = w.merge(daily[['date', 'workout_raw', 'conditions', 'quality_distance_m']],
                on='date', how='left')

    # Rule: a tempo whose decomposition is implicit (no explicit "Nx<dist>" in
    # the log) and lands at rep_dist >= 1600 should be treated as an interval.
    # Explicit-Nx tempos like "4x1600t" stay as tempos. Catches 2024-05-04 only.
    import re
    has_nx = w['workout_raw'].astype(str).str.contains(r'\d+x\d+', regex=True, na=False)
    mask = (w['type'] == 'tempo') & (w['rep_dist'] >= 1600) & (~has_nx)
    w.loc[mask, 'type'] = 'interval'

    # Reps excluded: anaerobic top-end fitness, noisy contributor.
    w = w[w['type'] != 'rep'].copy()

    # Snow prune: 'snow' in log string or conditions. Removes 4 sessions.
    snow_w = w['workout_raw'].astype(str).str.contains('snow', case=False, na=False)
    snow_c = w['conditions'].astype(str).str.contains('snow', case=False, na=False)
    w = w[~(snow_w | snow_c)].copy()

    # XC correction: two rules combined.
    # (1) 2016-07 through 2016-10 was XC season — Max's HS XC year. No surface
    #     field in the workout log, but the period is XC by recall.
    # (2) Any tempo with quality_distance_m == 5000 was on his HS 5K course,
    #     run as 5 segments with very short rest. Always XC.
    # Both get the standard -6% pace correction (divide by 1.06), same
    # convention as the XC race correction.
    fall_2016 = (w['date'] >= pd.Timestamp('2016-07-01')) & (w['date'] <= pd.Timestamp('2016-10-31'))
    hs_5k = (w['type'] == 'tempo') & (w['quality_distance_m'] == 5000)
    xc_mask = fall_2016 | hs_5k
    w.loc[xc_mask, 'pace_per_mile'] = w.loc[xc_mask, 'pace_per_mile'] / 1.06
    w['xc_corrected'] = xc_mask

    w = add_cs(w, cs, epoch)
    decay = np.exp(-w['rest_per_mile'] / TAU)
    w['D_eff']    = w['rep_dist'] * (1 + (w['rep_count'] - 1) * decay)
    w['t_eff']    = w['pace_per_mile'] * w['D_eff'] / 1609.344
    w['t_5k_hyp'] = (5000 - w['dp_t']) * w['t_eff'] / (w['D_eff'] - w['dp_t'])
    w['p5k_min']  = w['t_5k_hyp'] * 1609.344 / 5000 / 60.0
    w['raw_resid'] = (w['p5k_min'] - w['p5k_cs_min']) * 60
    w['category'] = w['type']
    return w


def project_long_runs(cs, epoch):
    d = pd.read_csv(DAILY_PATH, parse_dates=['date'])
    lr = d[d['run_type'] == 'long'].copy().dropna(subset=['recovery_pace_sec_per_mi', 'miles'])
    lr = lr[lr['miles'] >= LR_MILES_MIN].copy()
    # Snow prune.
    snow_w = lr['workout_raw'].astype(str).str.contains('snow', case=False, na=False)
    snow_c = lr['conditions'].astype(str).str.contains('snow', case=False, na=False)
    lr = lr[~(snow_w | snow_c)].copy()

    lr = add_cs(lr, cs, epoch)
    lr['t_run']   = lr['recovery_pace_sec_per_mi'] * lr['miles']
    lr['d_m']     = lr['miles'] * 1609.344
    lr['t_5k_hyp'] = (5000 - lr['dp_t']) * lr['t_run'] / (lr['d_m'] - lr['dp_t'])
    lr['p5k_min'] = lr['t_5k_hyp'] * 1609.344 / 5000.0 / 60.0
    lr['raw_resid'] = (lr['p5k_min'] - lr['p5k_cs_min']) * 60
    lr['category'] = np.where(lr['miles'] < LR_BIN_SPLIT, 'lr_20-22.9', 'lr_23+')
    return lr


# ---------- hill continuous ----------
# Loop lookup: only `distance_m` is needed for TQ projection. Per-loop offsets
# absorb all loop-specific systematics (grade, surface, terrain), so no grade
# adjustment is applied here. Elevation data lives in the "hills" tab of
# Max's Running Data; it's used by the future all-workouts qualitative plot
# for display purposes (estimated total climb), not in TQ.
HC_LOOPS = {
    'lc':   {'distance_m': 1290},
    'rc':   {'distance_m':  850},
    'pwr1': {'distance_m':  620},
}


def project_hill_continuous(cs, epoch):
    """Project hill_cont sessions to P5K via CS-hyp on actual loop pace.

    No grade adjustment. The per-loop offset (Stage 5) absorbs all loop-
    specific systematics — grade, surface, terrain — by construction, since
    Minetti-style adjustments produce a uniform multiplicative shift per
    loop that the offset would just subtract back out.

    Scope: lc, rc, pwr1 only (n>7 cutoff). Other loops are dropped here and
    will be displayed without TQ contribution on the all-workouts plot.
    """
    import re
    d = pd.read_csv(DAILY_PATH, parse_dates=['date'])
    h = d[d['run_type'] == 'hill_cont'].copy()
    # Drop hc/rep hybrids (Sept 2016)
    h = h[~h['workout_raw'].astype(str).str.contains(r'hc/rep', regex=True, na=False)].copy()

    def _parse(row):
        s = str(row['workout_raw'])
        m_min = re.search(r'(\d+)hc-', s)
        minutes = int(m_min.group(1)) if m_min else None
        m_n = re.search(r'hc-(\d+)x', s)
        nreps = int(m_n.group(1)) if m_n else None
        m_loc = re.search(r'hc-\d+x\s+([a-zA-Z0-9]+)', s)
        loop = m_loc.group(1) if m_loc else None
        if loop is None:
            loc_col = str(row['location']).lower()
            if 'rollercoaster' in loc_col: loop = 'rc'
            elif 'powerline west' in loc_col: loop = 'pwr1'
        return pd.Series([minutes, nreps, loop])

    h[['session_min', 'nreps', 'loop']] = h.apply(_parse, axis=1)
    h = h[h['loop'].isin(HC_LOOPS.keys())].copy()
    h = h.dropna(subset=['session_min', 'nreps'])

    # Snow prune (consistency with workouts/long runs)
    snow_w = h['workout_raw'].astype(str).str.contains('snow', case=False, na=False)
    snow_c = h['conditions'].astype(str).str.contains('snow', case=False, na=False)
    h = h[~(snow_w | snow_c)].copy()

    h['quality_dist_m'] = h.apply(
        lambda r: r['nreps'] * HC_LOOPS[r['loop']]['distance_m'], axis=1)
    h['actual_pace_s'] = (h['session_min'] * 60.0) / (h['quality_dist_m'] / 1609.344)
    h['d_m']     = h['quality_dist_m']
    h['t_eff']   = h['actual_pace_s'] * h['d_m'] / 1609.344
    h = add_cs(h, cs, epoch)
    h['t_5k_hyp'] = (5000 - h['dp_t']) * h['t_eff'] / (h['d_m'] - h['dp_t'])
    h['p5k_min']  = h['t_5k_hyp'] * 1609.344 / 5000.0 / 60.0
    h['raw_resid'] = (h['p5k_min'] - h['p5k_cs_min']) * 60
    h['category'] = 'hill_' + h['loop'].astype(str)
    return h


def apply_offsets(workouts, long_runs, hills=None):
    """Compute per-category median offsets across the combined set, return all
    frames augmented with offset/resid columns, plus the offsets dict."""
    parts = [
        workouts[['date', 'category', 'raw_resid']],
        long_runs[['date', 'category', 'raw_resid']],
    ]
    if hills is not None and len(hills):
        parts.append(hills[['date', 'category', 'raw_resid']])
    combined = pd.concat(parts, ignore_index=True)
    offsets = combined.groupby('category')['raw_resid'].median().to_dict()

    workouts = workouts.copy()
    workouts['offset'] = workouts['category'].map(offsets)
    workouts['resid']  = workouts['raw_resid'] - workouts['offset']

    long_runs = long_runs.copy()
    long_runs['offset'] = long_runs['category'].map(offsets)
    long_runs['resid']  = long_runs['raw_resid'] - long_runs['offset']

    if hills is not None:
        hills = hills.copy()
        hills['offset'] = hills['category'].map(offsets)
        hills['resid']  = hills['raw_resid'] - hills['offset']

    return workouts, long_runs, hills, offsets


# ---------- hover string builders ----------
# Per-session HTML for the smart-spikeline scaffold's snap mode and the
# smooth-mode "Nearest session" caption. The scaffold prepends the trend
# section (CS, smoother, diff) and the date itself in smooth mode, so
# this content focuses on what's session-specific. Category label stays
# as a small heading.
def workout_hover(r):
    xc_note = ' <i style="color:#9cf">[XC-corrected -6%]</i>' if r.get('xc_corrected') else ''
    parts = [
        f"<b>{CAT_LABEL.get(r['category'], r['category'])}</b>{xc_note}",
        f"{int(r['rep_count'])} × {int(r['rep_dist'])}m @ "
        f"{sec_to_mss(r['pace_per_mile'])}/mi"
        + (f", rest {sec_to_mss(r['rest_per_mile'])}/mi"
           if pd.notna(r['rest_per_mile']) and r['rest_per_mile'] > 0 else ""),
        f"<b>P5K projected:</b> {fmt_min(r['p5k_min'])}/mi   "
        f"<b>P5K from CS:</b> {fmt_min(r['p5k_cs_min'])}/mi",
        f"<b>Raw resid:</b> {r['raw_resid']:+.1f}s/mi   "
        f"<b>Corrected:</b> {r['resid']:+.1f}s/mi",
    ]
    return "<br>".join(p for p in parts if p)


def long_run_hover(r):
    parts = [
        f"<b>{CAT_LABEL.get(r['category'], r['category'])}</b>",
        f"<b>Distance:</b> {r['miles']:.1f}mi   "
        f"<b>Pace:</b> {sec_to_mss(r['recovery_pace_sec_per_mi'])}/mi",
        f"<b>P5K projected:</b> {fmt_min(r['p5k_min'])}/mi   "
        f"<b>P5K from CS:</b> {fmt_min(r['p5k_cs_min'])}/mi",
        f"<b>Raw resid:</b> {r['raw_resid']:+.1f}s/mi   "
        f"<b>Corrected:</b> {r['resid']:+.1f}s/mi",
    ]
    return "<br>".join(p for p in parts if p)


def hill_hover(r):
    parts = [
        f"<b>{CAT_LABEL.get(r['category'], r['category'])}</b>",
        f"{int(r['nreps'])} × {r['loop']} loop, {int(r['session_min'])} min total",
        f"<b>Actual pace:</b> {sec_to_mss(r['actual_pace_s'])}/mi",
        f"<b>P5K projected:</b> {fmt_min(r['p5k_min'])}/mi   "
        f"<b>P5K from CS:</b> {fmt_min(r['p5k_cs_min'])}/mi",
        f"<b>Raw resid:</b> {r['raw_resid']:+.1f}s/mi   "
        f"<b>Corrected:</b> {r['resid']:+.1f}s/mi",
    ]
    return "<br>".join(p for p in parts if p)


# ---------- main ----------
def main():
    cs, epoch = load_cs()
    workouts  = project_workouts(cs, epoch)
    long_runs = project_long_runs(cs, epoch)
    hills     = project_hill_continuous(cs, epoch)

    # Iterative resid-cutoff prune. After each pass, recompute category offsets
    # on the pruned set and check whether any remaining points now exceed the
    # cutoff (since the median can drop after removing high outliers).
    CUTOFF = 23.3
    print(f'\n--- Iterative resid > +{CUTOFF} prune ---')
    pruned_w_idx = set()
    pruned_lr_idx = set()
    pruned_h_idx = set()
    initial_offsets = None
    for it in range(15):
        w_keep = workouts.drop(index=list(pruned_w_idx))
        lr_keep = long_runs.drop(index=list(pruned_lr_idx))
        h_keep = hills.drop(index=list(pruned_h_idx))
        _, _, _, offsets = apply_offsets(w_keep, lr_keep, h_keep)
        if initial_offsets is None:
            initial_offsets = offsets
        w_keep = w_keep.copy()
        lr_keep = lr_keep.copy()
        h_keep = h_keep.copy()
        w_keep['resid'] = w_keep['raw_resid'] - w_keep['category'].map(offsets)
        lr_keep['resid'] = lr_keep['raw_resid'] - lr_keep['category'].map(offsets)
        h_keep['resid'] = h_keep['raw_resid'] - h_keep['category'].map(offsets)

        new_w = w_keep.index[w_keep['resid'] > CUTOFF].tolist()
        new_lr = lr_keep.index[lr_keep['resid'] > CUTOFF].tolist()
        new_h = h_keep.index[h_keep['resid'] > CUTOFF].tolist()
        if not new_w and not new_lr and not new_h:
            print(f'  Iteration {it+1}: stable. Done.')
            break

        pruned_w_idx.update(new_w)
        pruned_lr_idx.update(new_lr)
        pruned_h_idx.update(new_h)
        print(f'  Iteration {it+1}: +{len(new_w)} workouts, +{len(new_lr)} long runs, +{len(new_h)} hills')
        for i in new_w:
            r = workouts.loc[i]
            print(f'    W  {r["date"].date()}  {r["category"]:<22} resid={w_keep.loc[i,"resid"]:+5.1f}  '
                  f'raw={r["raw_resid"]:+5.1f}  pace={int(r["pace_per_mile"])}s/mi')
        for i in new_lr:
            r = long_runs.loc[i]
            print(f'    LR {r["date"].date()}  {r["category"]:<22} resid={lr_keep.loc[i,"resid"]:+5.1f}  '
                  f'raw={r["raw_resid"]:+5.1f}  miles={r["miles"]:.1f}  pace={int(r["recovery_pace_sec_per_mi"])}s/mi')
        for i in new_h:
            r = hills.loc[i]
            print(f'    H  {r["date"].date()}  {r["category"]:<22} resid={h_keep.loc[i,"resid"]:+5.1f}  '
                  f'raw={r["raw_resid"]:+5.1f}  loop={r["loop"]}  {int(r["nreps"])}x{int(r["session_min"])}min')

    workouts = workouts.drop(index=list(pruned_w_idx)).copy()
    long_runs = long_runs.drop(index=list(pruned_lr_idx)).copy()
    hills = hills.drop(index=list(pruned_h_idx)).copy()
    workouts, long_runs, hills, offsets = apply_offsets(workouts, long_runs, hills)

    print('\n--- Offset shifts (initial -> final) ---')
    for cat in sorted(set(initial_offsets) | set(offsets)):
        i_off = initial_offsets.get(cat, float('nan'))
        f_off = offsets.get(cat, float('nan'))
        print(f'  {cat:<22} {i_off:+6.2f}  ->  {f_off:+6.2f}  (Δ {f_off-i_off:+.2f})')
    print(f'\nKept: {len(workouts)} workouts, {len(long_runs)} long runs, {len(hills)} hills')

    print(f'Workouts: {len(workouts)},  long runs: {len(long_runs)},  hills: {len(hills)}')
    print('Per-category offsets (median raw resid):')
    for c, o in sorted(offsets.items()):
        print(f'  {c:<22} offset={o:+6.2f}')

    # Combined for the smoother
    combined = pd.concat([
        workouts[['date', 'resid']],
        long_runs[['date', 'resid']],
        hills[['date', 'resid']],
    ], ignore_index=True).sort_values('date').reset_index(drop=True)

    ds = (combined['date'] - epoch).dt.days.astype(float).values
    res = combined['resid'].values

    grid_dates = pd.date_range(combined['date'].min(),
                               combined['date'].max(),
                               freq=GRID_FREQ)
    grid_days = (grid_dates - epoch).days.astype(float).values
    smoothed = adaptive_gauss_smoother(
        ds, res, grid_days,
        target_ess=GAUSS_TARGET_ESS,
        base_bw=GAUSS_BASE_BW_DAYS,
        max_bw=GAUSS_MAX_BW_DAYS,
    )

    # Break the track in any gap > GAP_BREAK_DAYS in the training data.
    # NaN values create disconnected line segments naturally in plotly.
    sorted_input = combined['date'].sort_values().reset_index(drop=True)
    diffs = sorted_input.diff().dt.days
    gap_idx = diffs[diffs > GAP_BREAK_DAYS].index
    for idx in gap_idx:
        gap_start = sorted_input[idx - 1]
        gap_end = sorted_input[idx]
        mask = (grid_dates > gap_start) & (grid_dates < gap_end)
        smoothed[mask] = np.nan
        print(f'Track broken: {gap_start.date()} -> {gap_end.date()} '
              f'({(gap_end - gap_start).days} days)')

    p5k_at_grid = np.interp(grid_days, cs['day'].values, cs['p5k_implied_min'].values)
    track = p5k_at_grid + smoothed / 60.0

    # Position each session at CS-implied + corrected residual
    workouts['pos_min']  = workouts['p5k_cs_min']  + workouts['resid']  / 60.0
    long_runs['pos_min'] = long_runs['p5k_cs_min'] + long_runs['resid'] / 60.0
    hills['pos_min']     = hills['p5k_cs_min']     + hills['resid']     / 60.0

    # ---------- build figure ----------
    fig = go.Figure()

    # CS-implied 5K reference (full series, clipped at plot range)
    cs_plot = cs[cs['date'] >= pd.Timestamp('2016-01-01')]
    fig.add_trace(go.Scatter(
        x=cs_plot['date'], y=cs_plot['p5k_implied_min'],
        mode='lines', name='CS-implied 5K',
        line=dict(color='steelblue', width=2),
        hoverinfo='skip',
    ))

    # Smoother track. Pass full array including NaN values: plotly's line mode
    # skips NaN naturally and creates the visual break at the gap.
    fig.add_trace(go.Scatter(
        x=grid_dates, y=track,
        mode='lines', name='Training-quality track',
        line=dict(color='#eaeaea', width=2.5),
        connectgaps=False,
        hoverinfo='skip',
    ))

    # Workouts: one trace per category for legend filtering (reps excluded
    # above). Per-marker customdata + meta.snap_eligible feeds the smart
    # spikeline scaffold's snap mode — hover near a marker shows that
    # session's details, hover anywhere else shows the per-day smooth
    # tooltip (CS pace + smoother trend + nearest session).
    for cat in ['interval', 'tempo', 'continuous_fartlek']:
        sub = workouts[workouts['category'] == cat]
        if sub.empty:
            continue
        cd = [workout_hover(r) for _, r in sub.iterrows()]
        fig.add_trace(go.Scatter(
            x=sub['date'], y=sub['pos_min'],
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

    # Long runs: one trace per bin
    for cat in ['lr_20-22.9', 'lr_23+']:
        sub = long_runs[long_runs['category'] == cat]
        if sub.empty:
            continue
        cd = [long_run_hover(r) for _, r in sub.iterrows()]
        fig.add_trace(go.Scatter(
            x=sub['date'], y=sub['pos_min'],
            mode='markers',
            name=f'{CAT_LABEL[cat]} (n={len(sub)})',
            marker=dict(color=CAT_COLORS[cat], size=7, symbol='diamond',
                        line=dict(color='rgba(255,255,255,0.4)', width=0.5),
                        opacity=0.85),
            customdata=cd,
            hoverinfo='skip',
            legendgroup='long_runs', legendgrouptitle_text='Long runs',
            meta={'snap_eligible': True},
        ))

    # Hill continuous: single trace, all loops together
    if len(hills):
        cd = [hill_hover(r) for _, r in hills.iterrows()]
        fig.add_trace(go.Scatter(
            x=hills['date'], y=hills['pos_min'],
            mode='markers',
            name=f'Hill continuous (n={len(hills)})',
            marker=dict(color='#e377c2', size=7, symbol='triangle-up',
                        line=dict(color='rgba(255,255,255,0.4)', width=0.5),
                        opacity=0.85),
            customdata=cd,
            hoverinfo='skip',
            legendgroup='workouts', legendgrouptitle_text='Workouts',
            meta={'snap_eligible': True},
        ))

    # ---------- layout ----------
    y_min, y_max = 4.0 + 20/60, 6.00
    # 10-second tick spacing from 4:20 to 6:00
    ytick_vals = [4.0 + (20 + 10 * k) / 60 for k in range(11)]
    ytick_txt  = [sec_to_mss(v * 60) for v in ytick_vals]

    apply_default_layout(
        fig,
        title=title_block(
            'Training quality vs. observed race fitness',
            '5K fitness trend across normalized training data compared to race-derived baseline',
        ),
        margin=dict(t=TITLE_MARGIN_TOP, l=70, r=220, b=60),
        hovermode=False,
        legend=dict(yanchor='top', y=0.99, xanchor='left', x=1.02,
                    groupclick='toggleitem', font=dict(size=11)),
        xaxis=dict(title='Date', showgrid=True, gridcolor=GRID,
                   tick0='2016-01-01', dtick='M12',
                   range=[pd.Timestamp('2016-01-01'),
                          combined['date'].max() + pd.Timedelta(days=30)]),
        yaxis=dict(title='5K-equivalent pace (min/mi)',
                   range=[y_max, y_min],
                   tickmode='array', tickvals=ytick_vals, ticktext=ytick_txt,
                   showgrid=True, gridcolor=GRID),
    )

    # ---------- cursor-tooltip payload ----------
    # JS gets ms-since-1970 from plotly's date axis, so payload is keyed by
    # day-since-1970 (not the in-script 2016-01-01 epoch).
    js_epoch = pd.Timestamp('1970-01-01')

    plot_start = pd.Timestamp('2016-01-01')
    plot_end   = combined['date'].max() + pd.Timedelta(days=30)
    all_days   = pd.date_range(plot_start, plot_end, freq='D')

    target_days_2016 = (all_days - epoch).days.astype(float).values
    cs_pace_per_day = np.interp(target_days_2016,
                                cs['day'].values,
                                cs['p5k_implied_min'].values)

    # Smoother pace per day. Linear interp between 7-day grid points, but
    # if either bracketing grid point is NaN (gap), the result is NaN.
    smoother_per_day = np.full(len(target_days_2016), np.nan)
    for i, t in enumerate(target_days_2016):
        j = np.searchsorted(grid_days, t)
        if j == 0 or j >= len(grid_days):
            continue
        lv, rv = track[j-1], track[j]
        if np.isnan(lv) or np.isnan(rv):
            continue
        lg, rg = grid_days[j-1], grid_days[j]
        frac = (t - lg) / (rg - lg) if rg != lg else 0.0
        smoother_per_day[i] = lv * (1 - frac) + rv * frac

    sessions = []
    for _, r in workouts.iterrows():
        sessions.append({'day': int((r['date'] - js_epoch).days),
                         'html': workout_hover(r)})
    for _, r in long_runs.iterrows():
        sessions.append({'day': int((r['date'] - js_epoch).days),
                         'html': long_run_hover(r)})
    for _, r in hills.iterrows():
        sessions.append({'day': int((r['date'] - js_epoch).days),
                         'html': hill_hover(r)})
    sessions.sort(key=lambda s: s['day'])

    first_day = int((all_days[0] - js_epoch).days)
    last_day  = int((all_days[-1] - js_epoch).days)

    payload = {
        'first_day': first_day,
        'cs_pace':   [round(float(v), 4) for v in cs_pace_per_day],
        'smoother':  [None if np.isnan(v) else round(float(v), 4)
                      for v in smoother_per_day],
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
  function fmtDiff(s) {
    if (s === null) return '—';
    var sign = s > 0 ? '+' : (s < 0 ? '' : '');
    return sign + s + 's/mi';
  }
  function dateLabel(d) {
    var dt = new Date(d * 86400000);
    var y = dt.getUTCFullYear();
    var m = String(dt.getUTCMonth() + 1).padStart(2, '0');
    var dd = String(dt.getUTCDate()).padStart(2, '0');
    return y + '-' + m + '-' + dd + ' (' + DOW[dt.getUTCDay()] + ')';
  }

  var cs   = P.cs_pace[idx];
  var sm   = P.smoother[idx];
  var diff = (sm !== null && sm !== undefined) ? Math.round((sm - cs) * 60) : null;

  var html = '';
  html += '<div class="tt-date">' + dateLabel(day) + '</div>';

  // Section 1: trend info — race fitness from CS, training-quality
  // smoother, and their difference at this date.
  html += '<div class="tt-section">';
  html += '<div class="tt-row"><span>Race fitness</span><b>' + fmtMin(cs) + '/mi</b></div>';
  html += '<div class="tt-row"><span>Training quality</span><b>' + fmtMin(sm) + '/mi</b></div>';
  if (diff !== null) {
    html += '<div class="tt-row"><span>Diff</span><b>' + fmtDiff(diff) + '</b></div>';
  }
  html += '</div>';

  // Section 2: session details. Smooth = nearest within window.
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
        title_slug='training_quality',
        page_title='Training quality',
        cursor_tooltip=CursorTooltip(
            payload=payload,
            build_js=build_js,
            first_day=first_day,
            last_day=last_day,
        ),
    )
    print(f'\nWrote {OUT_HTML}')


if __name__ == '__main__':
    main()
