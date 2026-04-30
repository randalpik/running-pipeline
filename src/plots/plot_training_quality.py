"""
plot_training_quality.py — Interactive Plotly version of the training-quality
visualization.

Shows every workout (from workout_decomposed_v7.csv) and qualifying long run
plotted at CS_implied + corrected residual, with the adaptive-Gaussian smoother
track laid on top of the CS-implied 5K curve.

Each point's hover shows the original log string plus all derived fields,
intended for pruning / outlier evaluation.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go


# ---------- paths ----------
def _resolve(name, candidates):
    for c in candidates:
        if Path(c).exists():
            return c
    raise SystemExit(f'Could not find {name}. Tried: {candidates}')


def _resolve_out_dir():
    for d in ['./output', '/mnt/user-data/outputs']:
        p = Path(d)
        if p.exists() and p.is_dir():
            return p
    p = Path('./output'); p.mkdir(exist_ok=True); return p


WORKOUTS_PATH = _resolve('workout_decomposed_v7.csv', [
    './output/workout_decomposed_v7.csv',
    './workout_decomposed_v7.csv',
    '/mnt/user-data/outputs/workout_decomposed_v7.csv',
])
DAILY_PATH = _resolve('daily.csv', [
    './output/daily.csv', './daily.csv', '/mnt/project/daily.csv',
])
CS_PATH = _resolve('bayes_cs_summary_v11.csv', [
    './output/bayes_cs_summary_v11.csv',
    './bayes_cs_summary_v11.csv',
    '/mnt/project/bayes_cs_summary_v11.csv',
])
OUT_HTML = str(_resolve_out_dir() / 'training_quality.html')

# ---------- pipeline parameters (match training_unified_pipeline.py) ----------
TAU = 210.0
LR_MILES_MIN = 20         # long run filter, low end (no upper bound)
LR_BIN_SPLIT = 23         # split point for two-bin long run classification

# ---------- smoother parameters (match training_quality_track.py) ----------
GAUSS_BASE_BW_DAYS = 30
GAUSS_TARGET_ESS   = 12
GAUSS_MAX_BW_DAYS  = 400
GRID_FREQ          = '7D'

# If this many days pass without a training datapoint (workout, long run, or
# hill_cont), break the smoother track. The 2020-11 -> 2021-04 labrum gap is
# the canonical case. Set high enough that race-only periods (e.g. summer
# 2018) don't trigger.
GAP_BREAK_DAYS = 90

# ---------- visual config ----------
CAT_COLORS = {
    'interval':           '#d62728',
    'tempo':              '#2ca02c',
    'rep':                '#9467bd',
    'continuous_fartlek': '#ff7f0e',
    'lr_20-22.9':         '#17becf',
    'lr_23+':             '#1f77b4',
    'hill_lc':            '#e377c2',
    'hill_rc':            '#bcbd22',
    'hill_pwr1':          '#8c564b',
}
CAT_LABEL = {
    'interval': 'Interval', 'tempo': 'Tempo', 'rep': 'Rep',
    'continuous_fartlek': 'Cont. fartlek',
    'lr_20-22.9': 'Long 20–22.9', 'lr_23+': 'Long 23+',
    'hill_lc': 'Hill (lc)', 'hill_rc': 'Hill (rc)', 'hill_pwr1': 'Hill (pwr1)',
}


# ---------- formatting helpers ----------
def sec_to_mss(s):
    if pd.isna(s):
        return ''
    s = float(s)
    m = int(s // 60)
    sec = int(round(s - 60 * m))
    if sec == 60:
        m += 1
        sec = 0
    return f'{m}:{sec:02d}'


def fmt_min(m):
    """Format minutes (float) as M:SS."""
    if pd.isna(m):
        return ''
    return sec_to_mss(m * 60)


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


def adaptive_gauss_smoother(ds, res, grid_days,
                             target_ess=GAUSS_TARGET_ESS,
                             base_bw=GAUSS_BASE_BW_DAYS,
                             max_bw=GAUSS_MAX_BW_DAYS):
    """Adaptive Gaussian smoother. Bandwidth is the smallest value such that
    ESS >= target_ess at this grid point, found by bisection so it varies
    continuously across the grid (avoids visible jogs from discrete steps)."""
    out = np.full(len(grid_days), np.nan)

    def weights_and_ess(t, bw):
        w = np.exp(-0.5 * ((ds - t) / bw) ** 2)
        s = w.sum()
        ss = (w * w).sum()
        return w, (s * s / ss) if ss > 0 else 0.0

    for i, t in enumerate(grid_days):
        # If base bandwidth already satisfies ESS, use it (smoothest result).
        w, ess = weights_and_ess(t, base_bw)
        if ess >= target_ess:
            out[i] = (w * res).sum() / w.sum()
            continue

        # Find an upper bracket where ESS clears the target.
        hi = base_bw
        while hi < max_bw:
            hi = min(hi * 2, max_bw)
            w, ess = weights_and_ess(t, hi)
            if ess >= target_ess:
                break
        if ess < target_ess:
            continue  # max_bw insufficient; leave NaN

        # Bisect to find the smallest bw that satisfies ESS >= target.
        lo = base_bw
        for _ in range(40):
            if hi - lo < 0.5:
                break
            mid = 0.5 * (lo + hi)
            w, ess = weights_and_ess(t, mid)
            if ess >= target_ess:
                hi = mid
            else:
                lo = mid
        w, _ = weights_and_ess(t, hi)
        out[i] = (w * res).sum() / w.sum()
    return out


# ---------- hover string builders ----------
def workout_hover(r):
    xc_note = ' <i style="color:#9cf">[XC-corrected -6%]</i>' if r.get('xc_corrected') else ''
    parts = [
        f"<b>{CAT_LABEL.get(r['category'], r['category'])}</b>  "
        f"{r['date'].strftime('%Y-%m-%d')}  ({r['date'].strftime('%a')}){xc_note}",
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
        f"<b>{CAT_LABEL.get(r['category'], r['category'])}</b>  "
        f"{r['date'].strftime('%Y-%m-%d')}  ({r['date'].strftime('%a')})",
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
        f"<b>{CAT_LABEL.get(r['category'], r['category'])}</b>  "
        f"{r['date'].strftime('%Y-%m-%d')}  ({r['date'].strftime('%a')})",
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
    smoothed = adaptive_gauss_smoother(ds, res, grid_days)

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

    # Workouts: one trace per category for legend filtering (reps excluded above)
    for cat in ['interval', 'tempo', 'continuous_fartlek']:
        sub = workouts[workouts['category'] == cat]
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub['date'], y=sub['pos_min'],
            mode='markers',
            name=f'{CAT_LABEL[cat]} (n={len(sub)})',
            marker=dict(color=CAT_COLORS[cat], size=7,
                        line=dict(color='rgba(255,255,255,0.4)', width=0.5),
                        opacity=0.85),
            hoverinfo='skip',
            legendgroup='workouts', legendgrouptitle_text='Workouts',
        ))

    # Long runs: one trace per bin
    for cat in ['lr_20-22.9', 'lr_23+']:
        sub = long_runs[long_runs['category'] == cat]
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub['date'], y=sub['pos_min'],
            mode='markers',
            name=f'{CAT_LABEL[cat]} (n={len(sub)})',
            marker=dict(color=CAT_COLORS[cat], size=7, symbol='diamond',
                        line=dict(color='rgba(255,255,255,0.4)', width=0.5),
                        opacity=0.85),
            hoverinfo='skip',
            legendgroup='long_runs', legendgrouptitle_text='Long runs',
        ))

    # Hill continuous: single trace, all loops together
    if len(hills):
        fig.add_trace(go.Scatter(
            x=hills['date'], y=hills['pos_min'],
            mode='markers',
            name=f'Hill continuous (n={len(hills)})',
            marker=dict(color='#e377c2', size=7, symbol='triangle-up',
                        line=dict(color='rgba(255,255,255,0.4)', width=0.5),
                        opacity=0.85),
            hoverinfo='skip',
            legendgroup='workouts', legendgrouptitle_text='Workouts',
        ))

    # ---------- layout ----------
    y_min, y_max = 4.0 + 20/60, 6.00
    # 10-second tick spacing from 4:20 to 6:00
    ytick_vals = [4.0 + (20 + 10 * k) / 60 for k in range(11)]
    ytick_txt  = [sec_to_mss(v * 60) for v in ytick_vals]

    fig.update_layout(
        title=dict(
            text='Training quality vs. observed race fitness'
                 '<br><sub style="font-size:13px;color:#bbb">'
                 '5K fitness trend across normalized training data compared to race-derived baseline'
                 '</sub>',
            y=0.965),
        template='plotly_dark',
        paper_bgcolor='#1a1a1a', plot_bgcolor='#1a1a1a',
        font=dict(color='#eee'),
        autosize=True,
        margin=dict(t=110, l=70, r=220, b=60),
        hovermode=False,
        legend=dict(yanchor='top', y=0.99, xanchor='left', x=1.02,
                    groupclick='toggleitem', font=dict(size=11)),
        xaxis=dict(title='Date', showgrid=True, gridcolor='#333',
                   tick0='2016-01-01', dtick='M12',
                   range=[pd.Timestamp('2016-01-01'),
                          combined['date'].max() + pd.Timedelta(days=30)]),
        yaxis=dict(title='5K-equivalent pace (min/mi)',
                   range=[y_max, y_min],
                   tickmode='array', tickvals=ytick_vals, ticktext=ytick_txt,
                   showgrid=True, gridcolor='#333'),
    )

    # ---------- write HTML with full-screen dark wrapper ----------
    Path(OUT_HTML).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(OUT_HTML, include_plotlyjs=True, full_html=True,
                   config={'responsive': True})

    # ----- Build hover payload for custom vertical-hover tooltip -----
    # JS gets ms-since-1970 from plotly's date axis, so payload is keyed by
    # day-since-1970 (not the in-script 2016-01-01 epoch).
    js_epoch = pd.Timestamp('1970-01-01')

    plot_start = pd.Timestamp('2016-01-01')
    plot_end   = combined['date'].max() + pd.Timedelta(days=30)
    all_days   = pd.date_range(plot_start, plot_end, freq='D')

    # CS pace per day (linear interp from cs daily series)
    target_days_2016 = (all_days - epoch).days.astype(float).values
    cs_pace_per_day = np.interp(target_days_2016,
                                cs['day'].values,
                                cs['p5k_implied_min'].values)

    # Smoother pace per day. Linear interp between 7-day grid points, but
    # if either bracketing grid point is NaN (gap), the result is NaN.
    target_days_grid = target_days_2016
    smoother_per_day = np.full(len(target_days_grid), np.nan)
    for i, t in enumerate(target_days_grid):
        j = np.searchsorted(grid_days, t)
        if j == 0 or j >= len(grid_days):
            continue
        lv, rv = track[j-1], track[j]
        if np.isnan(lv) or np.isnan(rv):
            continue
        lg, rg = grid_days[j-1], grid_days[j]
        frac = (t - lg) / (rg - lg) if rg != lg else 0.0
        smoother_per_day[i] = lv * (1 - frac) + rv * frac

    # Sessions list, sorted by date, with rendered HTML
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
        'last_day':  last_day,
        'cs_pace':   [round(float(v), 4) for v in cs_pace_per_day],
        'smoother':  [None if np.isnan(v) else round(float(v), 4)
                      for v in smoother_per_day],
        'sessions':  sessions,
    }
    payload_json = json.dumps(payload, separators=(',', ':'))

    # ----- CSS + custom tooltip + spike line + JS -----
    css_and_js = r"""
<style>
html, body {
  margin: 0; padding: 0;
  width: 100%; height: 100%;
  background: #1a1a1a; color: #eee;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
               "Helvetica Neue", Arial, sans-serif;
}
.plotly-graph-div, .js-plotly-plot {
  width: 100% !important;
  height: 100vh !important;
}

#tq-tooltip {
  position: fixed; top: 0; left: 0;
  background: rgba(26,26,26,0.96);
  color: #eee;
  border: 1px solid #555;
  padding: 10px 14px;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
               "Helvetica Neue", Arial, sans-serif;
  font-size: 12px;
  line-height: 1.5;
  border-radius: 4px;
  pointer-events: none;
  z-index: 9999;
  max-width: 420px;
  display: none;
  box-shadow: 0 4px 12px rgba(0,0,0,0.4);
}
#tq-tooltip .tq-day {
  font-weight: 600;
  font-size: 13px;
  color: #fff;
  margin-bottom: 4px;
}
#tq-tooltip .tq-row {
  display: flex; justify-content: space-between;
  gap: 18px; white-space: nowrap;
}
#tq-tooltip .tq-row > span:first-child { color: #aaa; }
#tq-tooltip .tq-section {
  margin-top: 8px; padding-top: 8px;
  border-top: 1px solid #444;
}
#tq-tooltip .tq-near-label {
  color: #aaa; font-size: 11px; margin-bottom: 2px;
}
#tq-tooltip b { color: #fff; font-weight: 600; }
#tq-tooltip i { color: #9cf; }
#tq-tooltip code {
  background: rgba(255,255,255,0.08);
  padding: 1px 5px;
  border-radius: 3px;
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  font-size: 11px;
}
#tq-spike {
  position: fixed; top: 0; left: 0;
  width: 1px; height: 100vh;
  background: rgba(255,255,255,0.3);
  pointer-events: none;
  z-index: 9998;
  display: none;
}
</style>
<div id="tq-tooltip"></div>
<div id="tq-spike"></div>
<script>
(function() {
  var data = __TQ_PAYLOAD__;
  var firstDay = data.first_day;
  var lastDay  = data.last_day;
  var csPace   = data.cs_pace;
  var smoother = data.smoother;
  var sessions = data.sessions;
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
  function dayLabel(day) {
    var d = new Date(day * 86400000);
    var y = d.getUTCFullYear();
    var m = String(d.getUTCMonth() + 1).padStart(2, '0');
    var dd = String(d.getUTCDate()).padStart(2, '0');
    return y + '-' + m + '-' + dd + ' (' + DOW[d.getUTCDay()] + ')';
  }
  function nearestSession(day) {
    if (sessions.length === 0) return null;
    if (day <= sessions[0].day) return sessions[0];
    if (day >= sessions[sessions.length - 1].day) return sessions[sessions.length - 1];
    var lo = 0, hi = sessions.length - 1;
    while (lo < hi) {
      var mid = (lo + hi) >> 1;
      if (sessions[mid].day < day) lo = mid + 1;
      else hi = mid;
    }
    var a = sessions[lo - 1], b = sessions[lo];
    return Math.abs(a.day - day) <= Math.abs(b.day - day) ? a : b;
  }
  function buildTooltip(day) {
    var idx = day - firstDay;
    if (idx < 0 || idx >= csPace.length) return '';
    var cs = csPace[idx];
    var sm = smoother[idx];
    var diff = (sm !== null && sm !== undefined) ? Math.round((sm - cs) * 60) : null;

    var html = '<div class="tq-day">' + dayLabel(day) + '</div>';
    html += '<div class="tq-row"><span>Race fitness:</span><span>' + fmtMin(cs) + '/mi</span></div>';
    html += '<div class="tq-row"><span>Training quality:</span><span>' + fmtMin(sm) + '/mi</span></div>';
    if (diff !== null) {
      html += '<div class="tq-row"><span>Diff:</span><span>' + fmtDiff(diff) + '</span></div>';
    }
    var n = nearestSession(day);
    if (n) {
      var dd = n.day - day;
      var lbl = dd === 0 ? 'same day' : (dd > 0 ? '+' + dd + ' day' + (dd === 1 ? '' : 's') : dd + ' day' + (dd === -1 ? '' : 's'));
      html += '<div class="tq-section">';
      html += '<div class="tq-near-label">Nearest session [' + lbl + ']</div>';
      html += n.html;
      html += '</div>';
    }
    return html;
  }

  var tt = document.getElementById('tq-tooltip');
  var spike = document.getElementById('tq-spike');
  var lastContent = '', ttW = 0, ttH = 0;
  var rafScheduled = false, pendingX = 0, pendingY = 0;
  var pendingContent = '', pendingSpikeX = 0, pendingShow = false;

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
      ttW = tt.offsetWidth; ttH = tt.offsetHeight;
    }
    var x = pendingX + 15, y = pendingY + 10;
    if (x + ttW > window.innerWidth)  x = pendingX - ttW - 15;
    if (y + ttH > window.innerHeight) y = pendingY - ttH - 10;
    tt.style.transform = 'translate(' + x + 'px,' + y + 'px)';
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
      var pl = rect.left + bg.l, pr = rect.left + bg.l + bg.w;
      var pt = rect.top + bg.t,  pb = rect.top + bg.t + bg.h;
      if (e.clientX < pl || e.clientX > pr || e.clientY < pt || e.clientY > pb) {
        pendingShow = false;
        if (!rafScheduled) { rafScheduled = true; requestAnimationFrame(update); }
        return;
      }
      var dataX = xa.p2c(e.clientX - rect.left - bg.l);
      var day = Math.round(dataX / 86400000);
      if (day < firstDay) day = firstDay;
      if (day > lastDay)  day = lastDay;
      pendingContent = buildTooltip(day);
      pendingX = e.clientX; pendingY = e.clientY;
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
"""
    css_and_js = css_and_js.replace('__TQ_PAYLOAD__', payload_json)

    with open(OUT_HTML) as f:
        html = f.read()
    html = html.replace('<body>', '<body style="margin:0;padding:0;background:#1a1a1a;">')
    html = html.replace('</body>', css_and_js + '</body>')
    with open(OUT_HTML, 'w') as f:
        f.write(html)

    print(f'\nWrote {OUT_HTML}')


if __name__ == '__main__':
    main()
