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
from src.shared.workouts import (
    load_cs, add_cs,
    project_workouts, project_long_runs, project_hill_continuous,
    LONG_FLOOR, LONG_CEIL, LR_INTERNAL_BIN,
    WORKOUTS_PATH, DAILY_PATH, CS_PATH,
)
from src.shared.long_run_model import (
    fit_long_run_model, MIN_ROUTE_N, PRUNE_SIGMA,
)
from src.plotting import widgets
from src.plotting import (render_plot, CursorTooltip, apply_default_layout,
                            right_margin_for_anchored_box,
                            sec_to_mss, fmt_min, CAT_COLORS, GRID, CS_LINE,
                            SURFACES, GAP_BREAK_DAYS, adaptive_gauss_smoother,
                            yearly_x_axis_kwargs)

# Width of the route-betas box (#tq-routes); also used to size margin.r.
ROUTES_BOX_WIDTH = 196

_PLOTS_DIR = Path(__file__).resolve().parent
_TQ_JS = _PLOTS_DIR / 'plot_training_quality.js'


OUTPUT_DIR.mkdir(exist_ok=True)
OUT_HTML = str(OUTPUT_DIR / 'training_quality.html')
TRACK_CSV = DATA_DIR / 'training_quality_track.csv'

for _required in (WORKOUTS_PATH, DAILY_PATH, CS_PATH):
    if not _required.exists():
        raise SystemExit(f'Could not find {_required}')

# ---------- pipeline parameters ----------
# MIN_ROUTE_N and PRUNE_SIGMA now live in src.shared.long_run_model so the
# Dashboard tab can reuse the same fit without importing this plot script.

# ---------- smoother parameters (match training_quality_track.py) ----------
GAUSS_BASE_BW_DAYS = 30
GAUSS_TARGET_ESS   = 12
GAUSS_MAX_BW_DAYS  = 400
GRID_FREQ          = '7D'

# ---------- visual config ----------
CAT_LABEL = {
    'interval': 'Interval', 'tempo': 'Tempo', 'rep': 'Rep',
    'continuous_fartlek': 'Cont. fartlek',
    'lr_lo': f'Long {LONG_FLOOR:g}–{LR_INTERNAL_BIN - 0.1:g}',
    'lr_hi': f'Long {LR_INTERNAL_BIN:g}–{LONG_CEIL:g}',
    'hill_lc': 'Hill (lc)', 'hill_rc': 'Hill (rc)', 'hill_pwr1': 'Hill (pwr1)',
}


# ---------- pipeline ----------
# load_cs, add_cs, project_workouts, project_long_runs, project_hill_continuous,
# HC_LOOPS, HILL_LOOP_META are imported from src.shared.workouts.


def apply_offsets(workouts, hills=None):
    """Compute per-category median offsets across workouts (+ optional hills),
    return both frames augmented with offset/resid columns plus the offsets
    dict. Long runs are corrected by `fit_long_run_model` instead of pooled
    here — their categories ('lr_lo'/'lr_hi') don't appear in this dict."""
    parts = [workouts[['date', 'category', 'raw_resid']]]
    if hills is not None and len(hills):
        parts.append(hills[['date', 'category', 'raw_resid']])
    combined = pd.concat(parts, ignore_index=True)
    offsets = combined.groupby('category')['raw_resid'].median().to_dict()

    workouts = workouts.copy()
    workouts['offset'] = workouts['category'].map(offsets)
    workouts['resid']  = workouts['raw_resid'] - workouts['offset']

    if hills is not None:
        hills = hills.copy()
        hills['offset'] = hills['category'].map(offsets)
        hills['resid']  = hills['raw_resid'] - hills['offset']

    return workouts, hills, offsets


# ---------- hover string builders ----------
# Per-session HTML for the smart-spikeline scaffold's snap mode and the
# smooth-mode "Nearest session" caption. The scaffold prepends the trend
# section (CS, smoother, diff) and the date itself in smooth mode, so
# this content focuses on what's session-specific. Category label stays
# as a small heading.

def _route_paren(display_name, city_state):
    """Format ' (display_name, city_state)' with graceful fallback when
    fields are blank. Returns '' if both are missing."""
    parts = [str(x).strip() for x in (display_name, city_state)
             if pd.notna(x) and str(x).strip()]
    return f' ({", ".join(parts)})' if parts else ''


# Tooltip-only labels (legend uses CAT_LABEL — keeps abbreviated form there).
TOOLTIP_TITLE = {
    'interval': 'Intervals',
    'continuous_fartlek': 'Continuous fartlek',
}


def workout_hover(r):
    cat = r['category']
    title = TOOLTIP_TITLE.get(cat, CAT_LABEL.get(cat, cat))
    title += _route_paren(r.get('display_name'), r.get('city_state'))
    xc_note = f' <span style="color:{SURFACES["XC"]}">(XC-corrected)</span>' if r.get('xc_corrected') else ''
    rep_count = int(r['rep_count'])
    rep_dist = int(r['rep_dist'])
    if cat == 'continuous_fartlek' and rep_count == 1:
        body = f"{rep_dist}m @ {sec_to_mss(r['pace_per_mile'])}/mi"
    else:
        body = (f"{rep_count} × {rep_dist}m @ "
                f"{sec_to_mss(r['pace_per_mile'])}/mi")
    if pd.notna(r['rest_per_mile']) and r['rest_per_mile'] > 0:
        body += f", rest {sec_to_mss(r['rest_per_mile'])}/mi"
    parts = [
        f"<b>{title}</b>{xc_note}",
        body,
        f"<b>P5K projected:</b> {fmt_min(r['p5k_min'])}/mi   "
        f"<b>P5K from CS:</b> {fmt_min(r['p5k_cs_min'])}/mi",
        f"<b>Raw residual:</b> {r['raw_resid']:+.1f}s/mi   "
        f"<b>Corrected:</b> {r['resid']:+.1f}s/mi",
    ]
    return "<br>".join(p for p in parts if p)


def long_run_hover(r):
    title = f"Long{_route_paren(r.get('display_name'), r.get('city_state'))}"
    parts = [
        f"<b>{title}</b>",
        f"<b>Distance:</b> {r['miles']:.1f}mi   "
        f"<b>Pace:</b> {sec_to_mss(r['recovery_pace_sec_per_mi'])}/mi",
        f"<b>P5K projected:</b> {fmt_min(r['p5k_min'])}/mi   "
        f"<b>P5K from CS:</b> {fmt_min(r['p5k_cs_min'])}/mi",
        f"<b>Raw residual:</b> {r['raw_resid']:+.1f}s/mi   "
        f"<b>Corrected:</b> {r['corrected']:+.1f}s/mi",
    ]
    return "<br>".join(p for p in parts if p)


def hill_hover(r):
    title = f"Continuous hills{_route_paren(r.get('loop_display_name'), r.get('loop_city_state'))}"
    nreps = int(r['nreps'])
    loops_word = 'loop' if nreps == 1 else 'loops'
    ft_gained = int(round(float(r.get('ft_gained') or 0)))
    parts = [
        f"<b>{title}</b>",
        f"{nreps} {loops_word}, {ft_gained} ft gained, {int(r['session_min'])} min total",
        f"<b>Actual pace:</b> {sec_to_mss(r['actual_pace_s'])}/mi",
        f"<b>P5K projected:</b> {fmt_min(r['p5k_min'])}/mi   "
        f"<b>P5K from CS:</b> {fmt_min(r['p5k_cs_min'])}/mi",
        f"<b>Raw residual:</b> {r['raw_resid']:+.1f}s/mi   "
        f"<b>Corrected:</b> {r['resid']:+.1f}s/mi",
    ]
    return "<br>".join(p for p in parts if p)


# ---------- main ----------
def main():
    cs, epoch = load_cs()
    # Shared projection helpers return ALL rows with `excluded_reason` flagged
    # for snow / reps / out-of-slice / hc-rep-hybrid / hc-loop-other. Training
    # plot keeps only in-scope rows (excluded_reason is None) and drops the
    # flag column afterwards.
    workouts  = project_workouts(cs, epoch)
    long_runs = project_long_runs(cs, epoch)
    hills     = project_hill_continuous(cs, epoch)
    workouts  = workouts[workouts['excluded_reason'].isna()].drop(columns=['excluded_reason']).copy()
    long_runs = long_runs[long_runs['excluded_reason'].isna()].drop(columns=['excluded_reason']).copy()
    hills     = hills[hills['excluded_reason'].isna()].drop(columns=['excluded_reason']).copy()

    # Long-run model: fit raw_resid ~ C(bin) + C(route) on the in-slice set
    # with iterative MAD-based outlier prune. Outliers are dropped from the
    # figure entirely (not displayed); kept rows carry per-row model offset
    # and corrected residual.
    print(f'\n--- Long-run model: raw_resid ~ bin + route, '
          f'iterative MAD prune (sigma={PRUNE_SIGMA}) ---')
    long_runs, lr_fit, qualifying_routes = fit_long_run_model(long_runs)
    n_in = len(long_runs)
    n_out = int(long_runs['is_outlier'].sum())
    print(f'  In-scope long runs: {n_in}  ({n_out} pruned as outliers)')
    print(f'  Qualifying routes (n>={MIN_ROUTE_N}): {len(qualifying_routes)}')
    for r in qualifying_routes:
        n_r = int((long_runs['route'] == r).sum())
        beta = lr_fit.route_coefs.get(r, 0.0)
        print(f'    {r:<22} n={n_r:<3d}  beta={beta:+6.2f}')
    print(f'  Intercept (bin=lr_hi, route=other): {lr_fit.intercept:+6.2f}')
    for b in ['lr_lo', 'lr_hi']:
        bcoef = lr_fit.bin_coefs.get(b, 0.0)
        n_b = int((long_runs['bin'] == b).sum())
        print(f'  Bin {b} (n={n_b}): coef={bcoef:+6.2f}')
    print(f'  R^2 = {lr_fit.rsquared:.3f}   resid SD = {lr_fit.resid_sd:.2f} sec/mi   '
          f'(n_kept = {lr_fit.n_kept})')
    if n_out:
        print('  Outlier rows pruned:')
        for _, row in long_runs[long_runs['is_outlier']].iterrows():
            print(f'    LR {row["date"].date()}  bin={row["bin"]:<5}  route={row["route"]:<22}  '
                  f'raw={row["raw_resid"]:+6.1f}  corrected={row["corrected"]:+6.1f}  '
                  f'miles={row["miles"]:.1f}')

    # Drop pruned long runs from the working frame — they don't contribute to
    # the smoother and aren't rendered.
    long_runs = long_runs[~long_runs['is_outlier']].copy()

    # Workouts/hills iterative resid-cutoff prune (long runs handled above).
    CUTOFF = 23.3
    print(f'\n--- Iterative resid > +{CUTOFF} prune (workouts/hills) ---')
    pruned_w_idx = set()
    pruned_h_idx = set()
    initial_offsets = None
    for it in range(15):
        w_keep = workouts.drop(index=list(pruned_w_idx))
        h_keep = hills.drop(index=list(pruned_h_idx))
        _, _, offsets = apply_offsets(w_keep, h_keep)
        if initial_offsets is None:
            initial_offsets = offsets
        w_keep = w_keep.copy()
        h_keep = h_keep.copy()
        w_keep['resid'] = w_keep['raw_resid'] - w_keep['category'].map(offsets)
        h_keep['resid'] = h_keep['raw_resid'] - h_keep['category'].map(offsets)

        new_w = w_keep.index[w_keep['resid'] > CUTOFF].tolist()
        new_h = h_keep.index[h_keep['resid'] > CUTOFF].tolist()
        if not new_w and not new_h:
            print(f'  Iteration {it+1}: stable. Done.')
            break

        pruned_w_idx.update(new_w)
        pruned_h_idx.update(new_h)
        print(f'  Iteration {it+1}: +{len(new_w)} workouts, +{len(new_h)} hills')
        for i in new_w:
            r = workouts.loc[i]
            print(f'    W  {r["date"].date()}  {r["category"]:<22} resid={w_keep.loc[i,"resid"]:+5.1f}  '
                  f'raw={r["raw_resid"]:+5.1f}  pace={int(r["pace_per_mile"])}s/mi')
        for i in new_h:
            r = hills.loc[i]
            print(f'    H  {r["date"].date()}  {r["category"]:<22} resid={h_keep.loc[i,"resid"]:+5.1f}  '
                  f'raw={r["raw_resid"]:+5.1f}  loop={r["loop"]}  {int(r["nreps"])}x{int(r["session_min"])}min')

    workouts = workouts.drop(index=list(pruned_w_idx)).copy()
    hills = hills.drop(index=list(pruned_h_idx)).copy()
    workouts, hills, offsets = apply_offsets(workouts, hills)

    print('\n--- Offset shifts (initial -> final) ---')
    for cat in sorted(set(initial_offsets) | set(offsets)):
        i_off = initial_offsets.get(cat, float('nan'))
        f_off = offsets.get(cat, float('nan'))
        print(f'  {cat:<22} {i_off:+6.2f}  ->  {f_off:+6.2f}  (Δ {f_off-i_off:+.2f})')
    print(f'\nKept: {len(workouts)} workouts, {len(long_runs)} long runs, {len(hills)} hills')

    print('Per-category offsets (median raw resid):')
    for c, o in sorted(offsets.items()):
        print(f'  {c:<22} offset={o:+6.2f}')

    # Persist final per-category offsets so the Workouts plot can position
    # markers on the same per-category baseline TQ uses for its smoother.
    offsets_csv = DATA_DIR / 'training_quality_offsets.csv'
    pd.DataFrame({
        'category': list(offsets.keys()),
        'offset_sec_per_mi': [float(v) for v in offsets.values()],
    }).to_csv(offsets_csv, index=False)
    print(f'Wrote {offsets_csv}')

    # Combined for the smoother. Long runs use the model-corrected residual.
    combined = pd.concat([
        workouts[['date', 'resid']],
        long_runs[['date', 'corrected']].rename(columns={'corrected': 'resid'}),
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

    # Persist the smoother track at daily resolution so other plots (Workouts
    # tab) can position hill_rep markers at the TQ smoother y-value on each
    # date. NaN values within the 2020-21 labrum gap (and any other > 90-day
    # gaps) are propagated to the daily output via np.interp; the track CSV
    # consumer treats them as "no smoother value here, skip the marker".
    track_dates_daily = pd.date_range(grid_dates[0], grid_dates[-1], freq='D')
    track_days_daily  = (track_dates_daily - epoch).days.astype(float).values
    track_min_daily = np.full(len(track_dates_daily), np.nan)
    for i, t in enumerate(track_days_daily):
        j = np.searchsorted(grid_days, t)
        if j == 0 or j >= len(grid_days):
            continue
        lv, rv = track[j-1], track[j]
        if np.isnan(lv) or np.isnan(rv):
            continue
        lg, rg = grid_days[j-1], grid_days[j]
        frac = (t - lg) / (rg - lg) if rg != lg else 0.0
        track_min_daily[i] = lv * (1 - frac) + rv * frac
    pd.DataFrame({
        'date': track_dates_daily,
        'p5k_track_min': track_min_daily,
    }).to_csv(TRACK_CSV, index=False)
    print(f'Wrote {TRACK_CSV} ({int(np.isfinite(track_min_daily).sum())} '
          f'finite of {len(track_min_daily)} daily points)')

    # Position each session at CS-implied + corrected residual.
    # `pos_min` is the raw min/mi position (CS line + residual/60), `pos_norm`
    # is the residual in sec/mi used when the "Normalize to CS" toggle is on.
    workouts['pos_min']  = workouts['p5k_cs_min']  + workouts['resid']     / 60.0
    long_runs['pos_min'] = long_runs['p5k_cs_min'] + long_runs['corrected'] / 60.0
    hills['pos_min']     = hills['p5k_cs_min']     + hills['resid']        / 60.0
    workouts['pos_norm']  = workouts['resid']
    long_runs['pos_norm'] = long_runs['corrected']
    hills['pos_norm']     = hills['resid']

    # ---------- build figure ----------
    # Each trace stashes both raw_y (min/mi pace) and norm_y (sec/mi residual
    # from CS) in `meta`. Initial figure renders in normalized mode (the
    # default state of the "Normalize to CS" checkbox); the overlay JS
    # restyles to raw when unchecked.
    def _y_safe(arr):
        return [None if (v is None or (isinstance(v, float) and np.isnan(v)))
                else float(v) for v in arr]

    fig = go.Figure()

    cs_plot = cs[cs['date'] >= pd.Timestamp('2016-01-01')]
    cs_raw  = _y_safe(cs_plot['p5k_implied_min'].values)
    cs_norm = [0.0] * len(cs_plot)
    fig.add_trace(go.Scatter(
        x=cs_plot['date'], y=cs_norm,
        mode='lines', name='CS-implied 5K',
        line=dict(color=CS_LINE, width=2),
        hoverinfo='skip',
        meta={'raw_y': cs_raw, 'norm_y': cs_norm},
    ))

    # Smoother track. NaN positions create the visual break at the
    # 2020-21 labrum gap; both raw_y and norm_y have NaN at the same indices.
    track_raw  = _y_safe(track)
    track_norm = _y_safe(smoothed)
    fig.add_trace(go.Scatter(
        x=grid_dates, y=track_norm,
        mode='lines', name='Training-quality track',
        line=dict(color='#eaeaea', width=2.5),
        connectgaps=False,
        hoverinfo='skip',
        meta={'raw_y': track_raw, 'norm_y': track_norm},
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
        raw_y = _y_safe(sub['pos_min'].values)
        norm_y = _y_safe(sub['pos_norm'].values)
        fig.add_trace(go.Scatter(
            x=sub['date'], y=norm_y,
            mode='markers',
            name=f'{CAT_LABEL[cat]} (n={len(sub)})',
            marker=dict(color=CAT_COLORS[cat], size=7,
                        line=dict(color='rgba(255,255,255,0.4)', width=0.5),
                        opacity=0.85),
            customdata=cd,
            hoverinfo='skip',
            legendgroup='workouts', legendgrouptitle_text='Workouts',
            meta={'snap_eligible': True, 'raw_y': raw_y, 'norm_y': norm_y},
        ))

    # Long runs: one trace per bin
    for cat in ['lr_lo', 'lr_hi']:
        sub = long_runs[long_runs['category'] == cat]
        if sub.empty:
            continue
        cd = [long_run_hover(r) for _, r in sub.iterrows()]
        raw_y = _y_safe(sub['pos_min'].values)
        norm_y = _y_safe(sub['pos_norm'].values)
        fig.add_trace(go.Scatter(
            x=sub['date'], y=norm_y,
            mode='markers',
            name=f'{CAT_LABEL[cat]} (n={len(sub)})',
            marker=dict(color=CAT_COLORS[cat], size=7,
                        line=dict(color='rgba(255,255,255,0.4)', width=0.5),
                        opacity=0.85),
            customdata=cd,
            hoverinfo='skip',
            legendgroup='long_runs', legendgrouptitle_text='Long runs',
            meta={'snap_eligible': True, 'raw_y': raw_y, 'norm_y': norm_y},
        ))

    # Hill continuous: single trace, all loops together
    if len(hills):
        cd = [hill_hover(r) for _, r in hills.iterrows()]
        raw_y = _y_safe(hills['pos_min'].values)
        norm_y = _y_safe(hills['pos_norm'].values)
        fig.add_trace(go.Scatter(
            x=hills['date'], y=norm_y,
            mode='markers',
            name=f'Cont. hills (n={len(hills)})',
            marker=dict(color=CAT_COLORS['hill_lc'], size=7,
                        line=dict(color='rgba(255,255,255,0.4)', width=0.5),
                        opacity=0.85),
            customdata=cd,
            hoverinfo='skip',
            legendgroup='workouts', legendgrouptitle_text='Workouts',
            meta={'snap_eligible': True, 'raw_y': raw_y, 'norm_y': norm_y},
        ))

    # ---------- layout ----------
    # Raw axis: 4:20–6:00 min/mi (descending = faster up).
    y_min_raw, y_max_raw = 4.0 + 20/60, 6.00
    raw_tickvals = [4.0 + (20 + 10 * k) / 60 for k in range(11)]
    raw_ticktext = [sec_to_mss(v * 60) for v in raw_tickvals]

    # Normalized axis: residual sec/mi vs CS, signed labels every 10 s/mi.
    # Range derived from the actual residuals (markers + smoother track) and
    # snapped outward to the next 10 s/mi with a small pad — keeps the CS
    # line (y=0) roughly centered instead of clinging to the top.
    norm_data = np.concatenate([
        workouts['pos_norm'].to_numpy(dtype=float),
        long_runs['pos_norm'].to_numpy(dtype=float),
        hills['pos_norm'].to_numpy(dtype=float),
        smoothed[np.isfinite(smoothed)],
    ])
    pad = 5.0
    norm_lo = int(np.floor((float(np.nanmin(norm_data)) - pad) / 10.0) * 10)
    norm_hi = int(np.ceil((float(np.nanmax(norm_data)) + pad) / 10.0) * 10)
    norm_tickvals = list(range(norm_lo, norm_hi + 1, 10))
    norm_ticktext = ['0' if v == 0 else f'{v:+d}' for v in norm_tickvals]
    norm_axis_range = [norm_hi, norm_lo]

    apply_default_layout(
        fig,
        margin=dict(t=20, l=70,
                    r=right_margin_for_anchored_box(ROUTES_BOX_WIDTH, legend_min_px=220),
                    b=60),
        hovermode=False,
        legend=dict(yanchor='top', y=0.99, xanchor='left', x=1.02,
                    groupclick='toggleitem', font=dict(size=11)),
        xaxis=yearly_x_axis_kwargs(
            pd.Timestamp('2016-01-01'),
            combined['date'].max() + pd.Timedelta(days=30),
            title='Date',
        ),
        yaxis=dict(title='Residual from CS (sec/mi)',
                   range=norm_axis_range,
                   tickmode='array',
                   tickvals=norm_tickvals, ticktext=norm_ticktext,
                   showgrid=True, gridcolor=GRID, zeroline=False),
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

    # ---------- "Normalize to CS" toggle overlay ----------
    # Checkbox sits in the top-right corner. When checked (default), traces
    # render in residual sec/mi from the CS curve flattened at y=0; when
    # unchecked, traces show raw 5K-equivalent pace in min/mi against the
    # actual CS curve. JS reads `meta.raw_y` / `meta.norm_y` from each trace
    # and restyles, then relayouts the y-axis.
    axis_raw = {
        'range': [y_max_raw, y_min_raw],
        'tickvals': raw_tickvals,
        'ticktext': raw_ticktext,
        'title': '5K-equivalent pace (min/mi)',
    }
    axis_norm = {
        'range': norm_axis_range,
        'tickvals': norm_tickvals,
        'ticktext': norm_ticktext,
        'title': 'Residual from CS (sec/mi)',
    }
    # Routes sorted by beta (most negative first); n is per-route count in
    # the kept set (post-prune), matching the recovery-plot convention.
    route_n = long_runs['route'].value_counts().to_dict()
    routes_by_beta = sorted(qualifying_routes,
                            key=lambda r: lr_fit.route_coefs.get(r, 0.0))
    route_rows_html = '\n'.join(
        f'<tr><td>{r}</td><td style="text-align:right">{int(route_n.get(r, 0))}</td>'
        f'<td style="text-align:right">{lr_fit.route_coefs.get(r, 0.0):+.1f}</td></tr>'
        for r in routes_by_beta)

    # Normalize-to-CS toggle (small fixed pill in the upper-right area
    # — distinct from the legend-anchored route table below).
    norm_toggle_html = (
        '<div id="tq-norm-toggle" class="rp-sidebar rp-sidebar-compact" '
        'style="right:60px; top:20px">'
        '<label class="rp-row" style="margin:0">'
        '<input type="checkbox" id="tq-norm-cb" checked> Normalize to CS'
        '</label></div>'
    )
    routes_table = widgets.table(
        ('Route', 'n', 'β'),
        [(r, int(route_n.get(r, 0)),
          f'{lr_fit.route_coefs.get(r, 0.0):+.1f}')
         for r in routes_by_beta],
        align=('left', 'right', 'right'),
    )
    routes_panel = widgets.sidebar(
        'tq-routes',
        body=(
            widgets.title('Long-run route offsets')
            + widgets.subtitle(
                f'vs. baseline (n &ge; {MIN_ROUTE_N}); '
                f'intercept {lr_fit.intercept:+.1f}, '
                f'lr_lo {lr_fit.bin_coefs.get("lr_lo", 0.0):+.1f}')
            + routes_table
        ),
        compact=True,
        width_px=ROUTES_BOX_WIDTH,
    )
    overlay_html = (
        widgets.js_globals({'AXIS_RAW': axis_raw, 'AXIS_NORM': axis_norm})
        + '\n' + norm_toggle_html + '\n' + routes_panel
    )

    render_plot(
        fig, OUT_HTML,
        title_slug='training_quality',
        page_title='Training quality',
        title='Training quality vs. observed race fitness',
        subtitle='5K fitness trend across normalized training data compared to race-derived baseline',
        cursor_tooltip=CursorTooltip(
            payload=payload,
            build_js=build_js,
            first_day=first_day,
            last_day=last_day,
        ),
        overlay_html=overlay_html,
        overlay_js_files=[_TQ_JS],
    )
    print(f'\nWrote {OUT_HTML}')


if __name__ == '__main__':
    main()
