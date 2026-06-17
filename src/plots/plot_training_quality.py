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
from src.shared.plot_window import daily_floor, clip_to_daily_floor
from src.shared.workouts import (
    load_cs, add_cs,
    project_workouts, project_long_runs, project_hill_continuous,
    project_hill_reps,
    WORKOUTS_PATH, DAILY_PATH, CS_PATH,
)
from src.shared.long_run_model import (
    fit_long_run_model, PRUNE_SIGMA,
)
from src.shared.hill_model import fit_hill_model
from src.shared.cs_projection import load_cs_outputs
from src.shared.performance_frontier import standard_demos, build_frontier
from src.plotting import widgets
from src.plotting import (render_plot, CursorTooltip, apply_default_layout,
                            right_margin_for_anchored_box, route_paren,
                            sec_to_mss, fmt_min, CAT_COLORS, GRID, CS_LINE,
                            FRONTIER_LINE,
                            SURFACES, GAP_BREAK_DAYS, adaptive_gauss_smoother,
                            yearly_x_axis_kwargs, nice_time_ticks,
                            nice_time_interval, time_ticks_at_interval)

# Right-rail width reserved next to the plot (sizes margin.r; the long-run
# model table that used to live there is gone — the margin keeps the legend
# clear of the plot area).
ROUTES_BOX_WIDTH = 196

_PLOTS_DIR = Path(__file__).resolve().parent
_TQ_JS = _PLOTS_DIR / 'plot_training_quality.js'


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
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
# Workout categories only — hill categories are runtime-derived per loop
# (hill_<loop>, informational grouping only) and render as one combined
# trace with their own hover title, so they never need registered labels.
CAT_LABEL = {
    'interval': 'Interval', 'tempo': 'Tempo', 'rep': 'Rep',
    'continuous_fartlek': 'Fartlek',
}


# ---------- pipeline ----------
# load_cs, add_cs, project_workouts, project_long_runs, project_hill_continuous,
# HC_LOOPS, HILL_LOOP_META are imported from src.shared.workouts.


# Per-category offsets were REMOVED in June 2026: every quality workout
# (interval / tempo / rep / cont. fartlek) shares one CS predictor with no
# label terms. The old per-category medians (tempo +19.7, cf +17.9) were
# label dummies absorbing ERA EFFORT POLICY — fitted almost entirely on
# 2016-17 threshold tempos and the 2019-20 fartlek block, then mis-applied
# to any modern day wearing the label (a 2024 long-interval day logged as
# tempo got a −19.7 era discount and displayed as the fastest workout in
# the set). Duration and piece-structure terms were tested and rejected:
# broken 3-5min-piece tempos still read +19 vs intervals at the same piece
# length — the gap tracks intent-as-executed per era, which is exactly the
# "training ahead of / behind capability" signal this plot exists to show.
# Same precedent as the long-run route-dummy removal (long_run_model.py).
# There are NO class constants either (a brief global-median centering was
# removed same-day): every point here and on the Workouts tab is the same
# number — the best attempt at predicting 5K race pace from that session.


# ---------- hover string builders ----------
# Per-session HTML for the smart-spikeline scaffold's snap mode and the
# smooth-mode "Nearest session" caption. The scaffold prepends the trend
# section (CS, smoother, diff) and the date itself in smooth mode, so
# this content focuses on what's session-specific. Category label stays
# as a small heading.

# route label/paren now shared (src.plotting.formatters.route_paren).


# Tooltip-only labels (legend uses CAT_LABEL — keeps abbreviated form there).
TOOLTIP_TITLE = {
    'interval': 'Intervals',
}


def residual_line(raw, corrected):
    """One tooltip line for the residual pair. When the correction is a
    no-op at display precision, collapse to a single 'Residual:' figure."""
    if f"{raw:+.1f}" == f"{corrected:+.1f}":
        return f"<b>Residual:</b> {raw:+.1f}s/mi"
    return (f"<b>Raw residual:</b> {raw:+.1f}s/mi   "
            f"<b>Corrected:</b> {corrected:+.1f}s/mi")


def workout_hover(r, single_type=False):
    cat = r['category']
    # Single workout type in the data (watch continuous-fartlek case) -> "Workout".
    title = 'Workout' if single_type else str(TOOLTIP_TITLE.get(cat, CAT_LABEL.get(cat, cat)))
    title += route_paren(r.get('display_name'), r.get('city_state'))
    xc_note = f' <span style="color:{SURFACES["XC"]}">(XC-corrected)</span>' if r.get('xc_corrected') else ''
    rep_count = int(r['rep_count'])
    rep_dist = int(r['rep_dist'])
    # pace_per_mile is log-owned end-to-end (enriched days are normalized to
    # the logged quality pace in parse_workouts).
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
    if pd.notna(r['rest_per_mile']) and r['rest_per_mile'] > 0:
        body += f", rest {sec_to_mss(r['rest_per_mile'])}/mi"
    parts = [
        f"<b>{title}</b>{xc_note}",
        body,
        f"<b>P5K projected:</b> {fmt_min(r['p5k_min'])}/mi   "
        f"<b>P5K from CS:</b> {fmt_min(r['p5k_cs_min'])}/mi",
        residual_line(r['raw_resid'], r['resid']),
    ]
    return "<br>".join(p for p in parts if p)


def lr_correction_line(r):
    """Secondary descriptor for a watch/rule-corrected long run — same role
    as the workouts' 'Watch:' measured_line. Empty string when the row is
    uncorrected. The primary Distance/Pace line shows the corrected values
    (what the projection consumed); this line keeps the logged figures
    visible and says where the correction came from."""
    logged = (f"logged {r['miles']:.1f}mi @ "
              f"{sec_to_mss(r['recovery_pace_sec_per_mi'])}/mi")
    if r.get('lr_watch'):
        pause = ''
        if pd.notna(r.get('pause_s')) and r['pause_s'] >= 30:
            pause = f" · {sec_to_mss(r['pause_s'])} paused"
        return f"<b>Watch:</b> {logged}{pause}"
    if r.get('lr_rule'):
        return f"<b>Mislogged route:</b> {logged}"
    return ''


def long_run_hover(r):
    title = f"Long{route_paren(r.get('display_name'), r.get('city_state'))}"
    # Corrected rows lead with the corrected figures — those are what the
    # P5K projection below consumed; the logged values move to the
    # correction line.
    if pd.notna(r.get('corr_miles')):
        dist_pace = (f"<b>Distance:</b> {r['corr_miles']:.1f}mi   "
                     f"<b>Pace:</b> {sec_to_mss(r['corr_pace_sec_per_mi'])}/mi")
    else:
        dist_pace = (f"<b>Distance:</b> {r['miles']:.1f}mi   "
                     f"<b>Pace:</b> {sec_to_mss(r['recovery_pace_sec_per_mi'])}/mi")
    parts = [
        f"<b>{title}</b>",
        dist_pace,
        lr_correction_line(r),
        f"<b>P5K projected:</b> {fmt_min(r['p5k_min'])}/mi   "
        f"<b>P5K from CS:</b> {fmt_min(r['p5k_cs_min'])}/mi",
        # Race-equivalent raw vs model-adjusted (phys+cov, level untouched).
        residual_line(r['raw_resid'], r['resid']),
    ]
    return "<br>".join(p for p in parts if p)


def hill_rep_hover(r):
    title = f"Hill repeats{route_paren(r.get('loop_display_name'), r.get('loop_city_state'))}"
    n = int(r['rep_count'])
    word = 'rep' if n == 1 else 'reps'
    rt = float(r['rep_time_min'])
    rt_str = f"{int(rt)} min" if rt == int(rt) else f"{int(rt)}:{int(round((rt-int(rt))*60)):02d}"
    parts = [
        f"<b>{title}</b>",
        f"{n} {word} × {rt_str}",
        f"<b>P5K projected:</b> {fmt_min(r['p5k_min'])}/mi   "
        f"<b>P5K from CS:</b> {fmt_min(r['p5k_cs_min'])}/mi",
        residual_line(r['raw_resid'], r['resid']),
    ]
    return "<br>".join(p for p in parts if p)


def hill_hover(r):
    title = f"Continuous hills{route_paren(r.get('loop_display_name'), r.get('loop_city_state'))}"
    nreps = int(r['nreps'])
    loops_word = 'loop' if nreps == 1 else 'loops'
    ft_gained = int(round(float(r.get('ft_gained') or 0)))
    time_part = (f"{sec_to_mss(r['t_eff'])} total"
                 if r.get('watch_measured')
                 else f"{int(r['session_min'])} min total")
    parts = [
        f"<b>{title}</b>",
        f"{nreps} {loops_word}, {ft_gained} ft gained, {time_part}",
        f"<b>Actual pace:</b> {sec_to_mss(r['actual_pace_s'])}/mi",
        f"<b>P5K projected:</b> {fmt_min(r['p5k_min'])}/mi   "
        f"<b>P5K from CS:</b> {fmt_min(r['p5k_cs_min'])}/mi",
        residual_line(r['raw_resid'], r['resid']),
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
    # Watch-era hill reps (measured per-rep structure projected to 5K-equiv in
    # project_hill_reps) feed TQ like any other quality workout. Pre-watch reps
    # (no trustworthy gain -> p5k_min NaN) and snow days do not.
    hill_reps = project_hill_reps(cs, epoch)
    # Record why each quality workout is excluded from Training so the Workouts
    # plot can annotate it (the slow-"outlier" sessions show only there). Snow
    # is a category flag here; outliers are added during the prune below.
    tq_excluded = [{'date': r['date'], 'reason': r['excluded_reason'],
                    'resid': np.nan, 'src': 'workout'}
                   for _, r in workouts[workouts['excluded_reason'].notna()].iterrows()]
    workouts  = workouts[workouts['excluded_reason'].isna()].drop(columns=['excluded_reason']).copy()
    long_runs = long_runs[long_runs['excluded_reason'].isna()].drop(columns=['excluded_reason']).copy()
    hills     = hills[hills['excluded_reason'].isna()].drop(columns=['excluded_reason']).copy()
    if hill_reps.empty:
        hill_reps = hill_reps.assign(raw_resid=pd.Series(dtype=float),
                                     p5k_cs_min=pd.Series(dtype=float))
    else:
        for _, r in hill_reps[hill_reps['p5k_min'].notna()
                              & hill_reps['excluded_reason'].notna()].iterrows():
            tq_excluded.append({'date': r['date'], 'reason': r['excluded_reason'],
                                'resid': np.nan, 'src': 'hill_rep'})
        hill_reps = hill_reps[hill_reps['p5k_min'].notna()
                              & hill_reps['excluded_reason'].isna()].copy()

    # Long-run model, WITHOUT its intercept (Max, June 2026): the projection
    # itself is race-equivalent (β_long un-bias + watch/rule corrections in
    # project_long_runs) and carries no class constant — but the model's
    # physical terms (elevation, altitude) and covariates (temperature,
    # race fatigue) are verified, physically grounded effects, the same
    # family as the hills' Minetti correction, and are ALWAYS applied:
    # resid = raw − (phys + cov). The intercept — the long-run effort level
    # — is never subtracted; that constant would claim a long run
    # out-predicts a race at the same distance/pace. The model's internal
    # MAD prune only robustifies its betas; exclusion from TQ rides the
    # shared track-relative prune below, like every other category.
    print(f'\n--- Long-run model (level NOT applied): raw_resid ~ '
          f'temp/fatigue (elevation priced upstream) ---')
    long_runs, lr_fit, _qualifying_routes = fit_long_run_model(long_runs)
    long_runs['model_adj'] = (long_runs['phys_contrib']
                              + long_runs['cov_contrib'])
    print(f'  In-scope long runs: {len(long_runs)}')
    print(f'  Intercept (fit, NOT subtracted): {lr_fit.intercept:+6.2f}')
    for c, b in lr_fit.phys_coefs.items():
        print(f'  Phys {c:<16} beta={b:+6.2f}')
    for c, b in lr_fit.cov_coefs.items():
        print(f'  Cov {c:<16} beta={b:+6.2f}')
    print(f'  R^2 = {lr_fit.rsquared:.3f}   resid SD = {lr_fit.resid_sd:.2f} '
          f'sec/mi   (n_kept = {lr_fit.n_kept})')

    # Hill correction: pinned Minetti net gain cost (already applied in
    # project_hill_continuous as minetti_resid) + ONE fitted trail term
    # (src/shared/hill_model.py). No intercept — the hill-class effort gap
    # stays visible like tempo-era effort policy. Outliers (egregious easy
    # hill days, one-sided) are dropped from the figure entirely.
    print(f'\n--- Hill model: pinned Minetti net cost + fitted trail term, '
          f'iterative one-sided MAD prune (sigma={PRUNE_SIGMA}) ---')
    hills, hill_fit = fit_hill_model(hills)
    if hill_fit:
        print(f'  Trail term (fitted): {hill_fit["trail_coef"]:+6.2f} s/mi '
              f'(era-confounding caveat documented)')
        print(f'  resid SD = {hill_fit["resid_sd"]:.2f} sec/mi   '
              f'(n_kept = {hill_fit["n_kept"]})')
        n_hout = int(hills['is_outlier'].sum())
        if n_hout:
            print(f'  Easy-day rows pruned ({n_hout}):')
            for _, row in hills[hills['is_outlier']].iterrows():
                tq_excluded.append({'date': row['date'], 'reason': 'easy outlier',
                                    'resid': round(float(row['corrected']), 1),
                                    'src': 'hill'})
                print(f'    H  {row["date"].date()}  loop={row["loop"]:<6} '
                      f'raw={row["raw_resid"]:+6.1f}  '
                      f'corrected={row["corrected"]:+6.1f}')
        hills = hills[~hills['is_outlier']].copy()
        # Persist the trail term so the Workouts plot shares the correction
        # (the Minetti factor is recomputed from loop covariates there).
        hm_csv = DATA_DIR / 'hill_model.csv'
        pd.DataFrame({
            'term': ['is_trail'],
            'coef': [hill_fit['trail_coef']],
        }).to_csv(hm_csv, index=False)
        print(f'Wrote {hm_csv}')
    else:
        hills['corrected'] = pd.Series(dtype=float)

    # Track-relative prune + smoother, iterated to a fixed point. Outliers
    # are judged against the SURROUNDING track, not against CS or a label
    # baseline: fit the smoother, detrend every prunable point by the track
    # value at its date, drop the slow side beyond median+PRUNE_SIGMA*MAD of
    # the detrended residuals, refit. Era effort policy stays in (a soft
    # 2016 tempo sits near the soft 2016 track and survives); a session that
    # sticks out from its own surroundings goes.
    workouts = workouts.copy()
    rep_w = hill_w = lr_w = hr_w = 1.0
    thr = float('nan')
    pruned_w_idx, pruned_h_idx, pruned_lr_idx = set(), set(), set()
    pruned_hr_idx = set()
    print(f'\n--- Track-relative prune (one-sided, '
          f'median+{PRUNE_SIGMA}*MAD on detrended residuals) ---')
    for it in range(12):
        w_keep = workouts.drop(index=list(pruned_w_idx)).copy()
        h_keep = hills.drop(index=list(pruned_h_idx)).copy()
        lr_keep = long_runs.drop(index=list(pruned_lr_idx)).copy()
        hr_keep = hill_reps.drop(index=list(pruned_hr_idx)).copy()
        # No class constants anywhere: every point on this graph AND the
        # Workouts tab is the same number — the best attempt at predicting
        # 5K race pace from that session (Max's contract). Workouts enter
        # raw; hills enter Minetti+trail-corrected raw; long runs enter
        # race-equivalent raw (β_long in project_long_runs) minus the
        # long-run model's physical+covariate terms, level untouched.
        w_keep['resid'] = w_keep['raw_resid']
        h_keep['resid'] = h_keep['corrected']
        lr_keep['resid'] = lr_keep['raw_resid'] - lr_keep['model_adj']
        # Hill reps enter at their grade-adjusted raw residual (the Minetti
        # one-way + CP3 projection already handles the climb; no trail term).
        hr_keep['resid'] = hr_keep['raw_resid']

        # Scatter weights (reps, hills, and long runs are noisier CS
        # signals; same (sd_ref/sd)^2 construction as always), recomputed
        # on survivors.
        rep_mask = w_keep['category'] == 'rep'
        sd_ref = w_keep.loc[~rep_mask, 'resid'].std()
        rep_w = 1.0
        if rep_mask.any():
            sd_rep = w_keep.loc[rep_mask, 'resid'].std()
            if sd_rep and sd_rep > 0 and sd_ref and sd_ref > 0:
                rep_w = float(np.clip((sd_ref / sd_rep) ** 2, 0.1, 1.0))
        w_keep['weight'] = np.where(rep_mask, rep_w, 1.0)
        hill_w = 1.0
        if len(h_keep):
            sd_hill = h_keep['resid'].std()
            if sd_hill and sd_hill > 0 and sd_ref and sd_ref > 0:
                hill_w = float(np.clip((sd_ref / sd_hill) ** 2, 0.1, 1.0))
        lr_w = 1.0
        if len(lr_keep):
            sd_lr = lr_keep['resid'].std()
            if sd_lr and sd_lr > 0 and sd_ref and sd_ref > 0:
                lr_w = float(np.clip((sd_ref / sd_lr) ** 2, 0.1, 1.0))
        hr_w = 1.0
        if len(hr_keep):
            sd_hr = hr_keep['resid'].std()
            if sd_hr and sd_hr > 0 and sd_ref and sd_ref > 0:
                hr_w = float(np.clip((sd_ref / sd_hr) ** 2, 0.1, 1.0))

        combined = pd.concat([
            w_keep[['date', 'resid', 'weight']].assign(src='w', orig=w_keep.index),
            lr_keep[['date', 'resid']].assign(weight=lr_w, src='lr',
                                              orig=lr_keep.index),
            h_keep[['date', 'resid']].assign(weight=hill_w, src='h',
                                             orig=h_keep.index),
            hr_keep[['date', 'resid']].assign(weight=hr_w, src='hr',
                                              orig=hr_keep.index),
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
            point_weights=combined['weight'].values,
        )

        finite = np.isfinite(smoothed)
        if not finite.any():
            print('  Track all-NaN (tiny corpus) — prune skipped.')
            break
        track_at = np.interp(ds, grid_days[finite], smoothed[finite])
        detrended = res - track_at
        vals = detrended
        med = float(np.median(vals))
        thr = med + PRUNE_SIGMA * 1.4826 * float(np.median(np.abs(vals - med)))
        over = detrended > thr
        if not over.any():
            print(f'  Iteration {it+1}: stable (thr +{thr:.1f} vs track). Done.')
            break
        print(f'  Iteration {it+1}: +{int(over.sum())} pruned (thr +{thr:.1f})')
        for pos in np.nonzero(over)[0]:
            row = combined.iloc[pos]
            if row['src'] == 'w':
                r = workouts.loc[row['orig']]
                pruned_w_idx.add(row['orig'])
                tq_excluded.append({'date': r['date'], 'reason': 'outlier',
                                    'resid': round(float(detrended[pos]), 1),
                                    'src': 'workout'})
                print(f'    W  {r["date"].date()}  {r["category"]:<20} '
                      f'vs-track={detrended[pos]:+5.1f}  raw={r["raw_resid"]:+5.1f}  '
                      f'pace={int(r["pace_per_mile"])}s/mi')
            elif row['src'] == 'lr':
                r = long_runs.loc[row['orig']]
                pruned_lr_idx.add(row['orig'])
                tq_excluded.append({'date': r['date'], 'reason': 'outlier',
                                    'resid': round(float(detrended[pos]), 1),
                                    'src': 'long_run'})
                print(f'    LR {r["date"].date()}  {r["location"]:<20} '
                      f'vs-track={detrended[pos]:+5.1f}  raw={r["raw_resid"]:+5.1f}  '
                      f'miles={r["miles"]:.1f}')
            elif row['src'] == 'hr':
                r = hill_reps.loc[row['orig']]
                pruned_hr_idx.add(row['orig'])
                tq_excluded.append({'date': r['date'], 'reason': 'outlier',
                                    'resid': round(float(detrended[pos]), 1),
                                    'src': 'hill_rep'})
                print(f'    HR {r["date"].date()}  loop={r["loop"]:<6} '
                      f'vs-track={detrended[pos]:+5.1f}  raw={r["raw_resid"]:+5.1f}')
            else:
                r = hills.loc[row['orig']]
                pruned_h_idx.add(row['orig'])
                tq_excluded.append({'date': r['date'], 'reason': 'outlier',
                                    'resid': round(float(detrended[pos]), 1),
                                    'src': 'hill'})
                print(f'    H  {r["date"].date()}  loop={r["loop"]:<6} '
                      f'vs-track={detrended[pos]:+5.1f}  raw={r["raw_resid"]:+5.1f}')

    workouts = workouts.drop(index=list(pruned_w_idx)).copy()
    hills = hills.drop(index=list(pruned_h_idx)).copy()
    long_runs = long_runs.drop(index=list(pruned_lr_idx)).copy()
    hill_reps = hill_reps.drop(index=list(pruned_hr_idx)).copy()
    workouts['resid'] = workouts['raw_resid']
    long_runs['resid'] = long_runs['raw_resid'] - long_runs['model_adj']
    hill_reps['resid'] = hill_reps['raw_resid']
    rep_mask = workouts['category'] == 'rep'
    workouts['weight'] = np.where(rep_mask, rep_w, 1.0)
    if len(hills):
        hills['resid'] = hills['corrected']
    else:
        hills['resid'] = pd.Series(dtype=float)

    print(f'\nKept: {len(workouts)} workouts, {len(long_runs)} long runs, '
          f'{len(hills)} hills')
    print(f'  Reps scatter-weight {rep_w:.2f}; hills pooled weight {hill_w:.2f}; '
          f'long runs pooled weight {lr_w:.2f}')
    print('Per-category median resid (diagnostic only — effort policy, '
          'NOT corrected):')
    for c, g in workouts.groupby('category')['resid']:
        print(f'  {c:<22} {g.median():+6.2f}  (n={len(g)})')

    # Persist which sessions Training excluded (snow flag + residual
    # outliers; src distinguishes workouts from hills), so the Workouts plot
    # can annotate them in hover. cutoff is the track-relative prune
    # threshold; resid is the vs-track residual ('outlier') or the hill
    # model's corrected residual ('easy outlier').
    excl_csv = DATA_DIR / 'training_quality_exclusions.csv'
    excl_df = pd.DataFrame(tq_excluded, columns=['date', 'reason', 'resid', 'src'])
    excl_df['cutoff'] = round(thr, 1)
    excl_df.to_csv(excl_csv, index=False)
    print(f'Wrote {excl_csv}  ({len(excl_df)} excluded workouts)')

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

    p5k_at_grid = np.interp(grid_days, cs['day'].to_numpy(), cs['p5k_implied_min'].to_numpy())
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
    workouts['pos_min']  = workouts['p5k_cs_min']  + workouts['resid']  / 60.0
    long_runs['pos_min'] = long_runs['p5k_cs_min'] + long_runs['resid'] / 60.0
    hills['pos_min']     = hills['p5k_cs_min']     + hills['resid']     / 60.0
    hill_reps['pos_min'] = hill_reps['p5k_cs_min'] + hill_reps['resid'] / 60.0

    # Persist the kept corpus (post-filter, post-prune, corrected residuals)
    # as a data artifact. Consumer: the performance frontier on the Fitness
    # tab (src/shared/performance_frontier.py) — every kept point is a
    # demonstration of 5K capability at p5k_corr_min. run_plots.sh runs this
    # script before bayes_cs_plot so the artifact is fresh.
    def _lr_detail(r):
        name = r.get('display_name')
        if pd.isna(name) or not str(name).strip():
            name = r.get('location', '')
        return f"{r['miles']:.1f}mi {name}"

    def _wk_detail(r):
        # The Fitness tab shows this as the workout's only descriptor, so use
        # the rep DECOMPOSITION (the same `structure` the Workouts tab renders,
        # e.g. "2400 + 3×1200 + …"), NOT the raw coded workout string. Falls
        # back to the rep count × distance when no structure was decomposed.
        s = r.get('structure')
        if isinstance(s, str) and s.strip():
            return s
        rc, rd = int(r['rep_count']), int(r['rep_dist'])
        return f"{rd}m" if rc == 1 else f"{rc} × {rd}m"
    corpus = pd.concat([
        pd.DataFrame({'date': workouts['date'], 'src': 'workout',
                      'category': workouts['category'],
                      'p5k_corr_min': workouts['pos_min'],
                      'detail': [_wk_detail(r) for _, r in workouts.iterrows()]}),
        pd.DataFrame({'date': long_runs['date'], 'src': 'long_run',
                      'category': 'long',
                      'p5k_corr_min': long_runs['pos_min'],
                      'detail': [_lr_detail(r) for _, r in long_runs.iterrows()]}),
        pd.DataFrame({'date': hills['date'], 'src': 'hill',
                      'category': hills['category'],
                      'p5k_corr_min': hills['pos_min'],
                      'detail': hills['workout_raw'].astype(str)}),
        pd.DataFrame({'date': hill_reps['date'], 'src': 'hill_rep',
                      'category': 'hillrep_' + hill_reps['loop'].astype(str),
                      'p5k_corr_min': hill_reps['pos_min'],
                      'detail': hill_reps['workout_raw'].astype(str)}),
    ], ignore_index=True).sort_values('date').reset_index(drop=True)
    corpus_csv = DATA_DIR / 'training_quality_corpus.csv'
    corpus.to_csv(corpus_csv, index=False)
    print(f'Wrote {corpus_csv} ({len(corpus)} kept points)')

    # Performance frontier (red line), same canonical construction as the
    # Fitness tab — corpus passed in-memory (this script just built it).
    daily_summary, beta_long, d_thresh, xc_corr = load_cs_outputs(str(DATA_DIR))
    corpus_demos = corpus.rename(columns={'p5k_corr_min': 'pace_min'})
    demos = standard_demos(daily_summary, beta_long, d_thresh, xc_corr,
                           corpus=corpus_demos)
    front_plot = clip_to_daily_floor(daily_summary).copy()
    frontier, _ = build_frontier(demos, pd.DatetimeIndex(front_plot['date']),
                                 front_plot['p5k_implied_min'])
    print(f'Frontier: computed over {len(front_plot)} daily points')
    workouts['pos_norm']  = workouts['resid']
    long_runs['pos_norm'] = long_runs['resid']
    hills['pos_norm']     = hills['resid']
    hill_reps['pos_norm'] = hill_reps['resid']

    # ---------- build figure ----------
    # Each trace stashes both raw_y (min/mi pace) and norm_y (sec/mi residual
    # from CS) in `meta`. Initial figure renders in normalized mode (the
    # default state of the "Normalize to CS" checkbox); the overlay JS
    # restyles to raw when unchecked.
    def _y_safe(arr):
        return [None if (v is None or (isinstance(v, float) and np.isnan(v)))
                else float(v) for v in arr]

    fig = go.Figure()

    cs_plot = clip_to_daily_floor(cs)
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

    # Performance frontier: normalized = excess vs the CS-implied 5K
    # (<= 0, bulges below the zero line = demonstrated capability beyond
    # CS); raw = the frontier pace itself.
    front_raw = _y_safe(frontier['frontier_pace_min'].values)
    front_norm = _y_safe((frontier['frontier_pace_min'].to_numpy(float)
                          - front_plot['p5k_implied_min'].to_numpy(float)) * 60.0)
    fig.add_trace(go.Scatter(
        x=front_plot['date'], y=front_norm,
        mode='lines', name='Performance frontier',
        line=dict(color=FRONTIER_LINE, width=2),
        connectgaps=False,
        hoverinfo='skip',
        meta={'raw_y': front_raw, 'norm_y': front_norm},
    ))

    # Workouts: one trace per category for legend filtering (reps excluded
    # above). Per-marker customdata + meta.snap_eligible feeds the smart
    # spikeline scaffold's snap mode — hover near a marker shows that
    # session's details, hover anywhere else shows the per-day smooth
    # tooltip (CS pace + smoother trend + nearest session).
    # Single-type collapse: one workout/hill category present (watch CF case)
    # -> one generic "Workout" legend line. Category + CS analysis unchanged.
    present_cats = [c for c in ['interval', 'tempo', 'rep', 'continuous_fartlek']
                    if not workouts[workouts['category'] == c].empty]
    single_type = (len(present_cats) + (1 if len(hills) else 0)) == 1
    for cat in ['interval', 'tempo', 'rep', 'continuous_fartlek']:
        sub = workouts[workouts['category'] == cat]
        if sub.empty:
            continue
        cd = [workout_hover(r, single_type) for _, r in sub.iterrows()]
        raw_y = _y_safe(sub['pos_min'].values)
        norm_y = _y_safe(sub['pos_norm'].values)
        fig.add_trace(go.Scatter(
            x=sub['date'], y=norm_y,
            mode='markers',
            name=(f'Workout (n={len(sub)})' if single_type
                  else f'{CAT_LABEL[cat]} (n={len(sub)})'),
            marker=dict(color=CAT_COLORS[cat], size=7,
                        line=dict(color='rgba(255,255,255,0.4)', width=0.5),
                        opacity=0.85),
            customdata=cd,
            hoverinfo='skip',
            meta={'snap_eligible': True, 'raw_y': raw_y, 'norm_y': norm_y},
        ))

    # Long runs: single trace (distance carries no model coefficient and,
    # as of June 2026, no legend split either).
    if len(long_runs):
        cd = [long_run_hover(r) for _, r in long_runs.iterrows()]
        raw_y = _y_safe(long_runs['pos_min'].values)
        norm_y = _y_safe(long_runs['pos_norm'].values)
        fig.add_trace(go.Scatter(
            x=long_runs['date'], y=norm_y,
            mode='markers',
            name=f'Long (n={len(long_runs)})',
            marker=dict(color=CAT_COLORS['long'], size=7,
                        line=dict(color='rgba(255,255,255,0.4)', width=0.5),
                        opacity=0.85),
            customdata=cd,
            hoverinfo='skip',
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
            marker=dict(color=CAT_COLORS['hill_cont'], size=7,
                        line=dict(color='rgba(255,255,255,0.4)', width=0.5),
                        opacity=0.85),
            customdata=cd,
            hoverinfo='skip',
            meta={'snap_eligible': True, 'raw_y': raw_y, 'norm_y': norm_y},
        ))

    # Hill repeats: watch-era only (the projected ones), their own trace.
    if len(hill_reps):
        cd = [hill_rep_hover(r) for _, r in hill_reps.iterrows()]
        raw_y = _y_safe(hill_reps['pos_min'].values)
        norm_y = _y_safe(hill_reps['pos_norm'].values)
        fig.add_trace(go.Scatter(
            x=hill_reps['date'], y=norm_y,
            mode='markers',
            name=f'Hill reps (n={len(hill_reps)})',
            marker=dict(color=CAT_COLORS['hill_rep'], size=7,
                        line=dict(color='rgba(255,255,255,0.4)', width=0.5),
                        opacity=0.85),
            customdata=cd,
            hoverinfo='skip',
            meta={'snap_eligible': True, 'raw_y': raw_y, 'norm_y': norm_y},
        ))

    # ---------- layout ----------
    # Raw axis: actual pace (min/mi), descending = faster up. Data-driven over
    # the markers + CS line; target=10 reproduces the former 4:20–6:00 / 10 s/mi
    # look for Max and adapts to any profile's range.
    raw_all = np.concatenate([
        workouts['pos_min'].to_numpy(dtype=float),
        long_runs['pos_min'].to_numpy(dtype=float),
        hills['pos_min'].to_numpy(dtype=float),
        hill_reps['pos_min'].to_numpy(dtype=float),
        cs_plot['p5k_implied_min'].to_numpy(dtype=float),
        frontier['frontier_pace_min'].to_numpy(dtype=float),
    ])
    raw_all = raw_all[np.isfinite(raw_all)]
    _rlo, _rhi = (float(raw_all.min()), float(raw_all.max())) if len(raw_all) else (4.0 + 20/60, 6.0)
    _raw_sec, raw_ticktext = nice_time_ticks(_rlo * 60, _rhi * 60, target=10)
    raw_tickvals = [t / 60.0 for t in _raw_sec]
    y_min_raw, y_max_raw = raw_tickvals[0], raw_tickvals[-1]

    # Normalized axis: residual sec/mi vs CS, signed labels. Interval picked by
    # the shared ladder (target=9 → 10 s/mi over Max's residual span); 0 stays
    # on a gridline since ticks are multiples of the interval.
    norm_data = np.concatenate([
        workouts['pos_norm'].to_numpy(dtype=float),
        long_runs['pos_norm'].to_numpy(dtype=float),
        hills['pos_norm'].to_numpy(dtype=float),
        hill_reps['pos_norm'].to_numpy(dtype=float),
        smoothed[np.isfinite(smoothed)],
        np.asarray([v for v in ((frontier['frontier_pace_min'].to_numpy(float)
                                 - front_plot['p5k_implied_min'].to_numpy(float))
                                * 60.0) if np.isfinite(v)], dtype=float),
    ])
    _nlo, _nhi = float(np.nanmin(norm_data)), float(np.nanmax(norm_data))
    _niv = nice_time_interval(_nlo, _nhi, target=9)
    _nvals, _ = time_ticks_at_interval(_nlo, _nhi, _niv)
    norm_tickvals = [int(round(v)) for v in _nvals]
    norm_ticktext = ['0' if v == 0 else f'{v:+d}' for v in norm_tickvals]
    norm_lo, norm_hi = norm_tickvals[0], norm_tickvals[-1]
    norm_axis_range = [norm_hi, norm_lo]

    apply_default_layout(
        fig,
        margin=dict(t=20, l=70,
                    r=right_margin_for_anchored_box(ROUTES_BOX_WIDTH, legend_min_px=220),
                    b=28),
        hovermode=False,
        legend=dict(yanchor='top', y=0.99, xanchor='left', x=1.02,
                    groupclick='toggleitem', font=dict(size=11)),
        xaxis=yearly_x_axis_kwargs(
            daily_floor(),
            combined['date'].max() + pd.Timedelta(days=30),
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

    plot_start = daily_floor()
    plot_end   = combined['date'].max() + pd.Timedelta(days=30)
    all_days   = pd.date_range(plot_start, plot_end, freq='D')

    target_days_2016 = (all_days - epoch).days.astype(float).values
    cs_pace_per_day = np.interp(target_days_2016,
                                cs['day'].to_numpy(),
                                cs['p5k_implied_min'].to_numpy())

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
    for _, r in hill_reps.iterrows():
        sessions.append({'day': int((r['date'] - js_epoch).days),
                         'html': hill_rep_hover(r)})
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
    # Normalize-to-CS toggle (small fixed pill in the upper-right area
    # — distinct from the legend-anchored model table below).
    norm_toggle_html = (
        '<div id="tq-norm-toggle" class="rp-sidebar rp-sidebar-compact" '
        'style="right:60px; top:20px">'
        '<label class="rp-row" style="margin:0">'
        '<input type="checkbox" id="tq-norm-cb" checked> Normalize to CS'
        '</label></div>'
    )
    # Long-run adjustments box: documents EVERYTHING subtracted from a long
    # run to reach its flat / sea-level race-equivalent. Two groups —
    # PHYSICAL route costs (grade per-run from the watch; footing + altitude
    # pinned from the pooled recovery+long fit; all applied upstream in
    # project_long_runs, so already in raw_resid) and TRAINING STATE
    # (temp/fatigue, fit here, applied via model_adj). The effort level
    # (intercept) is fit but NEVER subtracted (see the model block in main).
    from src.shared.recovery_model import physical_route_betas
    from src.shared.elevation_cost import CLIMB_COST, REFUND_RECOVERY
    pb = physical_route_betas()
    phys_rows = [
        ('elevation, per ft/mi↑',
         f'{CLIMB_COST["paved"]:.2f}–{CLIMB_COST["mixed"]:.2f}'),
        ('off-road footing', f'{pb["is_offroad"]:+.1f}'),
        ('altitude, per 1000ft', f'{pb["alt_kft"]:+.2f}'),
    ]
    state_rows = []
    if lr_fit.cov_coefs:
        state_rows = [
            ('temp, per °C felt (ref 12)',
             f'{lr_fit.cov_coefs["temp_centered"]:+.2f}'),
            ('marathon fatigue, peak',
             f'{lr_fit.cov_coefs["fat_marathon"]:+.1f}'),
            ('short-race fatigue, peak',
             f'{lr_fit.cov_coefs["fat_race_short"]:+.1f}'),
        ]
    show_box = bool(state_rows) or any(abs(pb[k]) > 1e-9 for k in pb)
    routes_panel = ''
    if show_box:
        body = (
            widgets.title('Long-run adjustments (sec/mi)')
            + widgets.subtitle(
                'Subtracted to reach each run\'s flat / sea-level '
                'race-equivalent; the effort level (intercept '
                f'{lr_fit.intercept:+.0f}) is never applied.')
            + widgets.divider()
            + widgets.subtitle('Physical route — per-run measured grade '
                               f'(descent refunds {REFUND_RECOVERY["paved"]:.0%} '
                               f'paved / {REFUND_RECOVERY["mixed"]:.0%} off-road); '
                               'footing + altitude pinned from recovery+long')
            + widgets.table(('Term', 's/mi'), phys_rows, align=('left', 'right'))
        )
        if state_rows:
            body += (
                widgets.divider()
                + widgets.subtitle('Training state — fit on long runs')
                + widgets.table(('Term', 'β'), state_rows,
                                align=('left', 'right'))
            )
        routes_panel = widgets.sidebar(
            'tq-routes', body=body, compact=True, width_px=ROUTES_BOX_WIDTH,
        )
    overlay_html = (
        widgets.js_globals({'AXIS_RAW': axis_raw, 'AXIS_NORM': axis_norm})
        + '\n' + norm_toggle_html + '\n' + routes_panel
    )

    render_plot(
        fig, OUT_HTML,
        title_slug='training_quality',
        page_title='Training quality',
        title='Training quality vs. observed fitness',
        subtitle='5K fitness trend across normalized training data compared to performance-derived baseline and frontier',
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
