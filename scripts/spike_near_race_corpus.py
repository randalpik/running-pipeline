"""Spike: dump the kept TQ corpus (corrected residuals) for the CS
workout-enrichment near-race analysis.

Replicates plot_training_quality.main()'s filtering exactly (long-run model,
hill model, track-relative prune) but writes the surviving points — with the
same corrected residual the smoother consumes — to
output/debug/spike_cs_enrichment/tq_corpus.csv.

One-off spike tooling (June 2026); safe to delete after the
cs-workout-enrichment decision is recorded.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import src.shared.workouts as workouts_mod
from src.shared.workouts import (
    load_cs, project_workouts, project_long_runs, project_hill_continuous,
)
from src.shared.long_run_model import fit_long_run_model, PRUNE_SIGMA
from src.shared.hill_model import fit_hill_model
from src.plotting import adaptive_gauss_smoother, GAP_BREAK_DAYS

GAUSS_BASE_BW_DAYS = 30
GAUSS_TARGET_ESS   = 12
GAUSS_MAX_BW_DAYS  = 400
GRID_FREQ          = '7D'

OUT_DIR = Path(__file__).resolve().parents[1] / 'output' / 'debug' / 'spike_cs_enrichment'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cs-summary', default='',
                    help='Alternate bayes_cs_summary CSV to project against '
                         '(default: production data/bayes_cs_summary.csv)')
    ap.add_argument('--out', default=str(OUT_DIR / 'tq_corpus.csv'))
    args = ap.parse_args()
    if args.cs_summary:
        workouts_mod.CS_PATH = Path(args.cs_summary)
    cs, epoch = load_cs()
    workouts  = project_workouts(cs, epoch)
    long_runs = project_long_runs(cs, epoch)
    hills     = project_hill_continuous(cs, epoch)
    workouts  = workouts[workouts['excluded_reason'].isna()].copy()
    long_runs = long_runs[long_runs['excluded_reason'].isna()].copy()
    hills     = hills[hills['excluded_reason'].isna()].copy()

    long_runs, lr_fit, _ = fit_long_run_model(long_runs)
    long_runs['model_adj'] = long_runs['phys_contrib'] + long_runs['cov_contrib']

    hills, hill_fit = fit_hill_model(hills)
    hills = hills[~hills['is_outlier']].copy()

    # Track-relative prune, identical to plot_training_quality.main()
    pruned_w_idx, pruned_h_idx, pruned_lr_idx = set(), set(), set()
    for it in range(12):
        w_keep = workouts.drop(index=list(pruned_w_idx)).copy()
        h_keep = hills.drop(index=list(pruned_h_idx)).copy()
        lr_keep = long_runs.drop(index=list(pruned_lr_idx)).copy()
        w_keep['resid'] = w_keep['raw_resid']
        h_keep['resid'] = h_keep['corrected']
        lr_keep['resid'] = lr_keep['raw_resid'] - lr_keep['model_adj']

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

        combined = pd.concat([
            w_keep[['date', 'resid', 'weight']].assign(src='w', orig=w_keep.index),
            lr_keep[['date', 'resid']].assign(weight=lr_w, src='lr', orig=lr_keep.index),
            h_keep[['date', 'resid']].assign(weight=hill_w, src='h', orig=h_keep.index),
        ], ignore_index=True).sort_values('date').reset_index(drop=True)

        ds = (combined['date'] - epoch).dt.days.astype(float).values
        res = combined['resid'].values
        grid_dates = pd.date_range(combined['date'].min(), combined['date'].max(), freq=GRID_FREQ)
        grid_days = (grid_dates - epoch).days.astype(float).values
        smoothed = adaptive_gauss_smoother(
            ds, res, grid_days,
            target_ess=GAUSS_TARGET_ESS, base_bw=GAUSS_BASE_BW_DAYS,
            max_bw=GAUSS_MAX_BW_DAYS, point_weights=combined['weight'].values,
        )
        finite = np.isfinite(smoothed)
        if not finite.any():
            break
        track_at = np.interp(ds, grid_days[finite], smoothed[finite])
        detrended = res - track_at
        med = float(np.median(detrended))
        thr = med + PRUNE_SIGMA * 1.4826 * float(np.median(np.abs(detrended - med)))
        over = detrended > thr
        if not over.any():
            break
        for pos in np.nonzero(over)[0]:
            row = combined.iloc[pos]
            {'w': pruned_w_idx, 'lr': pruned_lr_idx, 'h': pruned_h_idx}[row['src']].add(row['orig'])

    workouts = workouts.drop(index=list(pruned_w_idx)).copy()
    hills = hills.drop(index=list(pruned_h_idx)).copy()
    long_runs = long_runs.drop(index=list(pruned_lr_idx)).copy()
    workouts['resid'] = workouts['raw_resid']
    long_runs['resid'] = long_runs['raw_resid'] - long_runs['model_adj']
    hills['resid'] = hills['corrected']

    # Watch verification status for workouts (mirrors project_workouts gate)
    from src.shared.workouts import _watch_verified_dates, watch_log_demotions
    verified = _watch_verified_dates() - watch_log_demotions()
    workouts['watch_verified'] = workouts['date'].dt.strftime('%Y-%m-%d').isin(verified)

    rows = []
    for _, r in workouts.iterrows():
        rows.append({
            'date': r['date'], 'src': 'workout', 'category': r['category'],
            'resid': r['resid'], 'raw_resid': r['raw_resid'],
            'p5k_min': r['p5k_min'], 'p5k_cs_min': r['p5k_cs_min'],
            'd_eff_m': r['D_eff'], 't_eff_s': r['t_eff'],
            'watch_verified': bool(r['watch_verified']),
            'is_track': bool(r['is_track']),
            'detail': str(r.get('workout_raw', ''))[:60],
        })
    for _, r in long_runs.iterrows():
        rows.append({
            'date': r['date'], 'src': 'long_run', 'category': 'long',
            'resid': r['resid'], 'raw_resid': r['raw_resid'],
            'p5k_min': r['p5k_min'], 'p5k_cs_min': r['p5k_cs_min'],
            'd_eff_m': r['d_m'], 't_eff_s': r['t_run'],
            'watch_verified': bool(r.get('lr_watch', False)),
            'is_track': False,
            'detail': f"{r['miles']:.1f}mi {str(r.get('location',''))[:40]}",
        })
    for _, r in hills.iterrows():
        rows.append({
            'date': r['date'], 'src': 'hill', 'category': r['category'],
            'resid': r['resid'], 'raw_resid': r['raw_resid'],
            'p5k_min': r['p5k_min'], 'p5k_cs_min': r['p5k_cs_min'],
            'd_eff_m': r['d_m'], 't_eff_s': r['t_eff'],
            'watch_verified': bool(r.get('watch_measured', False)),
            'is_track': False,
            'detail': str(r.get('workout_raw', ''))[:60],
        })
    corpus = pd.DataFrame(rows).sort_values('date').reset_index(drop=True)
    out = Path(args.out)
    corpus.to_csv(out, index=False)
    print(f'Wrote {out} ({len(corpus)} points: '
          f"{(corpus['src']=='workout').sum()} workouts, "
          f"{(corpus['src']=='long_run').sum()} long runs, "
          f"{(corpus['src']=='hill').sum()} hills)")
    print(f"Fastest corrected residual: {corpus['resid'].min():+.2f} s/mi on "
          f"{corpus.loc[corpus['resid'].idxmin(), 'date'].date()} "
          f"({corpus.loc[corpus['resid'].idxmin(), 'src']}, "
          f"{corpus.loc[corpus['resid'].idxmin(), 'detail']})")


if __name__ == '__main__':
    main()
