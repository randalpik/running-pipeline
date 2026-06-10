"""Long-run model variant experiment — REPORT ONLY, nothing here ships.

Compares Stage 5b candidates against the production long-run model
(``raw_resid ~ bin + route``, src/shared/long_run_model.py) to see whether
the TQ graph's long-run noise (±30 s/mi scatter around CS) can be reduced:

  V0   bin(21mi) + route                      — current model, regression guard
  V1a  miles(linear) + route                  — continuous distance, no bin
  V1b  miles + hinge(miles−21) + route        — piecewise-linear at the bin point
  V2   bin + route + temp/fatigue/TOD         — covariates fit FRESH on long runs
  V3   bin + route on (raw − recovery-sourced temp/fatigue/TOD contribution)

V2 vs V3 is the headline comparison: same covariates, betas estimated from
long runs themselves (V2) vs transferred from the recovery fit (V3). V3's
AIC is NOT comparable to V0–V2 (it fits a different y); compare V3 on
residual SD and the graph-noise metrics instead.

Every variant re-runs the production iterative MAD prune (PRUNE_SIGMA on the
corrected residuals), matching what would actually happen if the variant
landed. A common-kept-set block (rows kept by ALL variants) is reported for
apples-to-apples SD comparison.

Graph-noise metrics rebuild the TQ combined smoother per variant — the
workouts/hills pipeline (per-category median offsets + iterative +23.3 prune)
is replicated from plot_training_quality.py and HELD FIXED; only the long-run
corrected residuals change:
  noise_pts  = mean |LR corrected − smoothed track at the LR's date|
  roughness  = mean |Δ track| per 7-day grid step (gap-broken at 90 days)

Run (max profile, the only one with enough long runs):
  python -m src.analysis.lr_variant_experiment

Writes metrics CSV + per-variant beta table + PNG panels to output/debug/.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.shared.paths import DATA_DIR, DEBUG_DIR
from src.shared.plot_window import daily_floor
from src.shared.workouts import (load_cs, project_workouts, project_long_runs,
                                 project_hill_continuous, LR_INTERNAL_BIN)
from src.shared.long_run_model import MIN_ROUTE_N, PRUNE_SIGMA
from src.shared.cs_projection import load_cs_outputs
from src.shared.recovery_model import (fit_recovery_model,
                                       transferable_contributions,
                                       add_quality_features, tod_is_pm,
                                       TEMP_REFERENCE_C, QUALITY_CATS)
from src.plotting.smoothing import (adaptive_gauss_smoother, GAP_BREAK_DAYS)

# Smoother parameters — match plot_training_quality.py.
GAUSS_BASE_BW_DAYS = 30
GAUSS_TARGET_ESS   = 12
GAUSS_MAX_BW_DAYS  = 400
GRID_FREQ          = '7D'
WH_CUTOFF          = 23.3   # workouts/hills iterative prune cutoff


# ---------- generic OLS + iterative MAD prune (mirrors fit_long_run_model) ----------

def fit_pruned_ols(lr_in, X, y):
    """OLS with the production iterative MAD-based outlier prune.

    ``X`` is a DataFrame whose first column is the intercept; ``y`` a Series
    aligned with ``lr_in``. Returns coefficient map, per-row corrected
    residual (y − prediction, including pruned rows), kept mask, and fit
    metrics computed on the kept rows.
    """
    pruned: set = set()
    coef = np.zeros(X.shape[1])
    for _ in range(10):
        keep_mask = ~lr_in.index.isin(pruned)
        Xa = X.values[keep_mask]
        ya = y.values[keep_mask]
        coef, *_ = np.linalg.lstsq(Xa, ya, rcond=None)
        pred_full = X.values @ coef
        full_resid = y.values - pred_full
        active_resid = full_resid[keep_mask]
        center = float(np.median(active_resid))
        sd_robust = 1.4826 * float(np.median(np.abs(active_resid - center)))
        new_pruned = set(lr_in.index[np.abs(full_resid - center)
                                     > PRUNE_SIGMA * sd_robust])
        if new_pruned == pruned:
            break
        pruned = new_pruned

    keep_mask = ~lr_in.index.isin(pruned)
    corrected = y.values - X.values @ coef
    ya = y.values[keep_mask]
    pa = (X.values @ coef)[keep_mask]
    ss_res = float(np.sum((ya - pa) ** 2))
    ss_tot = float(np.sum((ya - ya.mean()) ** 2))
    n_kept = int(keep_mask.sum())
    k = X.shape[1]
    return SimpleNamespace(
        coefs={str(c): float(b) for c, b in zip(X.columns, coef)},
        corrected=corrected,
        keep_mask=np.asarray(keep_mask),
        rsquared=1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan'),
        resid_sd=float(np.sqrt(ss_res / (n_kept - k))) if n_kept > k else float('nan'),
        n_kept=n_kept,
        k=k,
        # AIC on the kept rows, k = columns of X (the prune is identical
        # across variants in form, so it cancels in ΔAIC).
        aic=n_kept * np.log(ss_res / n_kept) + 2 * k,
    )


# ---------- workouts/hills baseline (replicated from plot_training_quality) ----------

def apply_offsets(workouts, hills):
    parts = [workouts[['date', 'category', 'raw_resid']]]
    if hills is not None and len(hills):
        parts.append(hills[['date', 'category', 'raw_resid']])
    combined = pd.concat(parts, ignore_index=True)
    offsets = combined.groupby('category')['raw_resid'].median().to_dict()
    workouts = workouts.copy()
    workouts['resid'] = workouts['raw_resid'] - workouts['category'].map(offsets)
    hills = hills.copy()
    hills['resid'] = hills['raw_resid'] - hills['category'].map(offsets)
    return workouts, hills


def workouts_hills_resid(cs, epoch):
    """Workouts + hills with final per-category-offset residuals, after the
    iterative resid > +23.3 prune — the fixed (non-long-run) smoother input."""
    workouts = project_workouts(cs, epoch)
    hills = project_hill_continuous(cs, epoch)
    workouts = workouts[workouts['excluded_reason'].isna()].copy()
    hills = hills[hills['excluded_reason'].isna()].copy()

    pruned_w, pruned_h = set(), set()
    for _ in range(15):
        w_keep = workouts.drop(index=list(pruned_w))
        h_keep = hills.drop(index=list(pruned_h))
        w_keep, h_keep = apply_offsets(w_keep, h_keep)
        new_w = w_keep.index[w_keep['resid'] > WH_CUTOFF].tolist()
        new_h = h_keep.index[h_keep['resid'] > WH_CUTOFF].tolist()
        if not new_w and not new_h:
            break
        pruned_w.update(new_w)
        pruned_h.update(new_h)
    workouts = workouts.drop(index=list(pruned_w))
    hills = hills.drop(index=list(pruned_h))
    workouts, hills = apply_offsets(workouts, hills)
    return workouts[['date', 'resid']], hills[['date', 'resid']]


# ---------- graph-noise metrics ----------

def smoother_metrics(wh_resid, lr_dates, lr_corrected, epoch):
    """Rebuild the TQ combined smoother with this variant's long-run
    corrections; return (noise_pts, roughness, grid_dates, track)."""
    combined = pd.concat([
        wh_resid,
        pd.DataFrame({'date': lr_dates, 'resid': lr_corrected}),
    ], ignore_index=True).sort_values('date').reset_index(drop=True)

    ds = (combined['date'] - epoch).dt.days.astype(float).values
    res = combined['resid'].values
    grid_dates = pd.date_range(combined['date'].min(), combined['date'].max(),
                               freq=GRID_FREQ)
    grid_days = (grid_dates - epoch).days.astype(float).values
    smoothed = adaptive_gauss_smoother(
        ds, res, grid_days,
        target_ess=GAUSS_TARGET_ESS,
        base_bw=GAUSS_BASE_BW_DAYS,
        max_bw=GAUSS_MAX_BW_DAYS,
    )
    # Gap-break (the 2020–21 labrum gap) so roughness isn't dominated by
    # interpolation across a no-data window.
    sorted_input = combined['date'].sort_values().reset_index(drop=True)
    diffs = sorted_input.diff().dt.days
    for idx in diffs[diffs > GAP_BREAK_DAYS].index:
        mask = (grid_dates > sorted_input[idx - 1]) & (grid_dates < sorted_input[idx])
        smoothed[mask] = np.nan

    lr_days = (pd.to_datetime(lr_dates) - epoch).dt.days.astype(float).values
    finite = np.isfinite(smoothed)
    track_at_lr = np.interp(lr_days, grid_days[finite], smoothed[finite])
    noise_pts = float(np.mean(np.abs(lr_corrected - track_at_lr)))

    d = np.diff(smoothed)
    roughness = float(np.nanmean(np.abs(d)))
    return noise_pts, roughness, grid_dates, smoothed


# ---------- main ----------

def main():
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    cs, epoch = load_cs()

    lr = project_long_runs(cs, epoch)
    lr_in = lr[lr['excluded_reason'].isna()].copy().reset_index(drop=True)
    print(f'In-slice long runs: {len(lr_in)}')

    # Covariates on the long-run rows (same encodings as the recovery model).
    daily = pd.read_csv(DATA_DIR / 'daily.csv', parse_dates=['date'])
    daily = daily.sort_values('date')
    daily = daily[daily['date'] >= daily_floor()].reset_index(drop=True)
    races = pd.read_csv(DATA_DIR / 'races.csv', parse_dates=['date'])
    cs_summary, _, _, _ = load_cs_outputs(str(DATA_DIR), '')
    rec_fit = fit_recovery_model(daily, races, cs_summary, verbose=False)
    if rec_fit is None:
        raise SystemExit('Recovery fit unavailable — V3 needs it.')

    lr_in = add_quality_features(lr_in, rec_fit.quality_dates)
    lr_in['temp_centered'] = (lr_in['temp_c'] - TEMP_REFERENCE_C)
    lr_in['tod_is_pm'] = tod_is_pm(lr_in)
    n_temp = int(lr_in['temp_centered'].notna().sum())
    n_tod = int((lr_in['time_of_day'].astype(str).str.strip() != '')
                .where(lr_in['time_of_day'].notna(), False).sum())
    print(f'Coverage on in-slice long runs: temp {n_temp}/{len(lr_in)}, '
          f'time_of_day {n_tod}/{len(lr_in)}')
    # Missing temp = reference temp (contribution 0) so all variants fit the
    # same n — required for AIC comparability.
    lr_in['temp_centered'] = lr_in['temp_centered'].fillna(0.0)

    rec_contrib = transferable_contributions(lr_in, rec_fit.betas,
                                             rec_fit.quality_dates)
    print('Recovery betas used by V3: '
          + ', '.join(f'{f}={rec_fit.betas[f]:+.2f}'
                      for f in ('temp_centered', 'fat_marathon',
                                'fat_race_short', 'tod_is_pm')))

    # Shared design-matrix pieces (mirrors fit_long_run_model exactly).
    loc_counts = lr_in['location'].value_counts()
    qualifying_routes = sorted(loc_counts[loc_counts >= MIN_ROUTE_N].index)
    lr_in['route'] = lr_in['location'].where(
        lr_in['location'].isin(qualifying_routes), 'other')
    route_cat = pd.Categorical(lr_in['route'],
                               categories=['other'] + qualifying_routes)
    route_dum = pd.get_dummies(route_cat, prefix='route').drop(
        columns=['route_other']).astype(float)
    route_dum.index = lr_in.index

    intercept = pd.Series(1.0, index=lr_in.index, name='Intercept')
    bin_lo = pd.Series((lr_in['miles'] < LR_INTERNAL_BIN).astype(float),
                       index=lr_in.index, name='bin_lr_lo')
    miles_c = pd.Series(lr_in['miles'] - LR_INTERNAL_BIN,
                        index=lr_in.index, name='miles_c')
    hinge = pd.Series(np.maximum(0.0, lr_in['miles'] - LR_INTERNAL_BIN),
                      index=lr_in.index, name='hinge_21')
    covars = lr_in[['temp_centered', 'fat_marathon', 'fat_race_short',
                    'tod_is_pm']].astype(float)

    y_raw = lr_in['raw_resid'].astype(float)
    y_v3 = pd.Series(y_raw.values - rec_contrib, index=lr_in.index)

    variants = {
        'V0_bin_route':       (pd.concat([intercept, bin_lo, route_dum], axis=1), y_raw),
        'V1a_linear_route':   (pd.concat([intercept, miles_c, route_dum], axis=1), y_raw),
        'V1b_hinge_route':    (pd.concat([intercept, miles_c, hinge, route_dum], axis=1), y_raw),
        'V2_bin_route_cov':   (pd.concat([intercept, bin_lo, route_dum, covars], axis=1), y_raw),
        # Lean V2: drop tod_is_pm, which is dead on long runs (t≈0.25) despite
        # being a strong recovery factor — keeps only the covariates that earn
        # their AIC penalty.
        'V2b_fat_temp':       (pd.concat([intercept, bin_lo, route_dum,
                                          covars.drop(columns=['tod_is_pm'])],
                                         axis=1), y_raw),
        'V3_recovery_sourced': (pd.concat([intercept, bin_lo, route_dum], axis=1), y_v3),
    }

    wh_w, wh_h = workouts_hills_resid(cs, epoch)
    wh_resid = pd.concat([wh_w, wh_h], ignore_index=True)
    print(f'Fixed smoother input: {len(wh_w)} workouts + {len(wh_h)} hills')

    fits, rows, tracks = {}, [], {}
    for name, (X, y) in variants.items():
        f = fit_pruned_ols(lr_in, X, y)
        fits[name] = f
        kept = f.keep_mask
        noise_pts, roughness, grid_dates, track = smoother_metrics(
            wh_resid, lr_in.loc[kept, 'date'], f.corrected[kept], epoch)
        tracks[name] = (grid_dates, track, lr_in.loc[kept, 'date'],
                        f.corrected[kept])
        rows.append({
            'variant': name, 'k': f.k, 'n_kept': f.n_kept,
            'r2': round(f.rsquared, 3), 'resid_sd': round(f.resid_sd, 2),
            'aic': round(f.aic, 1),
            'noise_pts': round(noise_pts, 2), 'roughness': round(roughness, 3),
        })

    metrics = pd.DataFrame(rows)
    aic_v0 = metrics.loc[metrics['variant'] == 'V0_bin_route', 'aic'].iloc[0]
    metrics['d_aic_vs_v0'] = (metrics['aic'] - aic_v0).round(1)
    # V3 fits a different y — its AIC is not comparable to V0–V2.
    metrics.loc[metrics['variant'] == 'V3_recovery_sourced', 'd_aic_vs_v0'] = np.nan

    # Common-kept-set SD: rows every variant kept, so prune differences
    # don't flatter anyone.
    common = np.logical_and.reduce([f.keep_mask for f in fits.values()])
    for r in rows:
        c = fits[r['variant']].corrected[common]
        r['sd_common'] = round(float(np.std(c, ddof=1)), 2)
    metrics['sd_common'] = [r['sd_common'] for r in rows]
    print(f'\nCommon kept set: {int(common.sum())} rows')

    print('\n=== Variant metrics ===')
    print(metrics.to_string(index=False))
    metrics_csv = DEBUG_DIR / 'lr_variant_metrics.csv'
    metrics.to_csv(metrics_csv, index=False)
    print(f'\nWrote {metrics_csv}')

    # Per-variant coefficient table (route betas side by side).
    coef_tbl = pd.DataFrame({name: pd.Series(f.coefs)
                             for name, f in fits.items()}).round(2)
    coef_csv = DEBUG_DIR / 'lr_variant_coefs.csv'
    coef_tbl.to_csv(coef_csv)
    print(f'Wrote {coef_csv}')
    print('\n=== Coefficients (sec/mi) ===')
    print(coef_tbl.to_string())

    # ---------- visuals ----------
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    names = list(variants.keys())
    fig, axes = plt.subplots(len(names), 1, figsize=(14, 3.2 * len(names)),
                             sharex=True, sharey=True)
    for ax, name in zip(axes, names):
        grid_dates, track, d_kept, c_kept = tracks[name]
        m = next(r for r in rows if r['variant'] == name)
        ax.scatter(d_kept, c_kept, s=10, alpha=0.55, color='#3498DB',
                   edgecolors='none')
        ax.plot(grid_dates, track, color='#E67E22', lw=1.6)
        ax.axhline(0, color='#999', lw=0.6, ls=':')
        ax.set_title(f"{name}   resid SD {m['resid_sd']}  "
                     f"noise_pts {m['noise_pts']}  roughness {m['roughness']}",
                     fontsize=10, loc='left')
        ax.set_ylabel('corrected (s/mi)')
    axes[-1].set_xlabel('date')
    fig.suptitle('Long-run corrected residuals + rebuilt TQ smoother, per variant',
                 fontsize=12)
    fig.tight_layout()
    panel_png = DEBUG_DIR / 'lr_variants_panel.png'
    fig.savefig(panel_png, dpi=110)
    print(f'Wrote {panel_png}')

    fig2, ax2 = plt.subplots(figsize=(14, 5))
    for name, color in (('V2_bin_route_cov', '#3498DB'),
                        ('V3_recovery_sourced', '#E91E63')):
        grid_dates, track, d_kept, c_kept = tracks[name]
        ax2.scatter(d_kept, c_kept, s=10, alpha=0.4, color=color,
                    edgecolors='none', label=f'{name} points')
        ax2.plot(grid_dates, track, color=color, lw=1.8, label=f'{name} track')
    ax2.axhline(0, color='#999', lw=0.6, ls=':')
    ax2.set_ylabel('corrected residual (s/mi)')
    ax2.set_title('V2 (long-run-fit betas) vs V3 (recovery-sourced betas)')
    ax2.legend(fontsize=9)
    fig2.tight_layout()
    overlay_png = DEBUG_DIR / 'lr_v2_vs_v3.png'
    fig2.savefig(overlay_png, dpi=110)
    print(f'Wrote {overlay_png}')


if __name__ == '__main__':
    main()
