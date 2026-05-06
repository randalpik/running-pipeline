"""Bayesian latent-process model for CS(t) and D'(t) — standalone runner.

Generative model:
    log CS(t) ~ GP(μ_CS, Matern52(σ_f_CS, ℓ_CS))
    log D'(t) ~ GP(μ_D', Matern52(σ_f_D', ℓ_D'))
    log τ_i  ~ Normal(log[(d_i - D'(t_i)) / CS(t_i)], σ_obs)

Uses HSGP (Hilbert space approximation) to make the GP tractable on a long
time series — full-rank Latent GP is O(N³) per leapfrog and infeasible at
500+ grid points without a BLAS-linked PyTensor.

USAGE (local run):
    pip install pymc==5.* arviz pandas numpy scipy plotly
    python src/models/bayes_cs_fit.py

By default, reads `races.csv` from `data/` and writes outputs there too.
Override with --races / --out-dir if needed. Exclusions are derived
automatically from the eligible race set (see derive_exclusions below) —
no manual exclusion file is consumed.

Inputs:
    races.csv (or --races PATH)
        Required columns: date, distance_m, time_sec
        Optional columns: fatigued, surface, event
        This is the same file produced by build_dataset.py (data/races.csv).

Outputs (in --out-dir, default data/):
    bayes_cs_summary.csv          One row per grid point with median + 50/95% intervals
    bayes_cs_residuals.csv        One row per race: predicted vs actual time, residual
    bayes_cs_posterior.nc         Full InferenceData for any deeper analysis
    bayes_cs_diagnostics.txt      R-hat, divergences, residual bands
    bayes_cs_auto_exclusions.csv  Audit trail: which races the rule pruned and why

Expected runtime on a normal laptop with M=60 basis functions, 14d grid,
~200 races, 4 chains × 2000 total draws: 5–20 minutes.
"""
import argparse
import os
import sys
import time as tclock
import datetime as dt
from pathlib import Path
from typing import Any, cast
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.shared.paths import DATA_DIR, DEBUG_DIR


DEFAULT_RACES   = str(DATA_DIR / 'races.csv')
DEFAULT_OUT_DIR = str(DATA_DIR)


def build_eligible(races_path):
    """Load races.csv and apply hard-eligibility filters (no auto-exclusions yet).

    Filters:
        fatigued != True       (multi-race-day non-first races)
        surface != 'Downhill'  (downhill courses are not CS-comparable)
        time_sec >= 120        (anything sub-2-min is data noise)

    Auto-exclusions are applied separately by derive_exclusions().
    """
    races = pd.read_csv(races_path, parse_dates=['date'])
    races['date'] = races['date'].dt.date

    needed = ['date', 'distance_m', 'time_sec']
    missing = [c for c in needed if c not in races.columns]
    if missing:
        sys.exit(f"ERROR: races CSV missing columns: {missing}")

    # Optional columns — fill defaults if absent
    if 'fatigued' not in races.columns:
        races['fatigued'] = False
    if 'surface' not in races.columns:
        races['surface'] = 'Unknown'
    if 'event' not in races.columns:
        races['event'] = ''

    elig = races[
        (~races['fatigued'].astype(bool)) &
        (races['surface'] != 'Downhill') &
        (races['time_sec'] >= 120)
    ].copy().sort_values('date').reset_index(drop=True)
    return elig


def derive_exclusions(elig, xc_correction=0.06,
                      tier1_marathon_thresh=115.0,
                      tier1_hm_thresh=500.0,
                      tier2_z_thresh=2.5,
                      tier2_kmin=5,
                      window_years=2.0,
                      trim_pct=0.02,
                      sigma_iters=3):
    """Apply unified two-tier auto-exclusion rule.

    Replaces the historical hand-curated `cs_exclusions_v7.csv`. The rule
    treats long and short races differently because their residual
    distributions are structurally different:

    Tier 1 — Marathon (>=25K) and HM (15K-24.999K):
        Symmetric ±window_years same-band median residual in seconds. The
        kept-marathon residuals cluster tightly (good days) and bonks stick
        up clearly above; an absolute residual threshold separates them.
        Different thresholds for HM and Marathon because the noise floors
        differ (HM: kept_max +177s, excl_min +1754s — wide gap. Marathon:
        kept_max +113s, excl_min +118s — tight).

    Tier 2 — sub-marathon (<15K):
        Same-band past-only ±window_years log-pace median, K_min ≥ tier2_kmin.
        Past-only protects against fitness-rise contamination (early-career
        races aren't measured against future faster races). Log-pace makes
        residuals comparable across distances and eras.

        Global σ via iterated trimmed MAD over the full pool of sub-marathon
        log-residuals (trim_pct on each iter, sigma_iters total) — uniform
        scale across bands and eras avoids the local-cluster artifacts that
        per-race MAD-z would produce.

        z = (log_resid - global_median) / global_σ; prune if z > tier2_z_thresh.

    Design note: this rule is INTENTIONALLY less aggressive than the prior
    manual list for sub-marathon distances. With σ_robust ≈ 5% pace, races
    that look like "obvious bonks" (e.g. WGP XC days) often sit at z=2.4 —
    statistically indistinguishable from kept races at the same z. The CS
    model's σ_per_race ≈ 0.028 absorbs ~3σ events naturally without
    structural distortion. Only Tahoma 2014 (z=4.45, +20% slow) is a
    genuine sub-marathon outlier worth pruning.

    Args:
        elig: DataFrame returned by build_eligible (already-filtered).
        xc_correction: Same divisor as the main fit (default 0.06 = 6%).

    Returns:
        (excl_df, sigma_global, median_global)
            excl_df: DataFrame with one row per pruned race; columns
                date, distance_m, event, surface, tier, metric, value,
                threshold, n_neighbors, sigma_global.
            sigma_global, median_global: floats from sub-marathon pool;
                NaN if pool too small.
    """
    df = elig.copy().reset_index(drop=True)
    df['date'] = pd.to_datetime(df['date'])

    # XC pre-correction (matches main fit's pre-correction)
    df['time_sec_corr'] = df['time_sec'].astype(float)
    xc_mask = df['surface'].fillna('').astype(str).str.upper() == 'XC'
    df.loc[xc_mask, 'time_sec_corr'] = df.loc[xc_mask, 'time_sec_corr'] / (1.0 + xc_correction)

    # Distance bands
    def band(d):
        if d < 1500:  return '<1500m'
        if d < 3500:  return '1500-3499m'
        if d < 5500:  return '5K'
        if d < 8500:  return '5mi-8K'
        if d < 12000: return '10K'
        if d < 15000: return None  # 10K-15K gap; no races fall here historically
        if d < 25000: return 'HM'
        return 'Marathon'
    df['band'] = df['distance_m'].apply(band)
    df['log_pace'] = np.log(df['time_sec_corr'] / (df['distance_m'] / 1609.344))

    # ---- Tier 1: Marathon + HM, symmetric ±window_years same-band median ----
    df['t1_resid_sec'] = np.nan
    df['t1_n']         = 0
    win = pd.Timedelta(days=int(window_years * 365))
    for b in ('Marathon', 'HM'):
        idx = df.index[df['band'] == b].tolist()
        sub = df.loc[idx]
        for i in idx:
            d_i = df.loc[i, 'date']
            mask = (sub['date'] >= d_i - win) & (sub['date'] <= d_i + win) & (sub.index != i)
            nb = sub[mask]
            df.loc[i, 't1_n'] = len(nb)
            if len(nb) >= 2:
                df.loc[i, 't1_resid_sec'] = df.loc[i, 'time_sec_corr'] - nb['time_sec_corr'].median()

    # ---- Tier 2: sub-marathon, past-only same-band log-pace ----
    sub_bands = ['<1500m', '1500-3499m', '5K', '5mi-8K', '10K']
    df['t2_log_resid'] = np.nan
    df['t2_n_past']    = 0
    for b in sub_bands:
        idx = df.index[df['band'] == b].tolist()
        sub = df.loc[idx]
        for i in idx:
            d_i = df.loc[i, 'date']
            mask = (sub['date'] < d_i) & (sub['date'] >= d_i - win)
            nb = sub[mask]
            df.loc[i, 't2_n_past'] = len(nb)
            if len(nb) >= 2:
                df.loc[i, 't2_log_resid'] = df.loc[i, 'log_pace'] - nb['log_pace'].median()

    # Global σ from iterated trimmed MAD over the pool
    pool_mask = (df['band'].isin(sub_bands)) & (df['t2_n_past'] >= tier2_kmin) & df['t2_log_resid'].notna()
    pool = df.loc[pool_mask, 't2_log_resid'].values
    sigma_g = float('nan')
    med_g   = float('nan')
    if len(pool) >= 10:
        x = pool.copy()
        for _ in range(sigma_iters):
            q_lo, q_hi = np.percentile(x, [trim_pct * 100, (1 - trim_pct) * 100])
            x_t = x[(x >= q_lo) & (x <= q_hi)]
            med_g   = float(np.median(x_t))
            mad     = float(np.median(np.abs(x_t - med_g)))
            sigma_g = 1.4826 * mad
        df['t2_z'] = (df['t2_log_resid'] - med_g) / max(sigma_g, 1e-6)
    else:
        df['t2_z'] = np.nan

    # ---- Apply rules; build audit trail ----
    rows = []
    for _, r in df.iterrows():
        if r['band'] == 'Marathon':
            if pd.notna(r['t1_resid_sec']) and r['t1_resid_sec'] > tier1_marathon_thresh:
                rows.append({
                    'date': r['date'].date() if hasattr(r['date'], 'date') else r['date'],
                    'distance_m': int(r['distance_m']),
                    'event': r.get('event', ''), 'surface': r.get('surface', ''),
                    'tier': 'M', 'metric': 'symmetric_resid_sec',
                    'value': round(float(r['t1_resid_sec']), 1),
                    'threshold': tier1_marathon_thresh,
                    'n_neighbors': int(r['t1_n']),
                    'sigma_global': '',
                })
        elif r['band'] == 'HM':
            if pd.notna(r['t1_resid_sec']) and r['t1_resid_sec'] > tier1_hm_thresh:
                rows.append({
                    'date': r['date'].date() if hasattr(r['date'], 'date') else r['date'],
                    'distance_m': int(r['distance_m']),
                    'event': r.get('event', ''), 'surface': r.get('surface', ''),
                    'tier': 'HM', 'metric': 'symmetric_resid_sec',
                    'value': round(float(r['t1_resid_sec']), 1),
                    'threshold': tier1_hm_thresh,
                    'n_neighbors': int(r['t1_n']),
                    'sigma_global': '',
                })
        elif r['band'] in sub_bands:
            if r['t2_n_past'] >= tier2_kmin and pd.notna(r['t2_z']) and r['t2_z'] > tier2_z_thresh:
                rows.append({
                    'date': r['date'].date() if hasattr(r['date'], 'date') else r['date'],
                    'distance_m': int(r['distance_m']),
                    'event': r.get('event', ''), 'surface': r.get('surface', ''),
                    'tier': 'sub-M', 'metric': 'past_log_pace_z',
                    'value': round(float(r['t2_z']), 3),
                    'threshold': tier2_z_thresh,
                    'n_neighbors': int(r['t2_n_past']),
                    'sigma_global': round(sigma_g, 4),
                })

    excl_df = pd.DataFrame(rows)
    if len(excl_df) > 0:
        excl_df = excl_df.sort_values('date').reset_index(drop=True)
    return excl_df, sigma_g, med_g


def main():
    p = argparse.ArgumentParser(description=(__doc__ or '').split('\n\n')[0])
    p.add_argument('--races', default=DEFAULT_RACES,
                   help=f'Path to races.csv (default: {DEFAULT_RACES})')
    p.add_argument('--out-dir', default=DEFAULT_OUT_DIR,
                   help=f'Output directory (default: {DEFAULT_OUT_DIR})')
    p.add_argument('--grid-step', type=int, default=7,
                   help='Days between inference grid points (default 7)')
    p.add_argument('--m-basis', type=int, default=100,
                   help='Number of HSGP basis functions (default 100; '
                        'higher=more flexible, slower)')
    p.add_argument('--draws', type=int, default=1000)
    p.add_argument('--tune', type=int, default=1000)
    p.add_argument('--chains', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    # Tunable priors (for grid-search experimentation)
    p.add_argument('--sigma-base-prior', type=float, default=0.02,
                   help='HalfNormal σ for sigma_base (default 0.02)')
    p.add_argument('--ell-cs-mu', type=float, default=0.25,
                   help='LogNormal location (in years) for ell_cs (default 0.25)')
    p.add_argument('--ell-cs-sigma', type=float, default=0.4,
                   help='LogNormal scale (log-space) for ell_cs (default 0.4)')
    p.add_argument('--xc-correction', type=float, default=0.08,
                   help='Multiplicative correction applied to XC race times '
                        'before fitting (default 0.08 = 8%% terrain-effect '
                        'compensation). Pre-model adjustment: XC time_sec is '
                        'divided by (1+c) so XC races enter the model as if '
                        'they were equivalent flat-course times. Set to 0 to '
                        'disable. Iterate based on visual fall-vs-spring '
                        'alignment in the chart.')
    p.add_argument('--tag', default='',
                   help='Suffix for output filenames (e.g. "v4a") to keep '
                        'experiments separate')
    p.add_argument('--diagnostics', action='store_true',
                   help='Also write bayes_cs_residuals.csv, '
                        'bayes_cs_posterior.nc, and bayes_cs_diagnostics.txt '
                        'into output/debug/. These are not consumed by the '
                        'plot pipeline; off by default to keep data/ clean.')
    args = p.parse_args()

    if not os.path.exists(args.races):
        sys.exit(f"ERROR: races CSV not found at {args.races}")

    os.makedirs(args.out_dir, exist_ok=True)

    # File-naming suffix: append --tag if provided so multiple experiments
    # don't clobber each other's outputs. Computed early because the
    # auto-exclusions audit file is the first artifact written.
    suffix = f"_{args.tag}" if args.tag else ""

    # ---------- load + auto-derive exclusions + filter ----------
    elig_full = build_eligible(args.races)
    print(f"Hard-eligible races (post fatigued/Downhill/<120s filter): {len(elig_full)}")

    excl_df, sigma_g, sigma_med = derive_exclusions(elig_full,
                                                    xc_correction=args.xc_correction)
    audit_path = os.path.join(args.out_dir, f'bayes_cs_auto_exclusions{suffix}.csv')
    excl_df.to_csv(audit_path, index=False)
    print(f"Auto-derived {len(excl_df)} exclusions "
          f"(sub-marathon σ_global={sigma_g:.4f}, median={sigma_med:+.4f})")
    print(f"Wrote audit trail: {audit_path}")
    if len(excl_df):
        # Compact stdout summary
        for _, r in excl_df.iterrows():
            print(f"  PRUNE  {r['date']}  {r['distance_m']:>5}m  {r['surface']:5s}  "
                  f"tier={r['tier']:<5s}  {r['metric']}={r['value']}  "
                  f"thresh={r['threshold']}  n={r['n_neighbors']}  "
                  f"{str(r.get('event',''))[:35]}")

    # Filter eligibility by auto-derived set (composite key: date + distance_m
    # because some dates have multiple races with different race_seq)
    excl_keys = set(zip(pd.to_datetime(excl_df['date']).dt.date if len(excl_df) else [],
                        excl_df['distance_m'].astype(int) if len(excl_df) else []))
    elig_full = elig_full.copy()
    elig_full['_key'] = list(zip(
        pd.to_datetime(elig_full['date']).dt.date,
        elig_full['distance_m'].astype(int),
    ))
    elig = elig_full[~elig_full['_key'].isin(excl_keys)].drop(columns=['_key']).reset_index(drop=True)
    print(f"Eligible races after auto-exclusion: {len(elig)}")
    print(f"Distinct distances: {sorted(set(int(d) for d in elig['distance_m']))}")
    print(f"Date range: {elig['date'].min()} to {elig['date'].max()}")

    # ---------- XC pre-model correction ----------
    # XC race times are systematically slower than equivalent flat-course times
    # due to terrain (grass, gentle hills). The model can't empirically separate
    # this terrain penalty from real fitness loss because XC races cluster in
    # fall seasons with no concurrent track/road races to compare against.
    # We apply an exogenous correction: divide XC time by (1 + c) before
    # fitting so XC races enter the model as if they were on flat ground.
    # Original times are preserved in *_original columns for hover display.
    elig['time_sec_original'] = elig['time_sec'].copy()
    if 'pace_sec_per_mi' in elig.columns:
        elig['pace_sec_per_mi_original'] = elig['pace_sec_per_mi'].copy()
    is_xc_mask = elig['surface'].fillna('').astype(str).str.upper() == 'XC'
    n_xc = int(is_xc_mask.sum())
    if args.xc_correction > 0 and n_xc > 0:
        factor = 1.0 / (1.0 + args.xc_correction)
        elig.loc[is_xc_mask, 'time_sec'] = elig.loc[is_xc_mask, 'time_sec'] * factor
        if 'pace_sec_per_mi' in elig.columns:
            elig.loc[is_xc_mask, 'pace_sec_per_mi'] = elig.loc[is_xc_mask, 'pace_sec_per_mi'] * factor
        print(f"XC pre-correction: {n_xc} races, factor={factor:.4f} "
              f"(c={args.xc_correction:.3f} = {args.xc_correction*100:.1f}% terrain penalty)")
    elif n_xc > 0:
        print(f"XC pre-correction disabled (--xc-correction 0); {n_xc} XC races kept as-is")

    # ---------- inference grid ----------
    first_d = elig['date'].min()
    last_d  = max(elig['date'].max(), dt.date.today())
    grid_dates = []
    d = first_d
    while d <= last_d:
        grid_dates.append(d)
        d += dt.timedelta(days=args.grid_step)
    if grid_dates[-1] < last_d:
        grid_dates.append(last_d)

    grid_t = np.array([(g - first_d).days for g in grid_dates], dtype=float)
    n_grid = len(grid_dates)
    print(f"Inference grid: {n_grid} points (every {args.grid_step}d)")

    # Map race dates to grid indices (nearest grid point)
    race_grid_idx = np.array([
        min(int(round((rd - first_d).days / args.grid_step)), n_grid - 1)
        for rd in elig['date']
    ])

    race_distances = elig['distance_m'].to_numpy().astype(float)
    race_times = elig['time_sec'].to_numpy().astype(float)
    log_race_times = np.log(race_times)

    # Center grid_t for HSGP numerical stability
    grid_t_centered = (grid_t - grid_t.mean()) / 365.0  # in years
    L = (grid_t_centered.max() - grid_t_centered.min()) * 1.5  # boundary buffer
    X_grid = grid_t_centered.reshape(-1, 1)
    print(f"Grid spans {grid_t_centered.min():.2f} to {grid_t_centered.max():.2f} years (HSGP L={L:.2f})")

    # ---------- model ----------
    with pm.Model() as model:
        mu_cs = pm.Normal('mu_cs', mu=np.log(4.0), sigma=0.3)
        mu_dp = pm.Normal('mu_dp', mu=np.log(250.0), sigma=0.5)

        # Hierarchical CS structure (additive GPs):
        #   log_cs(t) = mu_cs + log_cs_trend(t) + log_cs_dev(t)
        # log_cs_trend captures slow long-term fitness trajectory (years scale).
        # log_cs_dev captures faster training-cycle wiggles (months scale).
        # Why this matters: at boundaries (past last race, before first race)
        # the SHORT-scale GP decays to zero quickly, but the LONG-scale trend
        # persists, anchoring extrapolated predictions to a smooth fitness
        # trajectory rather than letting them follow the most recent dip.
        sf_cs_long  = pm.HalfNormal('sf_cs_long',  sigma=0.3)
        ell_cs_long = pm.LogNormal('ell_cs_long', mu=np.log(5.0), sigma=0.5)
        sf_cs_dev   = pm.HalfNormal('sf_cs_dev',   sigma=0.10)
        ell_cs_dev  = pm.LogNormal('ell_cs_dev',
                                    mu=np.log(args.ell_cs_mu),
                                    sigma=args.ell_cs_sigma)

        sf_dp = pm.HalfNormal('sf_dp', sigma=0.5)
        ell_dp = pm.LogNormal('ell_dp', mu=np.log(0.5), sigma=0.5)

        # Distance-dependent observation noise. See earlier comment block;
        # CLI-controlled via --sigma-base-prior. Default HN(0.02).
        sigma_base = pm.HalfNormal('sigma_base', sigma=args.sigma_base_prior)
        alpha_sig = pm.HalfNormal('alpha_sig', sigma=0.3)
        D_REF = 5000.0
        sigma_per_race = sigma_base * (race_distances / D_REF) ** alpha_sig

        # Long-distance bias term: pure CS theory predicts marathon times that
        # are systematically too fast (no glycogen wall, no fuel constraints,
        # no thermoregulatory drift). Without correction, marathons drag CS
        # downward to "explain" their slowness, distorting the CS estimate
        # at marathon-adjacent dates.
        #
        # Model: predicted time gets multiplied by (1 + β * log(d/d_thresh))
        # for d > d_thresh, so the inflation is 0 below threshold and grows
        # smoothly with distance. β has prior centered at 0 — the data
        # determines the magnitude. If a future race breaks through the
        # historical bias (e.g. a well-executed marathon), it will register
        # as a strong negative residual, naturally pulling β downward.
        #
        # At β=0.05: HM bias ≈ +3.7%, marathon bias ≈ +7.2% — matches
        # observed v7 residuals (HM mean +4.2%, marathon mean +8.5%).
        d_thresh = 10000.0
        beta_long = pm.HalfNormal('beta_long', sigma=0.1)
        # log_d_ratio is 0 for d <= d_thresh, log(d/d_thresh) above
        log_d_ratio = np.maximum(0.0, np.log(race_distances / d_thresh))

        # XC bias is handled OUTSIDE the model via pre-correction of XC times
        # (see --xc-correction CLI flag). The model can't empirically separate
        # terrain effect from fitness loss in fall XC clusters, so we encode
        # the terrain penalty exogenously and let the model fit pre-corrected
        # times naturally.
        bias_factor = 1.0 + beta_long * log_d_ratio

        # HSGP — Hilbert space approximation, much cheaper than full Latent
        cov_cs_long = sf_cs_long ** 2 * pm.gp.cov.Matern52(input_dim=1, ls=ell_cs_long)
        cov_cs_dev  = sf_cs_dev  ** 2 * pm.gp.cov.Matern52(input_dim=1, ls=ell_cs_dev)
        cov_dp = sf_dp ** 2 * pm.gp.cov.Matern52(input_dim=1, ls=ell_dp)

        # Long-scale trend uses fewer basis functions: ratio of m to L tracks
        # how short of a wavelength the GP can express. With ell_long ~ 5y and
        # m_long = 15-20 we cover the whole timeline with smooth modes.
        m_long = max(20, args.m_basis // 3)
        gp_cs_long = pm.gp.HSGP(m=[m_long], L=[L], cov_func=cov_cs_long)
        gp_cs_dev  = pm.gp.HSGP(m=[args.m_basis], L=[L], cov_func=cov_cs_dev)
        gp_dp = pm.gp.HSGP(m=[args.m_basis], L=[L], cov_func=cov_dp)

        log_cs_trend = gp_cs_long.prior('log_cs_trend', X=X_grid)
        log_cs_dev   = gp_cs_dev.prior('log_cs_dev',   X=X_grid)
        log_dp = gp_dp.prior('log_dp', X=X_grid)

        log_cs_total = mu_cs + log_cs_trend + log_cs_dev
        log_dp_total = log_dp + mu_dp

        cs_at_race = cast(Any, pm.math.exp(log_cs_total[race_grid_idx]))
        dp_at_race = cast(Any, pm.math.exp(log_dp_total[race_grid_idx]))

        # Hyperbolic likelihood: t = (d - D')/CS, then long-distance correction
        expected_time = (race_distances - dp_at_race) / cs_at_race
        expected_time_corrected = expected_time * bias_factor
        log_expected = pm.math.log(expected_time_corrected)

        pm.Normal('obs', mu=log_expected, sigma=sigma_per_race, observed=log_race_times)

    # ---------- prior predictive ----------
    print("\nRunning prior predictive (200 samples)...")
    with model:
        prior_pred: Any = pm.sample_prior_predictive(samples=200, random_seed=args.seed)
    log_cs_pp = (prior_pred.prior['log_cs_trend'].values +
                 prior_pred.prior['log_cs_dev'].values +
                 prior_pred.prior['mu_cs'].values[..., None])
    cs_pp = np.exp(log_cs_pp)
    pace_pp = 1609.344 / cs_pp / 60
    print(f"  Prior CS pace (min/mi): "
          f"5%={np.percentile(pace_pp,5):.2f}  median={np.median(pace_pp):.2f}  "
          f"95%={np.percentile(pace_pp,95):.2f}")

    # ---------- posterior ----------
    print(f"\nSampling NUTS: tune={args.tune} draws={args.draws} chains={args.chains}")
    t0 = tclock.time()
    with model:
        trace: Any = pm.sample(
            draws=args.draws, tune=args.tune,
            chains=args.chains, cores=min(args.chains, os.cpu_count()),
            target_accept=0.95, random_seed=args.seed,
            return_inferencedata=True,
        )
    elapsed = tclock.time() - t0
    print(f"Sampling done in {elapsed/60:.1f} min")

    # ---------- output paths ----------
    # summary + params CSVs are essential inputs to the plot pipeline and
    # always go to args.out_dir (data/). The other three files (diagnostics
    # text, per-race residuals CSV, full posterior netCDF) are diagnostic-
    # only — gated behind --diagnostics and routed to DEBUG_DIR so they
    # don't clutter data/ on default runs.
    # The summary CSV is an essential input to the plot pipeline and lands
    # in args.out_dir (data/). Diagnostic outputs (residuals.csv,
    # posterior.nc, diagnostics.txt) are gated behind --diagnostics and
    # routed to DEBUG_DIR — see paths.py.
    summary_path = os.path.join(args.out_dir, f'bayes_cs_summary{suffix}.csv')

    if args.diagnostics:
        DEBUG_DIR.mkdir(parents=True, exist_ok=True)
        diag_path = str(DEBUG_DIR / f'bayes_cs_diagnostics{suffix}.txt')
        resid_path = str(DEBUG_DIR / f'bayes_cs_residuals{suffix}.csv')
        nc_path = str(DEBUG_DIR / f'bayes_cs_posterior{suffix}.nc')

        with open(diag_path, 'w') as f:
            f.write(f"=== Run config ===\n")
            f.write(f"tag: {args.tag or '(none)'}\n")
            f.write(f"sigma_base prior: HalfNormal(σ={args.sigma_base_prior})\n")
            f.write(f"ell_cs_dev prior: LogNormal(μ=log({args.ell_cs_mu}), σ={args.ell_cs_sigma})\n")
            f.write(f"chains: {args.chains}, draws: {args.draws}, tune: {args.tune}\n")
            f.write(f"target_accept: 0.95\n")
            f.write(f"xc_correction: {args.xc_correction:.4f} ({args.xc_correction*100:.1f}% terrain penalty)\n")

            f.write(f"\n=== Auto-derived exclusions ===\n")
            f.write(f"Total: {len(excl_df)} races pruned from {len(elig_full)} hard-eligible\n")
            f.write(f"Sub-marathon σ_global (iterated trimmed MAD): {sigma_g:.4f} "
                    f"(≈ {sigma_g*100:.2f}% pace) | median: {sigma_med:+.4f}\n")
            if len(excl_df):
                f.write(f"\n{'date':<12s} {'dist':>6s} {'surf':<5s} {'tier':<6s} "
                        f"{'metric':<22s} {'value':>10s} {'thresh':>8s} {'n':>3s}  event\n")
                for _, r in excl_df.iterrows():
                    f.write(f"{str(r['date']):<12s} {r['distance_m']:>6d} "
                            f"{str(r['surface'])[:5]:<5s} {r['tier']:<6s} "
                            f"{r['metric']:<22s} {r['value']:>10.2f} "
                            f"{r['threshold']:>8.1f} {r['n_neighbors']:>3d}  "
                            f"{str(r.get('event',''))[:40]}\n")

            f.write(f"\n=== Hyperparameter posterior summary ===\n")
            summ = az.summary(trace, var_names=['mu_cs', 'mu_dp',
                                                 'sf_cs_long', 'ell_cs_long',
                                                 'sf_cs_dev',  'ell_cs_dev',
                                                 'sf_dp', 'ell_dp',
                                                 'sigma_base', 'alpha_sig',
                                                 'beta_long'])
            f.write(summ.to_string())
            f.write("\n\n")
            n_div = int(trace.sample_stats['diverging'].sum())
            n_total = int(trace.sample_stats['diverging'].size)
            f.write(f"Divergences: {n_div} / {n_total}\n")
            f.write(f"Sampling time: {elapsed/60:.1f} min\n")
            f.write(f"Grid points: {n_grid}\n")
            f.write(f"HSGP basis size m: {args.m_basis}\n")

        print(f"Wrote {diag_path}")

    # ---------- posterior summary on grid ----------
    log_cs_trend_post = trace.posterior['log_cs_trend'].values
    log_cs_dev_post   = trace.posterior['log_cs_dev'].values
    mu_cs_post        = trace.posterior['mu_cs'].values[..., None]
    log_cs_post       = mu_cs_post + log_cs_trend_post + log_cs_dev_post
    log_dp_post = trace.posterior['log_dp'].values + trace.posterior['mu_dp'].values[..., None]

    cs_flat       = np.exp(log_cs_post).reshape(-1, n_grid)
    # Trend-only (mu_cs + slow component) for separate visualization/diagnostic
    cs_trend_flat = np.exp((mu_cs_post + log_cs_trend_post)).reshape(-1, n_grid)
    dp_flat = np.exp(log_dp_post).reshape(-1, n_grid)
    pace_flat = 1609.344 / cs_flat / 60
    pace_trend_flat = 1609.344 / cs_trend_flat / 60

    summary_rows = []
    for i, gd in enumerate(grid_dates):
        summary_rows.append({
            'date': gd,
            'cs_pace_med':  np.median(pace_flat[:, i]),
            'cs_pace_lo50': np.percentile(pace_flat[:, i], 25),
            'cs_pace_hi50': np.percentile(pace_flat[:, i], 75),
            'cs_pace_lo95': np.percentile(pace_flat[:, i], 2.5),
            'cs_pace_hi95': np.percentile(pace_flat[:, i], 97.5),
            'cs_mps_med':   np.median(cs_flat[:, i]),
            # Trend-only (slow component) for diagnostic / boundary fallback
            'cs_pace_trend_med':  np.median(pace_trend_flat[:, i]),
            'cs_pace_trend_lo95': np.percentile(pace_trend_flat[:, i], 2.5),
            'cs_pace_trend_hi95': np.percentile(pace_trend_flat[:, i], 97.5),
            'dp_med':       np.median(dp_flat[:, i]),
            'dp_lo50':      np.percentile(dp_flat[:, i], 25),
            'dp_hi50':      np.percentile(dp_flat[:, i], 75),
            'dp_lo95':      np.percentile(dp_flat[:, i], 2.5),
            'dp_hi95':      np.percentile(dp_flat[:, i], 97.5),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(summary_path, index=False)
    print(f"Wrote {summary_path} ({len(summary_df)} rows)")

    # ---------- per-race residuals + posterior predictive (diagnostics only) ----------
    if args.diagnostics:
        cs_at_race_med = np.median(cs_flat[:, race_grid_idx], axis=0)
        dp_at_race_med = np.median(dp_flat[:, race_grid_idx], axis=0)
        # Apply long-distance bias correction (matching the likelihood model).
        # XC races already had their times pre-corrected, so no β_xc here.
        beta_long_med = float(np.median(trace.posterior['beta_long'].values))
        log_d_ratio_np = np.maximum(0.0, np.log(race_distances / 10000.0))
        bias_factor_med = 1.0 + beta_long_med * log_d_ratio_np
        pred_t = (race_distances - dp_at_race_med) / cs_at_race_med * bias_factor_med
        pct_resid = (race_times / pred_t - 1) * 100

        resid_df = pd.DataFrame({
            'date': elig['date'].values,
            'distance_m': race_distances,
            'actual_sec': race_times,
            'actual_sec_original': elig['time_sec_original'].values,
            'surface': elig['surface'].values if 'surface' in elig.columns else [''] * len(elig),
            'predicted_sec': pred_t,
            'pct_resid': pct_resid,
            'cs_pace_med_at_race': 1609.344 / cs_at_race_med / 60,
            'dp_med_at_race': dp_at_race_med,
            'event': elig['event'].values if 'event' in elig.columns else [''] * len(elig),
        })
        resid_df.to_csv(resid_path, index=False)
        print(f"Wrote {resid_path}")

        # Append residual band summary to diagnostics
        with open(diag_path, 'a') as f:
            f.write("\n=== Residuals by distance band ===\n")
            bands = [
                ('< 1500m',         resid_df['distance_m'] < 1500),
                ('1500-3500m',      (resid_df['distance_m'] >= 1500) & (resid_df['distance_m'] < 3500)),
                ('5K (3500-5500)',  (resid_df['distance_m'] >= 3500) & (resid_df['distance_m'] < 5500)),
                ('10K (8K-12K)',    (resid_df['distance_m'] >= 8000) & (resid_df['distance_m'] < 12000)),
                ('HM (15K-25K)',    (resid_df['distance_m'] >= 15000) & (resid_df['distance_m'] < 25000)),
                ('Marathon (>25K)', resid_df['distance_m'] >= 25000),
            ]
            f.write(f"{'band':<22} {'n':>4} {'mean_pct':>10} {'std_pct':>10}\n")
            for label, mask in bands:
                sub = resid_df[mask]
                if len(sub) == 0: continue
                f.write(f"{label:<22} {len(sub):>4} {sub['pct_resid'].mean():>+9.2f}% "
                        f"{sub['pct_resid'].std():>9.2f}%\n")
            f.write("\n=== Marathon residuals (each) ===\n")
            for _, r in resid_df[resid_df['distance_m'] >= 25000].iterrows():
                f.write(f"  {r['date']}  actual={r['actual_sec']:.0f}s  pred={r['predicted_sec']:.0f}s  "
                        f"resid={r['pct_resid']:+.1f}%  "
                        f"CS@={r['cs_pace_med_at_race']:.2f} min/mi  D'@={r['dp_med_at_race']:.0f}m  "
                        f"{r['event']}\n")

            # ---------- posterior predictive coverage ----------
            # For each race, the model implies a posterior-predictive distribution
            # over log(time) given (CS, D', σ_per_race) at that race's date.
            # Mean: log((d - D')/CS).  Std: σ_per_race.
            # Coverage: % of races where the actual log(time) falls in the 50%
            # and 95% intervals of this distribution. A well-calibrated model has
            # 50%-coverage ≈ 50% and 95%-coverage ≈ 95%.
            cs_post = cs_flat[:, race_grid_idx]   # (samples, races)
            dp_post = dp_flat[:, race_grid_idx]   # (samples, races)
            # σ_per_race posterior: σ_base * (d/D_REF)^α, sample by sample
            sb_post = trace.posterior['sigma_base'].values.reshape(-1)
            a_post  = trace.posterior['alpha_sig'].values.reshape(-1)
            beta_post = trace.posterior['beta_long'].values.reshape(-1)
            # Broadcast σ across races: shape (samples, races)
            D_REF_arr = 5000.0
            ratio = (race_distances[None, :] / D_REF_arr) ** a_post[:, None]
            sig_per_race = sb_post[:, None] * ratio   # (samples, races)
            # Per-sample bias factor (long-distance only; XC corrected upstream)
            bias_factor_post = 1.0 + beta_post[:, None] * log_d_ratio_np[None, :]
            # Posterior-predictive log-time mean (with bias) and noise per sample
            expected_t_pp = (race_distances[None, :] - dp_post) / cs_post * bias_factor_post
            mean_log_t = np.log(expected_t_pp)  # (samples, races)
            # Draw one log_t per (sample, race): mean + σ * N(0,1)
            rng = np.random.default_rng(args.seed)
            eps = rng.standard_normal(mean_log_t.shape)
            pp_log_t = mean_log_t + sig_per_race * eps   # (samples, races)
            # Per-race quantiles
            q025 = np.percentile(pp_log_t, 2.5, axis=0)
            q975 = np.percentile(pp_log_t, 97.5, axis=0)
            q25  = np.percentile(pp_log_t, 25,   axis=0)
            q75  = np.percentile(pp_log_t, 75,   axis=0)
            actual_log_t = np.log(race_times)
            in_95 = (actual_log_t >= q025) & (actual_log_t <= q975)
            in_50 = (actual_log_t >= q25)  & (actual_log_t <= q75)

            f.write(f"\n=== Posterior predictive coverage ===\n")
            f.write(f"Well-calibrated target: 50%-coverage ≈ 50%, 95%-coverage ≈ 95%\n")
            f.write(f"Overall (n={len(elig)}): "
                    f"50%-coverage = {in_50.mean()*100:.1f}%  "
                    f"95%-coverage = {in_95.mean()*100:.1f}%\n")
            f.write(f"\nBy distance band:\n")
            f.write(f"{'band':<22} {'n':>4} {'cov_50':>9} {'cov_95':>9}\n")
            for label, mask in bands:
                mask_arr = mask.to_numpy()
                if mask_arr.sum() == 0: continue
                f.write(f"{label:<22} {int(mask_arr.sum()):>4} "
                        f"{in_50[mask_arr].mean()*100:>8.1f}% "
                        f"{in_95[mask_arr].mean()*100:>8.1f}%\n")

        # ---------- save full posterior ----------
        trace.to_netcdf(nc_path)
        print(f"Wrote {nc_path}")

    # ---------- write bias-parameter CSV for the plot script ----------
    # The plot script reads this to apply the same long-distance and XC bias
    # corrections to race diamonds that the model used internally — so the
    # diamonds visually match what the model believes the race "should" be on
    # a flat course at the corresponding distance band.
    params_path = os.path.join(args.out_dir, f'bayes_cs_params{suffix}.csv')
    params_df = pd.DataFrame([{
        'beta_long_med': float(np.median(trace.posterior['beta_long'].values)),
        'd_thresh_long': 10000.0,
        'xc_correction': float(args.xc_correction),
    }])
    params_df.to_csv(params_path, index=False)
    print(f"Wrote {params_path}")

    print(f"\nDone. Total runtime: {(tclock.time()-t0)/60:.1f} min")
    print(f"\nTo render the chart, run:")
    print(f"  python bayes_cs_plot.py" + (f" --tag {args.tag}" if args.tag else ""))


if __name__ == '__main__':
    main()
