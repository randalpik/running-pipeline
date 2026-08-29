"""Bayesian latent-process model for aerobic 5K-equivalent FITNESS — standalone runner.

Generative model (June 2026 redesign):
    log T5K(t) ~ μ + GP_trend(σ_f_long, ℓ_long) + GP_dev(σ_f_dev, ℓ_dev)
    log t5k_i  ~ Normal(log T5K(t_i), σ_obs)

where t5k_i is each AEROBIC race (≥1500 m) down-converted to its 5K-equivalent
time via the World Athletics tables (identity at 5K). The fit is a single
latent 5K-equiv fitness curve — NOT the old two-parameter CP2 (CS, D') model.

Why: once every aerobic race is homogenized to one distance (5000 m), the CP2
hyperbola t=(d−D')/CS is degenerate — only (5000−D')/CS is identified, not CS
and D' separately. The old fitted D'(t) was already flat (its time-variation
was unidentified), and a single hyperbola can't match IAAF's empirical shape
across 800 m–5K, which over-rated short races. So D' is demoted to a fixed
constant (src/shared/cs_projection.dprime_fixed) used only to back out a
nominal bare-CS = (5000−D')/T5K for the recovery/long-run baselines and to feed
the CP3 sub-1500 sprint projection. Sub-1500 sprints (400/800) are EXCLUDED
from this fit — they're frontier demos, not aerobic-fitness anchors.

Outputs keep the legacy schema (cs_pace/cs_mps/dp_med) for downstream
compatibility: cs_mps = (5000−D'_fixed)/T5K, dp_med = D'_fixed (constant), and
load_cs_outputs derives p5k_implied = the latent 5K-equiv pace.

Uses HSGP (Hilbert space approximation) to make the GP tractable on a long
time series — full-rank Latent GP is O(N³) per leapfrog and infeasible at
500+ grid points without a BLAS-linked PyTensor.

USAGE (local run):
    pip install pymc==5.* arviz pandas numpy scipy plotly
    python src/models/bayes_cs_fit.py

By default, reads `races.csv` from `data/` and writes outputs there too.
Override with --races / --out-dir if needed. There is no exclusion step —
every hard-eligible race informs the fit through the causal-shortfall
weighting (see build_eligible); no manual exclusion file is consumed.

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
    bayes_cs_race_weights.csv     Audit trail: every race's causal shortfall and weight

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
from src.shared.units import METERS_PER_MILE
from src.shared.plot_window import pad_range
from src.shared.recovery_model import (race_physical_correction,
                                       SURFACE_TERRAIN, TRAIL_FRAC)
from src.shared.cs_projection import (dprime_fixed, CP3_IAAF_BOUNDARY_M,
                                      admit_best_per_day)
from src.shared.wa_scoring import wa_5k_equiv_time


DEFAULT_RACES   = str(DATA_DIR / 'races.csv')
DEFAULT_DAILY   = str(DATA_DIR / 'daily.csv')
DEFAULT_OUT_DIR = str(DATA_DIR)


# ---- causal race weighting (Aug 2026) --------------------------------------
# Default scale for the shortfall weight, as a fraction. A race this far below
# the best already demonstrated counts about half. 5% keeps the median weight at
# 0.93 over Max's corpus, puts 21 of 222 races under 0.5 and 5 under 0.25 — i.e.
# it reproduces what the retired tier-1/tier-2 gates were reaching for, without a
# threshold. Interpretable, so pinned rather than fitted.
CAUSAL_SHORTFALL_SCALE = 0.05
# Trailing window over which a demonstrated capability remains the reference.
# Insensitive as a *shortfall* measure (365/548/730 give identical values on
# every decisive race in Max's corpus), but it carries a second, physical
# meaning: after this long without racing the old best stops constraining, which
# is what lets the curve follow a genuine DECLINE across a layoff. A decline
# while racing continuously is still resisted — deliberately, since fitness is
# built faster than it decays (Max, Aug 2026), and no such stretch exists on
# record.
CAUSAL_WINDOW_DAYS = 365
CAUSAL_WEIGHT_DF = 4.0


def causal_race_weights(dates, t5k, *, window_days=CAUSAL_WINDOW_DAYS,
                        scale=CAUSAL_SHORTFALL_SCALE, df=CAUSAL_WEIGHT_DF):
    """Per-race observation weights from CAUSAL shortfall (past-only).

    The failure this fixes: a residual measured against the fitted curve is
    contaminated by the future. In a steeply improving era a July lifetime PR
    reads as "slow" because November was faster, so any robust likelihood
    discounts the very races that define the rise — measured at +70s of
    over-claimed fitness across 2013, with 14 races in it.

    A race can only fall short of a capability ALREADY DEMONSTRATED. So the
    reference is the best 5K-equivalent inside the trailing window, excluding
    the race itself; a race at or under that reference cannot be a shortfall and
    keeps full weight, no matter how slow the curve thinks it is. With no past
    inside the window the shortfall is undefined and the weight is 1 — which is
    also what lets the fit follow a post-layoff decline.

    The weight itself is the Student-t IRLS weight on shortfall/scale,
    normalised to 1 at zero shortfall, applied downstream as a per-race variance
    inflation. The Gaussian family is retained: the robustness comes from WHERE
    the residual is measured, not from a heavy tail (a heavy tail on the model
    residual is exactly what failed).

    Returns (weights, shortfall) — shortfall is NaN where undefined.
    """
    d = np.asarray(pd.to_datetime(pd.Series(dates)).values, dtype='datetime64[D]')
    t = np.asarray(t5k, dtype=float)
    sf = np.full(len(t), np.nan)
    for i in range(len(t)):
        past = (d < d[i]) & (d >= d[i] - np.timedelta64(int(window_days), 'D'))
        if past.any():
            sf[i] = t[i] / np.nanmin(t[past]) - 1.0
    x = np.where(np.isnan(sf), 0.0, np.maximum(sf, 0.0)) / scale
    w = ((df + 1.0) / (df + x ** 2)) / ((df + 1.0) / df)
    return np.clip(w, 0.0, 1.0), sf


def build_eligible(races_path):
    """Load races.csv and apply hard-eligibility (VALIDITY) filters.

    Filters:
        surface != 'Downhill'  (downhill courses are not CS-comparable)
        time_sec >= 120        (anything sub-2-min is data noise)
        one race per day       (the best 5K-equivalent — admit_best_per_day;
                                replaced the old race_seq==1 / fatigued rule)

    These are validity gates only — they say "not a measurement of aerobic
    capability", never "outlier". Outliers are handled continuously by
    causal_race_weights, which replaced the tier-1/tier-2 exclusion rule in
    Aug 2026; there is no exclusion step any more.
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

    # Downhill is CS-ineligible by default — but a watch-covered Downhill race
    # is ADMITTED, because the measured grade correction (§B) discounts its
    # downhill-assisted time to a flat-equivalent that IS CS-comparable. The
    # categorical hard-exclusion remains the pre-watch fallback. (No current
    # Downhill race has watch coverage, so this is a no-op today; it arms the
    # behavior for any future watch-covered downhill course.)
    has_measured = race_physical_correction(races)['has_measured'].to_numpy()
    elig = races[
        ((races['surface'] != 'Downhill') | has_measured) &
        (races['time_sec'] >= 120)
    ].copy().sort_values('date')
    # Multi-race days contribute their BEST 5K-equivalent race, not whichever
    # ran first. Applied AFTER the hard filters above so the day's winner is
    # chosen among genuinely eligible races — that ordering is what admits
    # 2023-08-02, whose only non-400 race is race_seq 3.
    elig = admit_best_per_day(elig).reset_index(drop=True)
    return elig


def main():
    p = argparse.ArgumentParser(description=(__doc__ or '').split('\n\n')[0])
    p.add_argument('--races', default=DEFAULT_RACES,
                   help=f'Path to races.csv (default: {DEFAULT_RACES})')
    p.add_argument('--daily', default=DEFAULT_DAILY,
                   help=f'Path to daily.csv (default: {DEFAULT_DAILY}). Used '
                        'only to extend the inference grid to cover the latest '
                        'logged run (not just the latest race).')
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
    # --- Layer-1 likelihood (Aug 2026 re-evaluation; see
    #     ~/.claude/plans/fitness-model-reevaluation.md) ---
    p.add_argument('--causal-scale', type=float, default=CAUSAL_SHORTFALL_SCALE,
                   help=f'causal weights: shortfall scale (default '
                        f'{CAUSAL_SHORTFALL_SCALE}; a race this far below its '
                        f'recent best counts ~half)')
    p.add_argument('--causal-window', type=int, default=CAUSAL_WINDOW_DAYS,
                   help=f'causal weights: trailing window in days (default '
                        f'{CAUSAL_WINDOW_DAYS}); also sets how long a '
                        f'demonstrated capability keeps constraining')
    p.add_argument('--target-accept', type=float, default=0.95,
                   help='NUTS target_accept (default 0.95; the one-sided '
                        'likelihood wants 0.99).')
    p.add_argument('--sigma-base-prior', type=float, default=0.02,
                   help='HalfNormal σ for sigma_base (default 0.02)')
    p.add_argument('--ell-cs-mu', type=float, default=0.25,
                   help='LogNormal location (in years) for ell_cs (default 0.25)')
    p.add_argument('--ell-cs-sigma', type=float, default=0.4,
                   help='LogNormal scale (log-space) for ell_cs (default 0.4)')
    p.add_argument('--xc-correction', type=float, default=0.06,
                   help='Terrain-scaled flat correction applied to no-watch '
                        'off-road race times before fitting (default 0.06 = '
                        '6%% at full trail: XC gets the full percent, Offroad '
                        'half). Pre-model adjustment: time_sec is divided by '
                        '(1 + c*terrain_frac) so those races enter the model '
                        'as if they were equivalent flat-course times; '
                        'grade-measured races skip it (the physical '
                        'correction supersedes). Set to 0 to disable. '
                        'Iterate based on visual fall-vs-spring alignment '
                        'in the chart.')
    p.add_argument('--tag', default='',
                   help='Suffix for output filenames (e.g. "v4a") to keep '
                        'experiments separate')
    p.add_argument('--workout-obs', default='',
                   help='EXPERIMENTAL, default off (workout-enrichment of the '
                        'CS likelihood was halted on level-bias — see '
                        'docs/cs-model-reference.md): path to a CSV of near-race training '
                        'observations with columns date, t5k_sec, dp_fixed_m, '
                        'sigma_obs. Each row enters the likelihood as a '
                        '5K-equivalent effort at its date: '
                        'log(t5k_sec) ~ N(log((5000 - dp_fixed_m)/CS(t)), '
                        'sigma_obs). dp_fixed_m is the RACE-FIT D\' median at '
                        'that date (fixed, not the model\'s D\' — workouts '
                        'inform the CS curve only, races stay the sole D\' '
                        'anchor; Gate 2 of the enrichment plan). No beta_long '
                        '(5000m < d_thresh). Default off: race-only fit, '
                        'output unchanged.')
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
    # don't clobber each other's outputs.
    suffix = f"_{args.tag}" if args.tag else ""

    # Fixed D' (m): the CP2 (CS,D') decomposition was retired (see module
    # docstring); D' is a constant used only to back out a nominal bare-CS from
    # the latent 5K-equiv fitness and to feed the CP3 sub-1500 sprint leg.
    D_FIXED = dprime_fixed()
    print(f"Fixed D' (nominal bare-CS backout + sprint leg): {D_FIXED:.0f} m")

    # ---------- load + filter ----------
    elig_full = build_eligible(args.races)
    print(f"Hard-eligible races (post Downhill/<120s/best-per-day filter): {len(elig_full)}")

    elig = elig_full.reset_index(drop=True)
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

    # ---------- physical route correction (grade + footing + altitude) ----------
    # Convert each watch-covered race to its flat / sea-level / smooth-equivalent
    # TIME before it enters the likelihood, so CS measures fitness not the
    # course (docs/cs-model-reference.md "Race physical correction"). MUST match the same
    # helper in cs_projection (the displayed diamonds).
    # Net-downhill races get time ADDED (Boston discounted); net-uphill /
    # altitude races credited faster. Subtracted BEFORE the β_long un-bias.
    phys = race_physical_correction(elig)
    elig['phys_dt_sec'] = phys['dt_sec'].to_numpy()
    elig['time_sec'] = elig['time_sec'].astype(float) - elig['phys_dt_sec']
    if 'pace_sec_per_mi' in elig.columns:
        # Keep pace consistent with the corrected time — the likelihood reads
        # only time_sec, but this path and cs_projection must stay identical.
        elig['pace_sec_per_mi'] = (elig['pace_sec_per_mi'].astype(float)
                                   - phys['total_s_per_mi'].to_numpy())
    has_measured = phys['has_measured'].to_numpy()
    n_meas = int(has_measured.sum())
    if n_meas:
        moved = elig['phys_dt_sec'][has_measured]
        print(f"Physical route correction: {n_meas} grade-measured races "
              f"(dt {moved.min():+.0f}..{moved.max():+.0f}s, "
              f"median {moved.median():+.1f}s)")

    # Terrain-scaled categorical flat correction — the no-watch branch of the
    # binary system (measured grade+footing supersedes it): XC (trail) gets
    # the full percent, Offroad (mixed) half, everything else nothing.
    flat_frac = (elig['surface'].fillna('').astype(str).str.strip().str.lower()
                 .map(SURFACE_TERRAIN).fillna('paved').map(TRAIL_FRAC)).to_numpy()
    flat_mask = (flat_frac > 0) & ~has_measured
    n_flat = int(flat_mask.sum())
    if args.xc_correction > 0 and n_flat > 0:
        factor = 1.0 / (1.0 + args.xc_correction * flat_frac[flat_mask])
        elig.loc[flat_mask, 'time_sec'] = elig.loc[flat_mask, 'time_sec'] * factor
        if 'pace_sec_per_mi' in elig.columns:
            elig.loc[flat_mask, 'pace_sec_per_mi'] = elig.loc[flat_mask, 'pace_sec_per_mi'] * factor
        print(f"Flat terrain pre-correction: {n_flat} races "
              f"(c={args.xc_correction:.3f} = {args.xc_correction*100:.1f}% at "
              f"full trail; Offroad scaled ×{TRAIL_FRAC['mixed']:g})")
    elif n_flat > 0:
        print(f"Flat terrain pre-correction disabled (--xc-correction 0); "
              f"{n_flat} off-road races kept as-is")

    # ---------- inference grid ----------
    # Span the UNION of eligible-race dates and daily-run dates, padded ~2% on
    # each side. This (a) covers every logged run — not just the last race — so
    # the CS posterior is defined wherever a recovery/quality run exists, and
    # (b) extends a little past both data ends so downstream plots can draw the
    # CS line all the way to their axis edges. The HSGP long-trend extrapolates
    # cleanly into the margin (see the model boundary comment below).
    span_lo = pd.Timestamp(elig['date'].min())
    span_hi = pd.Timestamp(elig['date'].max())
    if os.path.exists(args.daily):
        daily_df = pd.read_csv(args.daily, parse_dates=['date'])
        if len(daily_df):
            span_lo = min(span_lo, pd.Timestamp(daily_df['date'].min()))
            span_hi = max(span_hi, pd.Timestamp(daily_df['date'].max()))
    else:
        print(f"WARNING: daily CSV not found at {args.daily}; "
              f"grid spans eligible races only.", file=sys.stderr)
    pad_lo, pad_hi = pad_range(span_lo, span_hi, 0.02)
    # The 2% pad is proportional to history length: ~4 months on an 18-year
    # profile but under a week on a first-season one, whose CS/frontier lines
    # then die days after the fit ran. Floor the FORWARD margin at 60 days so
    # every profile's lines stay drawable between fits (long histories already
    # exceed it; the HSGP long-trend extrapolates cleanly into the margin).
    pad_hi = max(pad_hi, span_hi + pd.Timedelta(days=60))
    first_d = pad_lo.date()
    last_d  = pad_hi.date()
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

    # ---------- fit corpus: aerobic races only, IAAF-homogenized to 5K --------
    # The fit is now a single latent 5K-equivalent FITNESS curve (see module
    # docstring). Every aerobic race (>=1500m) is down-converted to its 5K-equiv
    # via the World Athletics tables (identity at 5K) and enters as one
    # observation; sub-1500 sprints (400/800) are EXCLUDED — IAAF's population
    # equivalence distorts a distance specialist there, and they're frontier
    # demos, not fitness anchors (Max, June 2026). All observations now sit at
    # one distance, so the old CP2 (CS,D') hyperbola is degenerate; D' is no
    # longer fitted (fixed constant, used only downstream).
    fit_df = elig[elig['distance_m'] >= CP3_IAAF_BOUNDARY_M].reset_index(drop=True)
    n_sprint = int((elig['distance_m'] < CP3_IAAF_BOUNDARY_M).sum())
    print(f"Fit corpus: {len(fit_df)} aerobic races (>={CP3_IAAF_BOUNDARY_M:.0f}m); "
          f"{n_sprint} sub-1500 sprints excluded (frontier demos only)")

    # Map race dates to grid indices (nearest grid point)
    race_grid_idx = np.array([
        min(int(round((rd - first_d).days / args.grid_step)), n_grid - 1)
        for rd in fit_df['date']
    ])

    # Original distance/time kept for diagnostics; the likelihood sees the
    # 5K-equivalent time. Times are already physical/XC-corrected above, so the
    # WA score sees the corrected time.
    race_distances = fit_df['distance_m'].to_numpy().astype(float)
    race_times = fit_df['time_sec'].to_numpy().astype(float)
    t5k_equiv = np.array([wa_5k_equiv_time(float(d), float(t))
                          for d, t in zip(race_distances, race_times)])
    log_t5k = np.log(t5k_equiv)
    print(f"IAAF 5K-equiv: {len(fit_df)} aerobic races homogenized "
          f"(identity at 5K; WA tables for 1500m-marathon).")

    # Causal race weights (see causal_race_weights). Applied as a per-race
    # variance inflation sigma_i = sigma_base / sqrt(w_i) — standard weighted
    # regression, so a low-weight race is simply a less informative
    # observation rather than an excluded one. Written out per race so the
    # judgement is readable: "this race was N% slower than anything you had run
    # in the previous year, so it counted w".
    race_w, race_sf = causal_race_weights(
        fit_df['date'], t5k_equiv,
        window_days=args.causal_window, scale=args.causal_scale)
    wdf = pd.DataFrame({
        'date': pd.to_datetime(fit_df['date']).dt.date,
        'distance_m': fit_df['distance_m'].astype(int),
        'event': fit_df['event'] if 'event' in fit_df.columns else '',
        'surface': fit_df['surface'] if 'surface' in fit_df.columns else '',
        't5k_equiv_sec': np.round(t5k_equiv, 1),
        'causal_shortfall_pct': np.round(race_sf * 100, 2),
        'weight': np.round(race_w, 4)})
    wpath = os.path.join(args.out_dir, f'bayes_cs_race_weights{suffix}.csv')
    wdf = wdf.sort_values('weight').reset_index(drop=True)
    wdf.to_csv(wpath, index=False)
    print(f"Causal weights: scale={args.causal_scale:.3f} "
          f"window={args.causal_window}d | median {np.median(race_w):.3f}, "
          f"{int((race_w < 0.5).sum())} races < 0.5, "
          f"{int((race_w < 0.25).sum())} < 0.25 -> {wpath}")
    for _, r in wdf.head(5).iterrows():
        print(f"    {r['date']} {int(r['distance_m']):>5}m "
              f"{str(r['event'])[:30]:<30} shortfall "
              f"{r['causal_shortfall_pct']:+6.1f}%  weight {r['weight']:.3f}")

    # ---------- optional near-race workout observations (spike) ----------
    wobs = None
    if args.workout_obs:
        wobs = pd.read_csv(args.workout_obs, parse_dates=['date'])
        wobs_grid_idx = np.array([
            min(int(round((wd.date() - first_d).days / args.grid_step)), n_grid - 1)
            for wd in wobs['date']
        ])
        wobs_dp = wobs['dp_fixed_m'].to_numpy().astype(float)
        wobs_log_t = np.log(wobs['t5k_sec'].to_numpy().astype(float))
        wobs_sigma = wobs['sigma_obs'].to_numpy().astype(float)
        print(f"Workout observations: {len(wobs)} from {args.workout_obs} "
              f"(sigma_obs {wobs_sigma.min():.4f}-{wobs_sigma.max():.4f}; "
              f"D' fixed from race fit, CS-only likelihood)")

    # Center grid_t for HSGP numerical stability
    grid_t_centered = (grid_t - grid_t.mean()) / 365.0  # in years
    L = (grid_t_centered.max() - grid_t_centered.min()) * 1.5  # boundary buffer
    X_grid = grid_t_centered.reshape(-1, 1)
    print(f"Grid spans {grid_t_centered.min():.2f} to {grid_t_centered.max():.2f} years (HSGP L={L:.2f})")

    # ---------- model ----------
    with pm.Model() as model:
        # Single latent log-5K-equivalent-time fitness curve (additive GPs):
        #   log T5K(t) = mu_fit + log_fit_trend(t) + log_fit_dev(t)
        # trend = slow career arc (years scale); dev = training-cycle wiggles
        # (months scale). At boundaries (before first / past last data) the
        # SHORT dev GP decays to zero while the LONG trend persists, anchoring
        # extrapolation to a smooth trajectory rather than the most recent dip.
        # (Replaces the old two-GP CS+D' hyperbolic model — see module docstring.)
        mu_fit = pm.Normal('mu_fit', mu=np.log(950.0), sigma=0.3)

        sf_fit_long  = pm.HalfNormal('sf_fit_long',  sigma=0.3)
        ell_fit_long = pm.LogNormal('ell_fit_long', mu=np.log(5.0), sigma=0.5)
        sf_fit_dev   = pm.HalfNormal('sf_fit_dev',   sigma=0.10)
        ell_fit_dev  = pm.LogNormal('ell_fit_dev',
                                    mu=np.log(args.ell_cs_mu),
                                    sigma=args.ell_cs_sigma)

        # Observation noise. UNIFORM now: every observation is a 5K-equivalent
        # at one distance, so the old distance-scaling term (alpha_sig) is gone.
        # CLI-controlled via --sigma-base-prior. Default HN(0.02).
        # (XC/physical bias is still handled OUTSIDE the model via time
        # pre-correction; >5K and 1500-5K aerobic conversion is the WA table.)
        sigma_base = pm.HalfNormal('sigma_base', sigma=args.sigma_base_prior)

        # HSGP — Hilbert space approximation, much cheaper than full Latent.
        cov_long = sf_fit_long ** 2 * pm.gp.cov.Matern52(input_dim=1, ls=ell_fit_long)
        cov_dev  = sf_fit_dev  ** 2 * pm.gp.cov.Matern52(input_dim=1, ls=ell_fit_dev)
        # Long-scale trend uses fewer basis functions (ell_long ~5y, smooth modes).
        m_long = max(20, args.m_basis // 3)
        gp_long = pm.gp.HSGP(m=[m_long], L=[L], cov_func=cov_long)
        gp_dev  = pm.gp.HSGP(m=[args.m_basis], L=[L], cov_func=cov_dev)

        log_fit_trend = gp_long.prior('log_fit_trend', X=X_grid)
        log_fit_dev   = gp_dev.prior('log_fit_dev',   X=X_grid)
        log_t5k_total = mu_fit + log_fit_trend + log_fit_dev

        # Likelihood.
        #
        # Per-race variance inflation from the causal weights (1.0 = no change).
        sigma_obs_scale = 1.0 / np.sqrt(race_w)
        # 'normal' (historical): log t5k ~ N(latent, sigma). Symmetric, so the
        # latent is the conditional MEAN of race performance — a race is held as
        # likely to land 3% faster than fitness as 3% slower. That is coherent
        # for an average and incoherent for capability, and it is what forced the
        # one-sided auto-exclusion gates: with a symmetric likelihood the long
        # slow tail drags the curve, so slow races have to be trimmed by hand.
        # Measured on 216 kept races: skew +1.24, slow tail 1.81x the fast tail,
        # 59.3% of races FASTER than the curve, and all 6 exclusions slow ones.
        #
        # 'shortfall': every race is a max-effort attempt, so
        #     log t_race = log t_capability + s + eps,   s >= 0
        # with eps ~ N(0, sigma_meas) for timing/course precision. The latent is
        # then CAPABILITY. A slow race is cheap to explain (large s) regardless
        # of cause — fatigue, heat, illness, a bad day — so it barely moves the
        # curve, which is the point: no per-cause gate can cover the reasons that
        # were never logged. A fast race stays expensive and so stays
        # influential, which is the wanted fragility: fast races are the audited
        # ones, and an over-corrected or short-course input should be loud.
        #
        # s ~ Exponential convolved with the Gaussian eps is exactly the
        # exponentially-modified Gaussian, which has a closed-form logp — so
        # there is no per-race latent variable and NUTS sees no hard boundary.
        # nu is the mean shortfall; it is identified by the SKEW of the residual
        # distribution, which is why sigma_meas has to be pinned rather than fit.
        pm.Normal('obs', mu=log_t5k_total[race_grid_idx],
                  sigma=sigma_base * sigma_obs_scale, observed=log_t5k)

        # Near-race workout observations (spike; experimental, off by default):
        # 5K-equivalent efforts entering the same latent-fitness likelihood.
        if wobs is not None:
            pm.Normal('obs_workout', mu=log_t5k_total[wobs_grid_idx],
                      sigma=wobs_sigma, observed=wobs_log_t)

    # ---------- prior predictive ----------
    print("\nRunning prior predictive (200 samples)...")
    with model:
        prior_pred: Any = pm.sample_prior_predictive(samples=200, random_seed=args.seed)
    log_t5k_pp = (prior_pred.prior['log_fit_trend'].values +
                  prior_pred.prior['log_fit_dev'].values +
                  prior_pred.prior['mu_fit'].values[..., None])
    pace_pp = np.exp(log_t5k_pp) / (5000.0 / METERS_PER_MILE) / 60
    print(f"  Prior 5K-equiv pace (min/mi): "
          f"5%={np.percentile(pace_pp,5):.2f}  median={np.median(pace_pp):.2f}  "
          f"95%={np.percentile(pace_pp,95):.2f}")

    # ---------- posterior ----------
    print(f"\nSampling NUTS: tune={args.tune} draws={args.draws} chains={args.chains}")
    t0 = tclock.time()
    with model:
        trace: Any = pm.sample(
            draws=args.draws, tune=args.tune,
            chains=args.chains, cores=min(args.chains, os.cpu_count()),
            target_accept=args.target_accept, random_seed=args.seed,
            return_inferencedata=True,
        )
    elapsed = tclock.time() - t0
    print(f"Sampling done in {elapsed/60:.1f} min")
    # Likelihood-parameter posterior, always printed: these are the knobs the
    # Layer-1 re-evaluation turns, and reading them should not require
    # --diagnostics.
    _lik = [v for v in ('sigma_base',)
            if v in trace.posterior]
    if _lik:
        print('\nLikelihood parameters:')
        print(az.summary(trace, var_names=_lik)[
            ['mean', 'sd', 'hdi_3%', 'hdi_97%', 'ess_bulk', 'r_hat']].to_string())

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
            f.write(f"target_accept: {args.target_accept}\n")
            f.write(f"xc_correction: {args.xc_correction:.4f} ({args.xc_correction*100:.1f}% terrain penalty)\n")

            f.write(f"\n=== Causal race weights ===\n")
            f.write(f"scale={args.causal_scale:.3f} window={args.causal_window}d | "
                    f"median {np.median(race_w):.3f}, "
                    f"{int((race_w < 0.5).sum())} races < 0.5, "
                    f"{int((race_w < 0.25).sum())} < 0.25 "
                    f"(of {len(race_w)} in the fit corpus)\n")
            f.write(f"\n{'date':<12s} {'dist':>6s} {'shortfall':>10s} "
                    f"{'weight':>7s}  event\n")
            for _, r in wdf.head(15).iterrows():
                f.write(f"{str(r['date']):<12s} {int(r['distance_m']):>6d} "
                        f"{r['causal_shortfall_pct']:>9.2f}% {r['weight']:>7.3f}  "
                        f"{str(r.get('event',''))[:40]}\n")

            f.write(f"\n=== Hyperparameter posterior summary ===\n")
            summ = az.summary(trace, var_names=['mu_fit',
                                                 'sf_fit_long', 'ell_fit_long',
                                                 'sf_fit_dev',  'ell_fit_dev',
                                                 'sigma_base'])
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
    # Latent quantity is log 5K-equivalent TIME. Back out the legacy schema:
    #   cs_mps = (5000 - D_FIXED)/t5k   (nominal bare-CS for recovery/long-run)
    #   dp_med = D_FIXED (constant; D' no longer fitted)
    # load_cs_outputs then derives p5k_implied = the latent 5K-equiv pace.
    log_t5k_trend_post = trace.posterior['log_fit_trend'].values
    log_t5k_dev_post   = trace.posterior['log_fit_dev'].values
    mu_fit_post        = trace.posterior['mu_fit'].values[..., None]
    log_t5k_post       = mu_fit_post + log_t5k_trend_post + log_t5k_dev_post

    t5k_flat       = np.exp(log_t5k_post).reshape(-1, n_grid)            # 5K-equiv time (s)
    # Trend-only (mu + slow component) for separate visualization/diagnostic
    t5k_trend_flat = np.exp(mu_fit_post + log_t5k_trend_post).reshape(-1, n_grid)
    cs_flat        = (5000.0 - D_FIXED) / t5k_flat                       # nominal bare CS (m/s)
    cs_trend_flat  = (5000.0 - D_FIXED) / t5k_trend_flat
    pace_flat       = METERS_PER_MILE / cs_flat / 60
    pace_trend_flat = METERS_PER_MILE / cs_trend_flat / 60

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
            # D' is a fixed constant now (not fitted) — emit it flat so the
            # schema and dp_med consumers (cp3_dprime, p5k_implied) keep working.
            'dp_med':  D_FIXED,
            'dp_lo50': D_FIXED,
            'dp_hi50': D_FIXED,
            'dp_lo95': D_FIXED,
            'dp_hi95': D_FIXED,
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(summary_path, index=False)
    print(f"Wrote {summary_path} ({len(summary_df)} rows)")

    # ---------- per-race residuals + posterior predictive (diagnostics only) ----------
    if args.diagnostics:
        # Residual = observed 5K-equiv time vs the latent fitness prediction at
        # the race date. t5k_flat is the latent 5K-equiv time; cs_flat is the
        # nominal bare-CS backed out from it (see summary block).
        t5k_pred_med = np.median(t5k_flat[:, race_grid_idx], axis=0)
        cs_at_race_med = np.median(cs_flat[:, race_grid_idx], axis=0)
        pct_resid = (t5k_equiv / t5k_pred_med - 1) * 100

        resid_df = pd.DataFrame({
            'date': fit_df['date'].values,
            'distance_m': race_distances,
            'actual_sec': t5k_equiv,                  # 5K-equiv that entered the fit
            'actual_sec_original': fit_df['time_sec_original'].values,
            'surface': fit_df['surface'].values if 'surface' in fit_df.columns else [''] * len(fit_df),
            'predicted_sec': t5k_pred_med,            # latent 5K-equiv prediction
            'pct_resid': pct_resid,
            'cs_pace_med_at_race': METERS_PER_MILE / cs_at_race_med / 60,
            'dp_med_at_race': D_FIXED,
            'event': fit_df['event'].values if 'event' in fit_df.columns else [''] * len(fit_df),
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
            # Posterior-predictive log-time mean = the latent 5K-equiv time at
            # each race date; σ is uniform (sigma_base) now (no distance scaling).
            t5k_post = t5k_flat[:, race_grid_idx]   # (samples, races)
            sb_post = trace.posterior['sigma_base'].values.reshape(-1)
            mean_log_t = np.log(t5k_post)  # (samples, races)
            # Draw one log_t per (sample, race): mean + σ * N(0,1)
            rng = np.random.default_rng(args.seed)
            eps = rng.standard_normal(mean_log_t.shape)
            pp_log_t = mean_log_t + sb_post[:, None] * eps   # (samples, races)
            # Per-race quantiles
            q025 = np.percentile(pp_log_t, 2.5, axis=0)
            q975 = np.percentile(pp_log_t, 97.5, axis=0)
            q25  = np.percentile(pp_log_t, 25,   axis=0)
            q75  = np.percentile(pp_log_t, 75,   axis=0)
            actual_log_t = np.log(t5k_equiv)
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
        # Retired under the hybrid (June 2026): >5K efforts down-convert via the
        # WA tables before the fit, so no in-model fade is applied. Written as 0
        # so any consumer still reading beta_long applies a no-op factor.
        'beta_long_med': 0.0,
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
