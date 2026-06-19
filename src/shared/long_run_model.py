"""Long-run residual regression: ``raw_resid ~ physical route + covariates``.

Fits temperature / recent-race-fatigue covariates (and any FITTED physical
terms in ``LR_PHYS_FITTED``, empty by default) on long-run pace residuals (vs
CS-implied), with iterative MAD-based outlier prune. Elevation is no longer
fitted here — it's priced per-run upstream in ``workouts.project_long_runs``
(physical grade engine) and already lives in ``raw_resid``. Output is a
SimpleNamespace with ``intercept``, ``phys_coefs``, ``cov_coefs``,
``rsquared``, ``resid_sd``, ``n_kept``.

Distance carries NO term (June 2026): a threshold/knot sweep under the
physical model showed the old 21mi bin was worse than no distance term
(ΔAIC +0.8), and every apparently-better distance term was era
composition in disguise — within-era distance slopes flip sign between
eras (the 20.5mi Nashville staple vs the 24.5mi Chicago staple encode
effort policy, not distance physiology). The lr_lo/lr_hi labels are fully
retired: the TQ legend renders a single "Long" entry and the dashboard's
long-run prediction uses a recency-weighted mean over all familiar-route
long runs (the per-bin empirical means differed by 0.3 s/mi — the split
conditioned on nothing). Full sweep results are recorded in
docs/training-quality-reference.md.

Physical terms replaced per-route dummies in June 2026. Empirical route
betas were almost perfectly confounded with training era: every named route
lives in one contiguous era with its own long-run effort policy, so a
route's beta encoded "how hard I typically ran long runs that year", not
terrain — physically near-identical flat sea-level routes (south lakefront,
north greenway) carried betas 31 s/mi apart, and a moderate 2026 effort
could out-rank an all-time-best 2023 run after correction. Physical terms
only correct what effort can't explain (you can't choose your altitude),
leaving effort/era visible in the residuals. The ``route`` column and
``qualifying_routes`` (locations with ≥ ``MIN_ROUTE_N`` in-slice runs) are
still produced as DESCRIPTIVE labels — the dashboard's familiar-route
filter and outlier reporting use them — but the model no longer fits
per-route coefficients.

Temperature and race-fatigue are NO LONGER fit here (June 2026). They are
PINNED from the pooled recovery+long-run fit (``physical_route_betas`` — the
same single source of truth that already supplies footing/altitude), exactly
the way the recovery model pins them. The old per-model long-run fit of these
was indefensible complexity: the free temperature slope just reproduced
recovery's, the race-fatigue term was near-dead (only ~4 long runs carry
meaningful marathon load; leave-one-out swung the beta 1.4–8.6), and the betas
lurched with every model change. The earlier thesis — that temperature and
fatigue hit long runs differently from recovery because the effort regime
differs — was abandoned: the pool's ``is_long`` level dummy already absorbs the
mean-level difference between corpora, leaving a single shared slope per
channel, and the long-run intercept/level is display-dead anyway. So the
contrast is identified on the full ~2,300-row combined corpus instead of a
handful of long runs. ``cov_coefs`` carries the pinned per-category
``temp_centered`` / ``fat_marathon`` / ``fat_race_short`` keys so downstream
consumers (``transferable_contributions``, the TQ sidebar) see the familiar
shape. Time of day, a strong recovery factor, is dead on long runs (t≈0.25) and
is not in the pool's transferable set, so it never enters here.

ROLE CHANGED (June 2026): the INTERCEPT is display-dead. Long runs enter
TQ at their race-equivalent raw residual (``project_long_runs`` applies
the CS fit's β_long un-bias) minus this model's PHYSICAL + COVARIATE
contributions only (``phys_contrib + cov_contrib`` — verified, physically
grounded, always applied; Max intends to test the same template on
workouts). The intercept/level is fit (it centers the regressors and
robustifies the prune) but NEVER subtracted from any 5K-equivalent
position — a constant subtracted from every long run claims a long run
out-predicts a race at the same distance/pace, which is physically
indefensible (Max's no-class-constants contract). Long-run TQ pruning
rides the shared track-relative prune; this fit's internal MAD prune only
robustifies its betas. Other consumers: the Long Runs plot's Normalize
toggle (covariate betas) and the Dashboard's familiar-route
recency-weighted residual (``route`` labels + ``is_outlier``).
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.shared.paths import DATA_DIR
from src.shared.plot_window import clip_to_daily_floor
from src.shared.recovery_model import (quality_category_dates,
                                       add_quality_features,
                                       physical_route_betas,
                                       temp_centered_feature,
                                       QUALITY_CATS)

# Min long runs per location for the DESCRIPTIVE 'route' label (locations
# below this fold into 'other'). No longer a model term — see module
# docstring; the dashboard's familiar-route filter keys off this.
MIN_ROUTE_N = 5
# Iterative MAD-based outlier threshold used during the OLS fit.
PRUNE_SIGMA = 3.0
# Covariate + physical terms enter the fit only when the in-slice sample
# can support the extra parameters; below this the model stays bin-only
# (watch profiles with a handful of long runs).
MIN_COV_N = 30

# Transferable-state covariates (temperature heat slope + per-category
# race-fatigue decays) are PINNED from the pooled fit (physical_route_betas),
# no longer fit here — see the module docstring. Kept as a named list so the
# pinned keys and the contribution loop stay in one place.
LR_COVARIATES = ['temp_centered'] + [f'fat_{c}' for c in QUALITY_CATS]
# Physical route terms fitted here. ELEVATION IS NO LONGER ONE OF THEM
# (June 2026, watch-stream-enrichment §A): the route-constant median-centered
# slope (0.17·elev_per_mile) that used to live on this residual scale was
# replaced by the per-run, effort-aware physical grade engine, applied
# upstream as a TIME correction in workouts.project_long_runs — so raw_resid
# already carries the flat-equivalent grade correction by the time it reaches
# this fit. Pricing grade on the run's own pace (before the β un-bias and the
# hyperbola), with measured per-mile gain/loss and a terrain/effort-aware
# descent refund, is strictly better than a single pinned slope on a
# balanced-route constant centered at the era median.
#
# LR_PHYS_FITTED is the extension point for FITTED physical terms (footing,
# altitude). Empty by default: altitude was a fitted term until June 2026,
# REMOVED (Max) — on the race-equivalent residual scale its beta flipped sign
# (−0.47 s/mi per 1000 ft, "altitude makes you faster"), the same era/route
# confounding that sank the route dummies (Boulder-era long runs are also
# Boulder-era effort policy). A quality-workout cross-check found it dead too
# (β −0.37, t −0.6). Any term added here must clear the identification battery
# (LOO β-swing, sign sanity, count support, AIC vs no-term, collinearity with
# the upstream grade cost) and Max's explicit approval before it ships.
LR_PHYS_FITTED: list = []


def load_quality_dates():
    """Quality-day dates (marathon / short race) from the profile's daily.csv
    and races.csv — the recovery model's categorization, reused here for the
    long-run fatigue covariates. Filtered to the logging era so the dates
    match what the recovery fit sees."""
    daily = pd.read_csv(DATA_DIR / 'daily.csv', parse_dates=['date'])
    daily = clip_to_daily_floor(daily)
    races = pd.read_csv(DATA_DIR / 'races.csv', parse_dates=['date'])
    return quality_category_dates(daily, races)


def fit_long_run_model(lr_in, quality_dates=None):
    """Fit the (display-dead) long-run LEVEL on the in-slice long runs, with
    the transferable-state covariates (temperature + per-category race-fatigue
    decay) and any ``LR_PHYS_FITTED`` physical terms PINNED — not fit — from the
    pooled recovery+long-run estimator (``physical_route_betas``; see module
    docstring). Only the intercept (+ any ``LR_PHYS_FITTED``, empty by default —
    elevation is priced upstream in project_long_runs, not here) is fit, via the
    same iterative MAD-based outlier prune at ``PRUNE_SIGMA``. Covariates enter
    only when the sample has at least ``MIN_COV_N`` rows (preserving the
    sparse-profile "no state correction" behavior the Normalize toggle keys
    off). Distance carries no term — see module docstring.

    ``quality_dates`` is the per-category race-date dict from
    ``load_quality_dates()`` / ``recovery_model.quality_category_dates``;
    pass it when the caller already has it, else it's loaded from the
    profile's data dir.

    Returns ``(lr_in_augmented, fit, qualifying_routes)``. ``fit`` is a
    SimpleNamespace with ``intercept``, ``phys_coefs`` (dict), ``cov_coefs``
    (dict of the pinned per-category ``temp_centered`` / ``fat_marathon`` /
    ``fat_race_short`` keys, empty below ``MIN_COV_N``), ``rsquared``,
    ``resid_sd``, ``n_kept``, ``temp_ref``.

    The augmented frame carries ``route`` (descriptive label only), per-row
    ``phys_contrib``, ``cov_contrib``, ``model_offset``, ``corrected``, and
    an ``is_outlier`` flag.
    """
    loc_counts = lr_in['location'].value_counts()
    qualifying_routes = sorted(loc_counts[loc_counts >= MIN_ROUTE_N].index)
    if quality_dates is None:
        quality_dates = load_quality_dates()
    lr_in = add_quality_features(lr_in, quality_dates)
    # One-sided heat hinge max(0, air_temp − 6C), centered on the median hinge
    # (typical-day reference, like wind and the recovery model) so the
    # temperature ADJUSTMENT is symmetric around a normal day rather than
    # one-directional off the cold floor. Missing temp -> 0 so no row drops for
    # lacking a thermometer reading. Same shape AND slope as recovery now: the
    # slope is pinned from the pool below, not fit here.
    lr_in['temp_centered'] = temp_centered_feature(lr_in).fillna(0.0)
    temp_ref = float(lr_in['temp_centered'].median())
    lr_in['temp_centered'] = lr_in['temp_centered'] - temp_ref
    lr_in['route'] = lr_in['location'].where(
        lr_in['location'].isin(qualifying_routes), 'other')

    # Transferable-state covariates (temp + per-category fatigue) are PINNED
    # from the pooled recovery+long-run fit — identified on the full corpus, not
    # re-fit on the handful of in-slice long runs (see module docstring). Gated
    # by MIN_COV_N only to preserve the sparse-profile "no state correction"
    # behavior; the pool returns ~0 for such profiles anyway. LR_PHYS_FITTED
    # (footing/altitude) stays an empty extension point fit here; each such term
    # additionally requires variation (a constant column can't be fit).
    if len(lr_in) >= MIN_COV_N:
        pooled = physical_route_betas()
        cov_coefs = {c: float(pooled.get(c, 0.0)) for c in LR_COVARIATES}
        phys_cols = [c for c in LR_PHYS_FITTED if lr_in[c].nunique() > 1]
    else:
        cov_coefs, phys_cols = {}, []

    lr_in['cov_contrib'] = sum((b * lr_in[c] for c, b in cov_coefs.items()),
                               pd.Series(0.0, index=lr_in.index))

    # The only FITTED term is the (display-dead) level — intercept + any
    # LR_PHYS_FITTED — on raw_resid with the pinned covariate offset removed
    # (the way recovery subtracts its pinned offset). The iterative MAD prune
    # still runs: it produces is_outlier, used by the Long Runs plot + dashboard.
    X = pd.concat([
        pd.Series(1.0, index=lr_in.index, name='Intercept'),
        lr_in[phys_cols].astype(float),
    ], axis=1)
    y = (lr_in['raw_resid'] - lr_in['cov_contrib']).astype(float)

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

    coef_map = {str(name): float(c) for name, c in zip(X.columns, coef)}
    intercept = coef_map['Intercept']
    phys_coefs = {c: coef_map[c] for c in phys_cols}

    lr_in['phys_contrib'] = sum((coef_map[c] * lr_in[c] for c in phys_cols),
                                pd.Series(0.0, index=lr_in.index))
    lr_in['model_offset'] = (intercept + lr_in['phys_contrib']
                             + lr_in['cov_contrib'])
    lr_in['corrected'] = lr_in['raw_resid'] - lr_in['model_offset']
    lr_in['is_outlier'] = lr_in.index.isin(pruned)

    keep_mask = ~lr_in.index.isin(pruned)
    # R² against the RAW residual variance, with the pinned covariates + fitted
    # level as the prediction (so the pinned terms' explanatory share counts).
    raw_kept = lr_in['raw_resid'].astype(float).to_numpy()[keep_mask]
    pred_raw = (X.values @ coef + lr_in['cov_contrib'].to_numpy())[keep_mask]
    ss_res = float(np.sum((raw_kept - pred_raw) ** 2))
    ss_tot = float(np.sum((raw_kept - raw_kept.mean()) ** 2))
    n_kept = int(keep_mask.sum())
    p = X.shape[1]
    fit = SimpleNamespace(
        intercept=intercept,
        phys_coefs=phys_coefs,
        cov_coefs=cov_coefs,
        rsquared=1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan'),
        resid_sd=float(np.sqrt(ss_res / (n_kept - p))) if n_kept > p else float('nan'),
        n_kept=n_kept,
        temp_ref=temp_ref,
    )
    return lr_in, fit, qualifying_routes
