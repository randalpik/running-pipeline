"""Long-run residual regression: ``raw_resid ~ physical route + covariates``.

Fits PHYSICAL route terms (elevation gain per mile, altitude) and
temperature / recent-race-fatigue covariates on long-run pace residuals (vs
CS-implied), with iterative MAD-based outlier prune. Output is a
SimpleNamespace with ``intercept``, ``phys_coefs``, ``cov_coefs``,
``elev_ref``, ``rsquared``, ``resid_sd``, ``n_kept``.

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

Covariates use the recovery model's encodings (temp centered at 12°C,
exp(−days/τ) race-fatigue decay) but the betas are fit FRESH on long runs —
a June 2026 variant experiment showed recovery-sourced betas underperform:
marathons hit long runs ~2.3× harder than recovery runs (+39 vs +17 s/mi
peak), and time of day, a strong recovery factor, is dead on long runs
(t≈0.25) so it's excluded here.

Lifted out of ``src/plots/plot_training_quality.py`` so both the Training
plot and the Dashboard tab can import the same fit without one having to
import the other (which would drag plot-rendering imports into the
dashboard's no-Plotly path).
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.shared.paths import DATA_DIR
from src.shared.plot_window import daily_floor
from src.shared.recovery_model import (quality_category_dates,
                                       add_quality_features,
                                       TEMP_REFERENCE_C)

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

LR_COVARIATES = ['temp_centered', 'fat_marathon', 'fat_race_short']
# Physical route terms: feet of climbing per mile (centered at the in-slice
# median so the intercept reads as "typical-elevation route"; missing →
# reference), and altitude in thousands of feet (missing → sea level, the
# locations sheet only sets it where meaningfully high).
#
# The elevation slope is PINNED, not fitted: hilly routes are concentrated
# in quality-effort eras, so the empirically-fit slope comes out
# wrong-signed (−0.13, i.e. "hills make you faster") — the same era
# confounding that sank the route dummies, in continuous form. The pinned
# value is the paved-route slope from the recovery-side cross-route fit
# (route-normalization-reference.md: β = −13.7 + 0.17·elev_per_mile +
# 6.6·is_mixed, R² = 0.70), which is effort-uncontaminated because recovery
# effort is uniform and recovery routes span eras. Elevation cost is
# mechanical work, not a physiological state response, so unlike the
# fatigue betas it transfers across effort types. Altitude IS fitted: its
# coefficient is identified by the within-Boulder-era sea-level contrast
# and comes out physically sensible (≈ +3 s/mi per 1000 ft).
LR_ELEV_SLOPE = 0.17
LR_PHYS_FITTED = ['altitude_kft']


def load_quality_dates():
    """Quality-day dates (marathon / short race) from the profile's daily.csv
    and races.csv — the recovery model's categorization, reused here for the
    long-run fatigue covariates. Filtered to the logging era so the dates
    match what the recovery fit sees."""
    daily = pd.read_csv(DATA_DIR / 'daily.csv', parse_dates=['date'])
    daily = daily[daily['date'] >= daily_floor()]
    races = pd.read_csv(DATA_DIR / 'races.csv', parse_dates=['date'])
    return quality_category_dates(daily, races)


def fit_long_run_model(lr_in, quality_dates=None):
    """Fit ``raw_resid ~ physical route + covariates`` on the in-slice long
    runs via OLS, with an iterative MAD-based outlier prune at
    ``PRUNE_SIGMA`` on the corrected residuals. Physical terms (elevation
    gain per mile at the pinned ``LR_ELEV_SLOPE`` + fitted altitude) and
    covariates (``LR_COVARIATES``: temperature + marathon/short-race fatigue
    decay) are included only when the sample has at least ``MIN_COV_N``
    rows; each physical term additionally requires variation in the data.
    Distance carries no term — see module docstring.

    ``quality_dates`` is the per-category race-date dict from
    ``load_quality_dates()`` / ``recovery_model.quality_category_dates``;
    pass it when the caller already has it, else it's loaded from the
    profile's data dir.

    Returns ``(lr_in_augmented, fit, qualifying_routes)``. ``fit`` is a
    SimpleNamespace with ``intercept``, ``phys_coefs`` (dict), ``cov_coefs``
    (dict; both empty when gated off), ``elev_ref`` (the in-slice median
    ft/mi the elevation term is centered at), ``rsquared``, ``resid_sd``,
    ``n_kept``.

    The augmented frame carries ``route`` (descriptive label only), per-row
    ``phys_contrib``, ``cov_contrib``, ``model_offset``, ``corrected``, and
    an ``is_outlier`` flag.
    """
    loc_counts = lr_in['location'].value_counts()
    qualifying_routes = sorted(loc_counts[loc_counts >= MIN_ROUTE_N].index)
    if quality_dates is None:
        quality_dates = load_quality_dates()
    lr_in = add_quality_features(lr_in, quality_dates)
    # Missing temp = reference temp (contribution 0) so no row drops out of
    # the fit for lacking a thermometer reading.
    lr_in['temp_centered'] = (lr_in['temp_c'] - TEMP_REFERENCE_C).fillna(0.0)
    # Physical route terms. Elevation is centered at the in-slice median so
    # the intercept reads as a typical-elevation route; rows missing
    # elev_per_mile fall to the reference (contribution 0). Altitude is in
    # kft, missing = sea level.
    # .get(): watch profiles' daily.csv may lack the metadata columns
    # entirely, not just carry NaNs.
    _nan = pd.Series(np.nan, index=lr_in.index)
    elev = lr_in.get('elev_per_mile', _nan).astype(float)
    elev_ref = float(elev.median()) if elev.notna().any() else 0.0
    lr_in['elev_pm_c'] = (elev - elev_ref).fillna(0.0)
    lr_in['altitude_kft'] = lr_in.get('altitude', _nan).astype(float).fillna(0.0) / 1000.0
    # Gate extra terms on sample size, and each physical term additionally
    # on having any variation (watch profiles without route metadata get a
    # constant 0 column that can't be fit).
    if len(lr_in) >= MIN_COV_N:
        cov_cols = list(LR_COVARIATES)
        phys_cols = [c for c in LR_PHYS_FITTED if lr_in[c].nunique() > 1]
        elev_slope = LR_ELEV_SLOPE if lr_in['elev_pm_c'].nunique() > 1 else 0.0
    else:
        cov_cols, phys_cols, elev_slope = [], [], 0.0
    # Pinned elevation contribution comes off the target before the fit and
    # back into the model offset after — see LR_ELEV_SLOPE comment.
    lr_in['elev_contrib'] = elev_slope * lr_in['elev_pm_c']
    lr_in['route'] = lr_in['location'].where(
        lr_in['location'].isin(qualifying_routes), 'other')

    X = pd.concat([
        pd.Series(1.0, index=lr_in.index, name='Intercept'),
        lr_in[phys_cols].astype(float),
        lr_in[cov_cols].astype(float),
    ], axis=1)
    y = (lr_in['raw_resid'] - lr_in['elev_contrib']).astype(float)

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
    if elev_slope:
        phys_coefs['elev_pm_c'] = elev_slope  # pinned, not fitted
    cov_coefs = {c: coef_map[c] for c in cov_cols}

    lr_in['phys_contrib'] = sum((coef_map[c] * lr_in[c] for c in phys_cols),
                                lr_in['elev_contrib'].copy())
    lr_in['cov_contrib'] = sum((cov_coefs[c] * lr_in[c] for c in cov_cols),
                               pd.Series(0.0, index=lr_in.index))
    lr_in['model_offset'] = (intercept + lr_in['phys_contrib']
                             + lr_in['cov_contrib'])
    lr_in['corrected'] = lr_in['raw_resid'] - lr_in['model_offset']
    lr_in['is_outlier'] = lr_in.index.isin(pruned)

    keep_mask = ~lr_in.index.isin(pruned)
    ya = y.values[keep_mask]
    pa = (X.values @ coef)[keep_mask]
    ss_res = float(np.sum((ya - pa) ** 2))
    # R² against the RAW residual variance so the pinned elevation term's
    # explanatory share is included, not just the fitted columns'.
    raw_kept = lr_in['raw_resid'].astype(float).to_numpy()[keep_mask]
    ss_tot = float(np.sum((raw_kept - raw_kept.mean()) ** 2))
    n_kept = int(keep_mask.sum())
    p = X.shape[1]
    fit = SimpleNamespace(
        intercept=intercept,
        phys_coefs=phys_coefs,
        cov_coefs=cov_coefs,
        elev_ref=elev_ref,
        rsquared=1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan'),
        resid_sd=float(np.sqrt(ss_res / (n_kept - p))) if n_kept > p else float('nan'),
        n_kept=n_kept,
    )
    return lr_in, fit, qualifying_routes
