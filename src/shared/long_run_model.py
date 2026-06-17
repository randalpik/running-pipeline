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

Covariates use the recovery model's encodings (temp centered at 12°C,
exp(−days/τ) race-fatigue decay). Temperature is fit fresh on long runs.
Race fatigue is a SINGLE fitted scale on a combined regressor
``fat_race_short + ratio·fat_marathon`` where the marathon/short ratio is
pinned from the profile's recovery fit (``recovery_fatigue_ratio()``).
Rationale (June 2026): the free marathon beta was unidentified — only ~4
long runs carry meaningful marathon-fatigue load, and leave-one-out swung
the beta 1.4–8.6 (23–57 on the pre-12mi-floor slice). The recovery fit
identifies the marathon-vs-short contrast on ~2,300 rows; fatigue is the
same physiological state response, so the contrast transfers while the
amplitude is regime-specific — long runs hold the fatigued state for
12–25 mi nearer threshold, and the fitted scale comes out ~2.3× recovery's
(t=2.12 against scale=1, so transplanting recovery betas wholesale
under-corrects — that variant was tested and rejected in June 2026; an
F-test could not reject a single shared beta, p=0.39). Time of day, a
strong recovery factor, is dead on long runs (t≈0.25) so it's excluded
here. ``cov_coefs`` still exposes per-category ``fat_marathon`` /
``fat_race_short`` keys (the ratio-expanded scale) so downstream consumers
(``transferable_contributions``, the TQ sidebar) see the familiar shape.

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
                                       fit_recovery_model,
                                       temp_centered_feature)

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

# 'fat_race' is the combined race-fatigue regressor (short + ratio·marathon,
# ratio pinned from the recovery fit — see recovery_fatigue_ratio()). One
# fitted scale instead of two free betas; expanded back to per-category
# keys in cov_coefs after the fit.
LR_COVARIATES = ['temp_centered', 'fat_race']
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


_FATIGUE_RATIO_CACHE: dict = {}


def recovery_fatigue_ratio():
    """Marathon/short-race fatigue contrast from the profile's recovery fit
    (β_marathon / β_race_short), computed per profile from real data so a
    watch profile gets its own contrast as history accumulates. Falls back
    to 1.0 — a flat single fatigue term, which an F-test on Max's long runs
    couldn't reject anyway — when the recovery fit is unavailable (missing
    files, too little data) or either beta comes out non-positive (an
    unidentified contrast, e.g. a profile with no marathons yet — harmless,
    since such a profile's ``fat_marathon`` is all-zero too). Cached per
    data dir; the recovery OLS is cheap but not free."""
    key = str(DATA_DIR)
    if key not in _FATIGUE_RATIO_CACHE:
        ratio = 1.0
        try:
            daily = pd.read_csv(DATA_DIR / 'daily.csv', parse_dates=['date'])
            races = pd.read_csv(DATA_DIR / 'races.csv', parse_dates=['date'])
            cs = pd.read_csv(DATA_DIR / 'bayes_cs_summary.csv',
                             parse_dates=['date'])
            betas = fit_recovery_model(daily, races, cs, verbose=False).betas
            m = float(betas.get('fat_marathon', 0.0))
            rs = float(betas.get('fat_race_short', 0.0))
            if np.isfinite(m) and np.isfinite(rs) and m > 0 and rs > 0:
                ratio = m / rs
        except Exception:
            pass
        _FATIGUE_RATIO_CACHE[key] = ratio
    return _FATIGUE_RATIO_CACHE[key]


def load_quality_dates():
    """Quality-day dates (marathon / short race) from the profile's daily.csv
    and races.csv — the recovery model's categorization, reused here for the
    long-run fatigue covariates. Filtered to the logging era so the dates
    match what the recovery fit sees."""
    daily = pd.read_csv(DATA_DIR / 'daily.csv', parse_dates=['date'])
    daily = clip_to_daily_floor(daily)
    races = pd.read_csv(DATA_DIR / 'races.csv', parse_dates=['date'])
    return quality_category_dates(daily, races)


def fit_long_run_model(lr_in, quality_dates=None, fatigue_ratio=None):
    """Fit ``raw_resid ~ physical route + covariates`` on the in-slice long
    runs via OLS, with an iterative MAD-based outlier prune at
    ``PRUNE_SIGMA`` on the corrected residuals. Fitted physical terms
    (``LR_PHYS_FITTED``, empty by default — elevation is priced upstream in
    project_long_runs, not here) and covariates (``LR_COVARIATES``:
    temperature + the ratio-pinned combined race-fatigue decay — see module
    docstring) are included only when the sample has at least ``MIN_COV_N``
    rows; each physical term additionally requires variation in the data.
    Distance carries no term — see module docstring.

    ``quality_dates`` is the per-category race-date dict from
    ``load_quality_dates()`` / ``recovery_model.quality_category_dates``;
    pass it when the caller already has it, else it's loaded from the
    profile's data dir. ``fatigue_ratio`` is the marathon/short fatigue
    contrast; defaults to ``recovery_fatigue_ratio()`` (pass explicitly in
    experiments to bypass the recovery fit).

    Returns ``(lr_in_augmented, fit, qualifying_routes)``. ``fit`` is a
    SimpleNamespace with ``intercept``, ``phys_coefs`` (dict), ``cov_coefs``
    (dict; both empty when gated off; race fatigue appears as per-category
    ``fat_marathon`` / ``fat_race_short`` keys expanded from the single
    fitted scale), ``fatigue_ratio``, ``rsquared``, ``resid_sd``,
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
    # One-sided heat hinge max(0, air_temp − 6C) — the SAME shape recovery
    # uses, with a free (steeper) slope here: heat hurts long efforts ~1.7x,
    # and the warm-side penalty is drift-stable while the old symmetric form's
    # cold arm was a fitness-season confound. Missing temp -> 0 contribution so
    # no row drops for lacking a thermometer reading.
    lr_in['temp_centered'] = temp_centered_feature(lr_in).fillna(0.0)
    # Center on the median hinge (typical-day reference, like wind and the
    # recovery model) so the temperature ADJUSTMENT is symmetric around a
    # normal day rather than one-directional off the cold floor. The slope is
    # unchanged (centering shifts only the intercept). TQ does not apply the
    # intercept, so this re-references the displayed long-run level to a typical
    # day; the internal ``corrected`` value is invariant.
    temp_ref = float(lr_in['temp_centered'].median())
    lr_in['temp_centered'] = lr_in['temp_centered'] - temp_ref
    # Combined race-fatigue regressor: one fitted scale, marathon/short
    # contrast pinned from the recovery fit (see module docstring).
    if fatigue_ratio is None:
        fatigue_ratio = recovery_fatigue_ratio()
    lr_in['fat_race'] = (lr_in['fat_race_short']
                         + fatigue_ratio * lr_in['fat_marathon'])
    # Physical terms (LR_PHYS_FITTED) and covariates enter the fit only when
    # the sample can support them; each physical term additionally requires
    # variation (watch profiles without route metadata get a constant column
    # that can't be fit). Elevation is no longer fitted here — it's priced
    # upstream in project_long_runs and already lives in raw_resid (see the
    # LR_PHYS_FITTED comment).
    if len(lr_in) >= MIN_COV_N:
        cov_cols = list(LR_COVARIATES)
        phys_cols = [c for c in LR_PHYS_FITTED if lr_in[c].nunique() > 1]
    else:
        cov_cols, phys_cols = [], []
    lr_in['route'] = lr_in['location'].where(
        lr_in['location'].isin(qualifying_routes), 'other')

    X = pd.concat([
        pd.Series(1.0, index=lr_in.index, name='Intercept'),
        lr_in[phys_cols].astype(float),
        lr_in[cov_cols].astype(float),
    ], axis=1)
    y = lr_in['raw_resid'].astype(float)

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
    cov_coefs = {c: coef_map[c] for c in cov_cols}

    lr_in['phys_contrib'] = sum((coef_map[c] * lr_in[c] for c in phys_cols),
                                pd.Series(0.0, index=lr_in.index))
    lr_in['cov_contrib'] = sum((cov_coefs[c] * lr_in[c] for c in cov_cols),
                               pd.Series(0.0, index=lr_in.index))
    # Expand the combined fatigue scale back to per-category keys so
    # downstream consumers (transferable_contributions, the TQ sidebar)
    # see the familiar shape. Contribution-identical: s·fat_race ==
    # s·fat_race_short + (ratio·s)·fat_marathon.
    if 'fat_race' in cov_coefs:
        scale = cov_coefs.pop('fat_race')
        cov_coefs['fat_marathon'] = fatigue_ratio * scale
        cov_coefs['fat_race_short'] = scale
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
        fatigue_ratio=fatigue_ratio,
        rsquared=1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan'),
        resid_sd=float(np.sqrt(ss_res / (n_kept - p))) if n_kept > p else float('nan'),
        n_kept=n_kept,
        temp_ref=temp_ref,
    )
    return lr_in, fit, qualifying_routes
