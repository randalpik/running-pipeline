"""Long-run residual model: the pinned temperature term plus a display-dead
level fit.

Supplies the long-run temperature contribution (``temp_contribution``, pinned
from the pooled recovery+long-run fit) and fits the level (plus any FITTED
physical terms in ``LR_PHYS_FITTED``, empty by default) on long-run pace
residuals (vs CS-implied), with iterative MAD-based outlier prune. Elevation is
not fitted here — it's priced per-run upstream in ``workouts.project_long_runs``
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
handful of long runs. ``cov_coefs`` carries the pinned ``temp_centered`` key
for the TQ sidebar. Time of day, a strong recovery factor, is dead on long runs
(t≈0.25) and is not in the pool's transferable set, so it never enters here.

RACE FATIGUE REMOVED from long runs (Max, Sept 2026). The pooled fatigue betas
still serve recovery, but long runs no longer apply them. The credit keyed on
race OCCURRENCE, not on fatigue actually incurred, so a tactical or
low-cost race (Wednesday All-Comers 3000s, a downhill mile TT) credited the
next long run ~17 s of 5K time — and the demonstrated-capability frontier is a
max envelope, so a one-directional over-credit moves it directly while an
under-credit is hidden. Races and workouts on the same frontier get no
fatigue credit either. The pooled slope is also recovery-dominated (by-feel
effort, where fatigue lowers the chosen pace), and the long-run response
could not be identified on its own (June 2026: ~4 runs with marathon load,
LOO swing 1.4–8.6; the long-run-only fit read ~2.3× recovery's amplitude but
was era-confounded). At removal it decided 5 of the 22 long runs that set
the frontier somewhere.

SCALE (Sept 2026): the pooled betas are s/mi of RUN pace (the pool regresses
run pace − CS pace), but ``raw_resid`` is s/mi of 5K-EQUIVALENT pace. The
per-run contribution (``temp_contribution``) is therefore removed from the
run's time inside ``project_long_runs`` and re-scored through the same WA
conversion, which yields ``temp_adj`` on the 5K-equivalent scale — that is
what ``cov_contrib`` carries here. 1 s/mi of long-run pace is ~0.77 s/mi of
5K-equivalent pace for Max, so subtracting run-pace seconds directly (as the
code did from the June 2026 pinning until Sept 2026) over-applied the terms
by ~30 %. The temperature credit also feeds the pause-erosion effort gate
(Sept 2026), like the physical route credits — the gate polices the pace the
run actually claims.

ROLE CHANGED (June 2026): the INTERCEPT is display-dead. Long runs enter
TQ at their race-equivalent raw residual (``project_long_runs``: WA 5K
conversion of the flat-/altitude-corrected, pause-eroded run) minus this
model's PHYSICAL + COVARIATE contributions only (``phys_contrib +
cov_contrib`` — always applied). The intercept/level is fit (it
robustifies the prune) but NEVER subtracted from any 5K-equivalent
position — a constant subtracted from every long run claims a long run
out-predicts a race at the same distance/pace, which is physically
indefensible (Max's no-class-constants contract). Long-run TQ pruning
rides the shared track-relative prune; this fit's internal MAD prune only
robustifies its level. ``route`` labels + ``is_outlier`` are also read by
dashboard.py's ``_long_run_residual``, which is currently uncalled.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.shared.recovery_model import (physical_route_betas,
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

# Transferable-state covariates applied to long runs: the temperature heat
# slope only, PINNED from the pooled fit (physical_route_betas), not fit here.
# Race fatigue was removed in Sept 2026 — see the module docstring.
LR_COVARIATES = ['temp_centered']
# Physical route terms fitted here. ELEVATION IS NO LONGER ONE OF THEM
# (June 2026, watch-stream-enrichment §A): the route-constant median-centered
# slope (0.17·elev_per_mile) that used to live on this residual scale was
# replaced by the per-run, effort-aware physical grade engine, applied
# upstream as a TIME correction in workouts.project_long_runs — so raw_resid
# already carries the flat-equivalent grade correction by the time it reaches
# this fit. Pricing grade on the run's own pace (before the WA 5K
# conversion), from each run's measured hill segments through the two-channel
# model (climb cost + steepness-dependent descent benefit), is strictly better
# than a single pinned slope on a balanced-route constant centered at the era
# median.
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


def state_betas(n_in_scope):
    """The pinned covariate betas (s/mi of RUN pace) keyed by
    ``LR_COVARIATES``, from the pooled recovery+long-run fit — or ``{}`` below
    ``MIN_COV_N`` in-scope long runs (the sparse-profile "no state correction"
    behavior; the pool returns ~0 for such profiles anyway)."""
    if n_in_scope < MIN_COV_N:
        return {}
    pooled = physical_route_betas()
    return {c: float(pooled.get(c, 0.0)) for c in LR_COVARIATES}


def _temp_ref(hinge, in_scope):
    """Median heat hinge over the in-scope rows that HAVE a reading — the
    typical-day reference the temperature adjustment is centered on."""
    m = hinge[np.asarray(in_scope, bool)].median()
    return float(m) if pd.notna(m) else 0.0


def temp_contribution(lr, in_scope):
    """Per-row temperature pace contribution for long runs, in s/mi of RUN
    pace (positive = the heat slowed the run), a Series aligned to
    ``lr.index``. Zeros below ``MIN_COV_N`` in-scope rows.

    The one-sided heat hinge max(0, air_temp − 6 °C) times the pinned slope,
    centered on the in-scope median hinge (``_temp_ref``) so the adjustment is
    symmetric around a normal day rather than one-directional off the cold
    floor. A MISSING reading contributes exactly 0 (neutral, like an unlogged
    wind reading) — filling the hinge with 0 before centering, as the code did
    until Sept 2026, charged every unmeasured run the full sub-6 °C penalty.

    ``in_scope`` is a boolean mask over ``lr`` (the rows the TQ model admits);
    it sets both the ``MIN_COV_N`` gate and the temperature reference, so
    out-of-slice rows get contributions on the same footing as in-scope ones.
    Consumers on the 5K-equivalent scale must convert (see module docstring,
    SCALE)."""
    in_scope = np.asarray(in_scope, bool)
    coefs = state_betas(int(in_scope.sum()))
    if not coefs:
        return pd.Series(0.0, index=lr.index)
    hinge = temp_centered_feature(lr)
    return (coefs['temp_centered'] * (hinge - _temp_ref(hinge, in_scope))
            ).fillna(0.0)


def fit_long_run_model(lr_in):
    """Fit the (display-dead) long-run LEVEL on the in-slice long runs (the
    in-scope rows of ``workouts.project_long_runs``), with the temperature
    offset (PINNED from the pooled recovery+long-run estimator; see module
    docstring) taken from the projection's ``temp_adj`` column — already on
    the 5K-equivalent scale.
    Only the intercept (+ any ``LR_PHYS_FITTED``, empty by default —
    elevation is priced upstream in project_long_runs, not here) is fit, via the
    same iterative MAD-based outlier prune at ``PRUNE_SIGMA``. The temperature
    offset is nonzero only when the sample has at least ``MIN_COV_N`` rows (the
    sparse-profile "no state correction" behavior). Distance carries no term —
    see module docstring.

    Returns ``(lr_in_augmented, fit, qualifying_routes)``. ``fit`` is a
    SimpleNamespace with ``intercept``, ``phys_coefs`` (dict), ``cov_coefs``
    (dict of the pinned run-pace ``temp_centered`` beta for display, empty
    below ``MIN_COV_N``),
    ``rsquared``, ``resid_sd``, ``n_kept``, ``temp_ref``.

    The augmented frame carries ``route`` (descriptive label only), per-row
    ``phys_contrib``, ``cov_contrib``, ``model_offset``, ``corrected``, and
    an ``is_outlier`` flag.
    """
    loc_counts = lr_in['location'].value_counts()
    qualifying_routes = sorted(loc_counts[loc_counts >= MIN_ROUTE_N].index)
    lr_in = lr_in.copy()
    temp_ref = _temp_ref(temp_centered_feature(lr_in),
                         np.ones(len(lr_in), bool))
    lr_in['route'] = lr_in['location'].where(
        lr_in['location'].isin(qualifying_routes), 'other')

    # Temperature is PINNED from the pooled recovery+long-run fit — identified
    # on the full corpus, not re-fit on the handful of in-slice long runs (see
    # module docstring). project_long_runs applies it and converts it to the
    # 5K-equivalent scale (`temp_adj`); cov_coefs here is the run-pace beta, for
    # display only. LR_PHYS_FITTED (footing/altitude) stays an empty extension
    # point fit here; each such term additionally requires variation (a
    # constant column can't be fit).
    cov_coefs = state_betas(len(lr_in))
    phys_cols = ([c for c in LR_PHYS_FITTED if lr_in[c].nunique() > 1]
                 if len(lr_in) >= MIN_COV_N else [])

    lr_in['cov_contrib'] = lr_in['temp_adj'].astype(float)

    # The only FITTED term is the (display-dead) level — intercept + any
    # LR_PHYS_FITTED — on raw_resid with the pinned temperature offset removed
    # (the way recovery subtracts its pinned offset). The iterative MAD prune
    # still runs: it produces is_outlier (read only by dashboard.py's
    # uncalled _long_run_residual).
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
