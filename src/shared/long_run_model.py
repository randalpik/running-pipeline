"""Long-run residual regression: ``raw_resid ~ bin + route``.

Fits per-route + per-bin offsets on long-run pace residuals (vs CS-implied),
with iterative MAD-based outlier prune. Output is a SimpleNamespace with
``intercept``, ``bin_coefs``, ``route_coefs``, ``rsquared``, ``resid_sd``,
``n_kept``.

Lifted out of ``src/plots/plot_training_quality.py`` so both the Training
plot and the Dashboard tab can import the same fit without one having to
import the other (which would drag plot-rendering imports into the
dashboard's no-Plotly path).
"""
from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.shared.workouts import LR_INTERNAL_BIN

# Min long runs per location to qualify as a "route with a beta".
# A route below this gets folded into the 'other' reference category.
MIN_ROUTE_N = 5
# Iterative MAD-based outlier threshold used during the OLS fit.
PRUNE_SIGMA = 3.0


def fit_long_run_model(lr_in):
    """Fit ``raw_resid ~ bin + route`` (Treatment-encoded with 'lr_hi' and
    'other' as reference levels) on the in-slice long runs via OLS, with an
    iterative MAD-based outlier prune at ``PRUNE_SIGMA`` on the V1-corrected
    residuals.

    Returns ``(lr_in_augmented, fit, qualifying_routes)``. ``fit`` is a
    SimpleNamespace with ``intercept``, ``bin_coefs`` (dict, ref level
    absent), ``route_coefs`` (dict, 'other' absent), ``rsquared``,
    ``resid_sd``, ``n_kept``.

    The augmented frame carries ``bin``, ``route``, per-row ``bin_coef``,
    ``route_coef``, ``model_offset``, ``corrected``, an ``is_outlier`` flag,
    and ``category`` set to the bin label for downstream rendering.
    """
    loc_counts = lr_in['location'].value_counts()
    qualifying_routes = sorted(loc_counts[loc_counts >= MIN_ROUTE_N].index)
    lr_in = lr_in.copy()
    lr_in['route'] = lr_in['location'].where(
        lr_in['location'].isin(qualifying_routes), 'other')
    # Max splits long runs into two distance bins (lr_lo/lr_hi); other profiles
    # keep a single bin until there's enough history to support a split.
    if os.environ.get('RP_PROFILE', 'max') == 'max':
        lr_in['bin'] = np.where(lr_in['miles'] < LR_INTERNAL_BIN, 'lr_lo', 'lr_hi')
    else:
        lr_in['bin'] = 'lr_hi'

    # Treatment encoding with explicit reference categories. 'lr_hi' is the
    # bin reference (alphabetically first); 'other' is the route reference
    # so per-route betas read as offsets relative to the unqualified pool.
    bin_cat = pd.Categorical(lr_in['bin'], categories=['lr_hi', 'lr_lo'])
    bin_dum = pd.get_dummies(bin_cat, prefix='bin').drop(columns=['bin_lr_hi'])
    route_cat = pd.Categorical(lr_in['route'],
                               categories=['other'] + qualifying_routes)
    route_dum = pd.get_dummies(route_cat, prefix='route').drop(columns=['route_other'])
    bin_dum.index = lr_in.index
    route_dum.index = lr_in.index

    X = pd.concat([
        pd.Series(1.0, index=lr_in.index, name='Intercept'),
        bin_dum.astype(float),
        route_dum.astype(float),
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
    bin_coefs = {str(c).replace('bin_', ''): coef_map[str(c)] for c in bin_dum.columns}
    route_coefs = {str(c).replace('route_', ''): coef_map[str(c)] for c in route_dum.columns}

    lr_in['bin_coef'] = lr_in['bin'].map(lambda b: bin_coefs.get(b, 0.0))
    lr_in['route_coef'] = lr_in['route'].map(lambda r: route_coefs.get(r, 0.0))
    lr_in['model_offset'] = intercept + lr_in['bin_coef'] + lr_in['route_coef']
    lr_in['corrected'] = lr_in['raw_resid'] - lr_in['model_offset']
    lr_in['category'] = lr_in['bin']
    lr_in['is_outlier'] = lr_in.index.isin(pruned)

    keep_mask = ~lr_in.index.isin(pruned)
    ya = y.values[keep_mask]
    pa = (X.values @ coef)[keep_mask]
    ss_res = float(np.sum((ya - pa) ** 2))
    ss_tot = float(np.sum((ya - ya.mean()) ** 2))
    n_kept = int(keep_mask.sum())
    p = X.shape[1]
    fit = SimpleNamespace(
        intercept=intercept,
        bin_coefs=bin_coefs,
        route_coefs=route_coefs,
        rsquared=1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan'),
        resid_sd=float(np.sqrt(ss_res / (n_kept - p))) if n_kept > p else float('nan'),
        n_kept=n_kept,
    )
    return lr_in, fit, qualifying_routes
