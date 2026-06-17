"""Hill-continuous correction: pinned Minetti net cost + one fitted trail term.

June 2026, second iteration. The first replacement for per-loop offsets
fit a free gain slope (`raw_resid ~ intercept + ft_per_mi + is_trail`);
it treated all climbing as pure cost, ignoring that descents give most
of it back at moderate grades, so the slope absorbed effort gap and
over-corrected steep/fast days (a 5:09-pace lc session displayed as a
4:28/mi 5K-equivalent — sub-14 5K, never run). Rejected as not grounded.

Now the gain correction is PINNED from mechanism (the long-run model's
pinned-elev precedent): Minetti et al. 2002 energy cost of running at
gradient i, applied as a multiplicative NET factor for a loop that
climbs and descends symmetrically — ``(C(i)+C(−i)) / 2·C(0)`` with
``i = climb/(loop/2)`` (half the loop ascends, half descends; the hills
sheet records climb and distance, not climb fraction — refine there if
a loop is meaningfully asymmetric). Verified against intuition: net
cost ~+14 s/mi for lc (5.7% climb grade, paved), +26 rc, +48 pwr1;
2021-12-19 (4 mi @ 5:09 on lc) reads a 4:55/mi 5K-equiv, mid-15 shape.

The only FITTED term is one binary trail coefficient — the
trail-vs-paved DIFFERENCE of Minetti-corrected residuals (rocky/gravel
descents don't give the climb back the way pavement does). Caveat: trail
is partly era-confounded (pwr1 is all 2016-18), so the term absorbs some
era effort policy; documented, accepted by Max.

There is NO intercept and hills are NOT centered: the hill-class effort
gap (hill workouts are sub-max vs racing) stays visible in the
residuals, exactly like tempo-era effort policy does for flat workouts.

Prune is iterative, one-sided (slow only), ``PRUNE_SIGMA``-MAD on the
trail-corrected residuals — drops only the most egregious easy hill
days; a fast hill day is real fitness signal and always survives.

The trail coefficient is persisted by plot_training_quality to
``hill_model.csv`` so the Workouts plot shares the same correction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.shared.long_run_model import PRUNE_SIGMA
from src.shared.units import FT_PER_M  # re-exported: existing importers read it here


def minetti_cost(i):
    """Energy cost of running at gradient i (Minetti 2002), J/kg/m."""
    i = np.asarray(i, dtype=float)
    return (155.4 * i**5 - 30.4 * i**4 - 43.3 * i**3
            + 46.3 * i**2 + 19.5 * i + 3.6)


def minetti_net_factor(climb_ft, loop_m):
    """Net energy factor vs flat for one loop climbing ``climb_ft`` and
    descending it again, symmetric half-up/half-down approximation."""
    climb_m = np.asarray(climb_ft, dtype=float) * FT_PER_M
    i = climb_m / (np.asarray(loop_m, dtype=float) / 2.0)
    return (minetti_cost(i) + minetti_cost(-i)) / (2.0 * minetti_cost(0.0))


def fit_hill_model(hills: pd.DataFrame):
    """Fit the single trail term on Minetti-corrected residuals.

    Expects ``minetti_resid`` (residual after the pinned Minetti
    correction, computed in project_hill_continuous) and ``is_trail``.
    Returns ``(hills, fit)`` where ``hills`` gains ``corrected``
    (= minetti_resid − trail_coef·is_trail; effort level stays in) and
    ``is_outlier``; ``fit`` has ``trail_coef``, ``resid_sd``, ``n_kept``.
    """
    hills = hills.copy()
    if hills.empty:
        hills['corrected'] = pd.Series(dtype=float)
        hills['is_outlier'] = pd.Series(dtype=bool)
        return hills, None

    y_full = hills['minetti_resid'].to_numpy(dtype=float)
    trail = hills['is_trail'].to_numpy(dtype=float)

    pruned = set()
    trail_coef = 0.0
    for _ in range(15):
        keep = ~hills.index.isin(pruned)
        # OLS with intercept isolates the trail-vs-paved DIFFERENCE; the
        # intercept (paved-class effort level) is estimated but never
        # subtracted.
        X = np.column_stack([np.ones(int(keep.sum())), trail[keep]])
        coef, *_ = np.linalg.lstsq(X, y_full[keep], rcond=None)
        trail_coef = float(coef[1])
        resid = y_full - trail_coef * trail
        active = resid[keep]
        center = float(np.median(active))
        sd_robust = 1.4826 * float(np.median(np.abs(active - center)))
        new_pruned = set(hills.index[resid - center
                                     > PRUNE_SIGMA * sd_robust])
        if new_pruned == pruned:
            break
        pruned = new_pruned

    hills['corrected'] = hills['minetti_resid'] - trail_coef * hills['is_trail']
    hills['is_outlier'] = hills.index.isin(pruned)

    keep = ~hills['is_outlier']
    fit = {
        'trail_coef': trail_coef,
        'resid_sd': float(hills.loc[keep, 'corrected'].std()),
        'n_kept': int(keep.sum()),
    }
    return hills, fit
