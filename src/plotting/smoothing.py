"""Smoothing helpers used by the training-quality and recovery plots.

Two distinct algorithms live here:

- :func:`adaptive_gauss_smoother` — bandwidth grows continuously to keep
  effective sample size at or above ``target_ess``. Used by the
  training-quality track to handle uneven sampling without visible jogs.

- :func:`gaussian_rolling_trend` — fixed-bandwidth Gaussian on a regular
  daily grid. Mirrors the JS rolling trend in the recovery plot so the
  initial paint matches what the JS recompute would produce.

:data:`GAP_BREAK_DAYS` is the canonical 90-day gap-break threshold (the
2020-2021 labrum gap is the motivating case for the training-quality
track).
"""
from __future__ import annotations

import numpy as np


GAP_BREAK_DAYS = 90


def adaptive_gauss_smoother(ds, res, grid_days, *,
                             target_ess: float = 12,
                             base_bw: float = 30,
                             max_bw: float = 400):
    """Adaptive-bandwidth Gaussian smoother.

    Bandwidth at each grid point is the smallest value (found by bisection)
    such that effective sample size ESS = (Σw)² / Σw² ≥ ``target_ess``.
    Bisection makes the bandwidth vary continuously across the grid, so
    the resulting curve has no visible discontinuities from discrete
    bandwidth steps.

    ``ds``: source-point days. ``res``: source-point values.
    ``grid_days``: days at which to evaluate the smoother.
    Returns an array the same length as ``grid_days``; entries where
    even ``max_bw`` couldn't reach ``target_ess`` are NaN.
    """
    out = np.full(len(grid_days), np.nan)

    def weights_and_ess(t, bw):
        w = np.exp(-0.5 * ((ds - t) / bw) ** 2)
        s = w.sum()
        ss = (w * w).sum()
        return w, (s * s / ss) if ss > 0 else 0.0

    for i, t in enumerate(grid_days):
        w, ess = weights_and_ess(t, base_bw)
        if ess >= target_ess:
            out[i] = (w * res).sum() / w.sum()
            continue

        hi = base_bw
        while hi < max_bw:
            hi = min(hi * 2, max_bw)
            w, ess = weights_and_ess(t, hi)
            if ess >= target_ess:
                break
        if ess < target_ess:
            continue

        lo = base_bw
        for _ in range(40):
            if hi - lo < 0.5:
                break
            mid = 0.5 * (lo + hi)
            w, ess = weights_and_ess(t, mid)
            if ess >= target_ess:
                hi = mid
            else:
                lo = mid
        w, _ = weights_and_ess(t, hi)
        out[i] = (w * res).sum() / w.sum()
    return out


def gaussian_rolling_trend(date_ms, ys, *, sigma_days, step_days: int = 1,
                            trunc_sigma: int = 4, min_count: int = 5):
    """Daily-grid Gaussian smoother — server-side mirror of the JS
    rollingTrend used by the recovery plot.

    Pre-computing the initial trend in Python lets the recovery plot
    render its trendline on first paint instead of waiting for the JS
    recompute to populate originally-empty trace x/y arrays.

    Returns ``(trend_dates_ms, trend_ys)``, both numpy arrays. Output is
    empty when there are fewer than ``min_count`` valid points in any
    given window.
    """
    if len(date_ms) == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=float)
    date_ms = np.asarray(date_ms, dtype=np.float64)
    ys_arr = np.asarray(ys, dtype=np.float64)
    valid = ~np.isnan(ys_arr)
    if not valid.any():
        return np.array([], dtype=np.int64), np.array([], dtype=float)

    sigma_ms  = sigma_days * 86_400_000
    trunc_ms  = trunc_sigma * sigma_ms
    step_ms   = step_days * 86_400_000
    two_sigsq = 2.0 * sigma_ms * sigma_ms

    t0, t1 = float(date_ms[0]), float(date_ms[-1])
    grid = np.arange(t0, t1 + 1, step_ms)

    trend_x: list[float] = []
    trend_y: list[float] = []
    lo = hi = 0
    n_src = len(date_ms)
    for t in grid:
        lo_t = t - trunc_ms
        hi_t = t + trunc_ms
        while lo < n_src and date_ms[lo] < lo_t:
            lo += 1
        while hi < n_src and date_ms[hi] <= hi_t:
            hi += 1
        if hi - lo < min_count:
            continue
        v_slice = valid[lo:hi]
        if int(v_slice.sum()) < min_count:
            continue
        dt = date_ms[lo:hi] - t
        w = np.exp(-(dt * dt) / two_sigsq) * v_slice
        sum_w = float(w.sum())
        if sum_w == 0.0:
            continue
        ys_slice = np.where(v_slice, ys_arr[lo:hi], 0.0)
        trend_x.append(t)
        trend_y.append(float((w * ys_slice).sum() / sum_w))
    return np.array(trend_x, dtype=np.int64), np.array(trend_y, dtype=float)
