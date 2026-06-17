"""Physical elevation cost for route normalization (June 2026, watch
enrichment — docs/route-normalization-reference.md, the elevation engine).

Replaces per-route dummy betas with a physical, era-free route cost:
``cost = c_up·gain − refund·c_up·loss`` (s/mi), with terrain-specific
parameters. Climbing always costs; the descent refunds a *fraction* of that
cost, and the fraction depends on terrain and effort:

* **Paved** — smooth, so descents refund nearly fully at easy effort (you bank
  them) and less as you approach race pace (you're near v_max and can't speed
  up to cash the descent). Refund falls from ~1.0+ (recovery) toward ~0.85
  (race) — see REFUND_PAVED_BY_EFFORT.
* **Mixed / trail** — rough footing caps the descent at *all* efforts, so the
  refund is low (~0.34) and roughly effort-flat; climbs also cost more.

These constants are the empirical values from the per-mile, terrain-bucketed,
distance-appropriate-effort analysis (validity-gated; paved refund ~136%→85%
across effort, mixed ~30–37%). They are deliberately exposed as named knobs to
tune against the rendered plots. The flat-footing terrain baseline (mixed/trail
slower even on the flat) is NOT here — it's fitted in the recovery model on top
of this pinned cost.
"""
import numpy as np

# Cost per ft/mi climbed (s/mi). Rough terrain costs more to climb.
CLIMB_COST = {'paved': 0.19, 'mixed': 0.26, 'trail': 0.26}

# Refund: fraction of the climb cost returned on the descent. Recovery-effort
# values (the recovery route model runs here). Paved ~full (descents bank the
# climbs at easy effort); mixed/trail low (rough descents don't refund).
REFUND_RECOVERY = {'paved': 1.00, 'mixed': 0.34, 'trail': 0.34}

# Paved refund vs effort (frontier pace at the run's distance / run pace;
# 1.0 = racing that distance). Linear; used by effort-aware consumers
# (long-run / race / CS). Mixed/trail stay ~REFUND_RECOVERY at all efforts.
REFUND_PAVED_BY_EFFORT = ((0.85, 1.00), (0.98, 0.85))


def paved_refund(effort):
    """Paved descent-refund fraction at a given effort (interpolated/clamped)."""
    xs = [REFUND_PAVED_BY_EFFORT[0][0], REFUND_PAVED_BY_EFFORT[1][0]]
    ys = [REFUND_PAVED_BY_EFFORT[0][1], REFUND_PAVED_BY_EFFORT[1][1]]
    return float(np.interp(effort, xs, ys))


def elevation_cost(gain_pm, loss_pm, terrain, refund=None):
    """Elevation pace cost (s/mi) for per-mile gain/loss (ft/mi) on ``terrain``
    (paved/mixed/trail; unknown → paved). ``refund`` overrides the terrain's
    recovery-effort refund (e.g. an effort-aware paved value). Vectorized over
    array inputs; ``terrain`` may be a scalar or array."""
    gain_pm = np.asarray(gain_pm, float)
    loss_pm = np.asarray(loss_pm, float)
    terr = np.asarray(terrain, dtype=object)
    cu = np.vectorize(lambda t: CLIMB_COST.get(t, CLIMB_COST['paved']))(terr)
    if refund is None:
        rf = np.vectorize(
            lambda t: REFUND_RECOVERY.get(t, REFUND_RECOVERY['paved']))(terr)
    else:
        rf = np.asarray(refund, float)
    return cu * gain_pm - rf * cu * loss_pm
