"""World Athletics (IAAF) scoring-based race-equivalence conversion.

Replaces the CS-internal beta_long fade for cross-distance equivalence. Each
performance is scored on the WA 2025 men's tables (points = a*T^2 + b*T + c per
event), then mapped to the equivalent 5K time at the same score. This homogenises
races, workouts (via connected-fatigue D_eff), and long runs to a common 5K-equiv
that grounds the CS fit and the demonstrated-capability frontier.

Coefficients: jchen1/iaaf-scoring-tables coefficients-2025.json, men's outdoor,
cross-checked to reproduce the owner's known scores (5K/10K 828, HM ~893,
marathon ~881 on the recalibrated 2025 marathon table). Non-standard distances
interpolate the iso-points TIME in log-distance between bracketing events.

Long runs additionally carry a PAUSE PENALTY, but that lives in
src/shared/durability.py (a durability + W'-balance model of the marginal effect
of a run's own stops); this module is purely the cross-distance WA conversion.
Races and workouts get no pause penalty.
"""
from __future__ import annotations
import math

# event_distance_m -> (a, b, c), points = a*T^2 + b*T + c, T seconds (2025 men's)
COEF = {
    400.0:    (1.0210130425695638,  -161.3092238081408,  6371.289298935095),
    800.0:    (0.1980049254166545,   -72.07136038821409, 6558.28160300618),
    1500.0:   (0.04065992529984008,  -31.307736299477256, 6026.662254345021),
    1609.34:  (0.035099677603458446, -29.132456259137143, 6044.924547011615),
    3000.0:   (0.008150049932713843, -13.691983542337312, 5750.59246378555),
    5000.0:   (0.002777997945427213, -8.000608112196687,  5760.418712362531),
    10000.0:  (5.243835511893474e-4, -3.302659028227424,  5200.274036400777),  # road 10k
    15000.0:  (2.1619813224982992e-4, -2.104692144502337, 5122.29998515785),
    20000.0:  (1.0849945387967144e-4, -1.4544357631051241, 4874.178186156554),
    21097.5:  (9.469710951061014e-5, -1.3521892901331114, 4827.020676429092),
    25000.0:  (6.380128595110182e-5, -1.102413160310192,  4762.120081442874),
    30000.0:  (4.1899103628650555e-5, -0.8882236041678482, 4707.383289585705),
    42195.0:  (2.0101186255287035e-5, -0.6150659606552438, 4705.042285787989),
}
_DISTS = sorted(COEF)
ANCHOR_M = 5000.0


def _time_at(ev, P):
    """Time (s) at event ev scoring P points — the fast (lower) root."""
    a, b, c = COEF[ev]
    disc = b * b - 4 * a * (c - P)
    return (-b - math.sqrt(max(disc, 0.0))) / (2 * a)


def wa_points(dist_m, time_s):
    """WA points for a performance (dist_m, time_s). Exact event when dist
    matches a tabled distance (<1%); otherwise interpolate the iso-points TIME
    in log-distance between the bracketing events and solve for the score."""
    if time_s <= 0 or dist_m <= 0:
        return float('nan')
    for ev in _DISTS:
        if abs(dist_m - ev) / ev < 0.01:
            a, b, c = COEF[ev]
            return a * time_s * time_s + b * time_s + c
    if dist_m <= _DISTS[0]:
        a, b, c = COEF[_DISTS[0]]; return a * time_s * time_s + b * time_s + c
    if dist_m >= _DISTS[-1]:
        a, b, c = COEF[_DISTS[-1]]; return a * time_s * time_s + b * time_s + c
    lo = max(e for e in _DISTS if e < dist_m)
    hi = min(e for e in _DISTS if e > dist_m)
    w = (math.log(dist_m) - math.log(lo)) / (math.log(hi) - math.log(lo))

    def iso_time(P):  # interpolated equivalent time at dist_m for score P
        return math.exp((1 - w) * math.log(_time_at(lo, P))
                        + w * math.log(_time_at(hi, P)))

    a, b = 50.0, 1400.0  # bisection on score (iso_time decreasing in P)
    for _ in range(60):
        m = (a + b) / 2
        a, b = (m, b) if iso_time(m) > time_s else (a, m)
    return (a + b) / 2


def _time_at_dist(dist_m, P):
    """Equivalent time (s) at any distance for score P (exact event or
    log-distance interpolation of the iso-points time between brackets)."""
    if P != P:
        return float('nan')
    for ev in _DISTS:
        if abs(dist_m - ev) / ev < 0.01:
            return _time_at(ev, P)
    if dist_m <= _DISTS[0]:
        return _time_at(_DISTS[0], P)
    if dist_m >= _DISTS[-1]:
        return _time_at(_DISTS[-1], P)
    lo = max(e for e in _DISTS if e < dist_m)
    hi = min(e for e in _DISTS if e > dist_m)
    w = (math.log(dist_m) - math.log(lo)) / (math.log(hi) - math.log(lo))
    return math.exp((1 - w) * math.log(_time_at(lo, P)) + w * math.log(_time_at(hi, P)))


def wa_5k_equiv_time(dist_m, time_s):
    """Equivalent 5K TIME (s) for a performance, via matching WA score (the
    DOWN-conversion: any aerobic distance -> 5K)."""
    return _time_at_dist(ANCHOR_M, wa_points(dist_m, time_s))


def wa_equiv_time_at(dist_m, time_5k_s):
    """Equivalent TIME (s) at dist_m for a 5K performance (the UP-conversion:
    5K -> any aerobic distance) — inverse of wa_5k_equiv_time. Used to project
    the 5K-equivalent CS frontier up to HM/marathon anchors for predictions and
    the by-distance race plot, replacing the retired beta_long fade."""
    return _time_at_dist(dist_m, wa_points(ANCHOR_M, time_5k_s))


