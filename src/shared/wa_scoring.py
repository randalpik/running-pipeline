"""World Athletics (IAAF) scoring-based race-equivalence conversion.

Replaces the CS-internal beta_long fade for cross-distance equivalence. Each
performance is scored on the WA 2025 men's tables (points = a*T^2 + b*T + c per
event), then mapped to the equivalent 5K time at the same score. This homogenises
races, workouts (via connected-fatigue D_eff), and long runs to a common 5K-equiv
that grounds the CS fit and the demonstrated-capability frontier.

Aerobic curve (>5K): the per-event WA coefficients for the intermediate road
distances (15/20/25/30k) are a track/road mix and do NOT lie on a smooth curve —
plotted, the iso-fitness curve sawtooths through them and is discontinuous at the
HM tab, so eroding a long run toward ~HM could spuriously speed up its conversion.
We therefore anchor ONLY on the real race distances we actually run — 5K, 10K, HM,
marathon — and interpolate the iso-points TIME between them with a MONOTONE cubic
(PCHIP / Fritsch–Carlson) in log-distance/log-time: C1-smooth, passes exactly
through the anchors, and provably can't overshoot (so the conversion is monotone
in distance — no spikes). Distances above the marathon extrapolate linearly in
log-log (stays monotone). Sub-5K stays on the track tabs (CP3 owns ≤5K in the
hybrid); only the 5K→marathon region changed. Since every race >5K is at an
anchor, no race conversion moves — only long runs / workouts at in-between
distances.

Coefficients: jchen1/iaaf-scoring-tables coefficients-2025.json, men's outdoor,
cross-checked to reproduce the owner's known scores (5K/10K 828, HM ~893,
marathon ~881 on the recalibrated 2025 marathon table).

Long runs additionally carry a pause-uncertainty erosion, but that lives in
src/shared/durability.py; this module is purely the cross-distance WA conversion.
Races and workouts get no pause penalty.
"""
from __future__ import annotations
import math

# event_distance_m -> (a, b, c), points = a*T^2 + b*T + c, T seconds (2025 men's).
# The intermediate road tabs (15/20/25/30k) were REMOVED: they don't lie on a
# smooth iso-fitness curve (track/road mix) and caused a non-monotone HM-tab
# spike. The aerobic curve now interpolates only between the real anchors below.
COEF = {
    400.0:    (1.0210130425695638,  -161.3092238081408,  6371.289298935095),
    800.0:    (0.1980049254166545,   -72.07136038821409, 6558.28160300618),
    1500.0:   (0.04065992529984008,  -31.307736299477256, 6026.662254345021),
    1609.34:  (0.035099677603458446, -29.132456259137143, 6044.924547011615),
    3000.0:   (0.008150049932713843, -13.691983542337312, 5750.59246378555),
    5000.0:   (0.002777997945427213, -8.000608112196687,  5760.418712362531),
    10000.0:  (5.243835511893474e-4, -3.302659028227424,  5200.274036400777),  # road 10k
    21097.5:  (9.469710951061014e-5, -1.3521892901331114, 4827.020676429092),  # HM
    42195.0:  (2.0101186255287035e-5, -0.6150659606552438, 4705.042285787989),  # marathon
}
_DISTS = sorted(COEF)
ANCHOR_M = 5000.0

# Aerobic anchors for the smooth monotone iso-time curve (real race distances).
_AERO = (5000.0, 10000.0, 21097.5, 42195.0)
_LOG_AERO = [math.log(d) for d in _AERO]


def _time_at(ev, P):
    """Time (s) at event ev scoring P points — the fast (lower) root."""
    a, b, c = COEF[ev]
    disc = b * b - 4 * a * (c - P)
    return (-b - math.sqrt(max(disc, 0.0))) / (2 * a)


def _mono_tangents(xs, ys):
    """Fritsch–Carlson monotone cubic tangents through (xs, ys) — guarantees a
    shape-preserving (no-overshoot) interpolant."""
    n = len(xs)
    h = [xs[i + 1] - xs[i] for i in range(n - 1)]
    d = [(ys[i + 1] - ys[i]) / h[i] for i in range(n - 1)]
    m = [0.0] * n
    for i in range(1, n - 1):
        if d[i - 1] * d[i] <= 0:
            m[i] = 0.0
        else:
            w1, w2 = 2 * h[i] + h[i - 1], h[i] + 2 * h[i - 1]
            m[i] = (w1 + w2) / (w1 / d[i - 1] + w2 / d[i])
    m[0] = ((2 * h[0] + h[1]) * d[0] - h[0] * d[1]) / (h[0] + h[1])
    if m[0] * d[0] <= 0:
        m[0] = 0.0
    elif d[0] * d[1] <= 0 and abs(m[0]) > 3 * abs(d[0]):
        m[0] = 3 * d[0]
    m[-1] = ((2 * h[-1] + h[-2]) * d[-1] - h[-1] * d[-2]) / (h[-1] + h[-2])
    if m[-1] * d[-1] <= 0:
        m[-1] = 0.0
    elif d[-1] * d[-2] <= 0 and abs(m[-1]) > 3 * abs(d[-1]):
        m[-1] = 3 * d[-1]
    return m


def _hermite(xs, ys, m, x):
    """Cubic Hermite eval at x; linear-in-log extrapolation past either end
    (keeps it monotone beyond the marathon anchor)."""
    n = len(xs)
    if x <= xs[0]:
        return ys[0] + m[0] * (x - xs[0])
    if x >= xs[-1]:
        return ys[-1] + m[-1] * (x - xs[-1])
    i = 0
    while i < n - 2 and x >= xs[i + 1]:
        i += 1
    h = xs[i + 1] - xs[i]
    t = (x - xs[i]) / h
    t2, t3 = t * t, t * t * t
    return ((2 * t3 - 3 * t2 + 1) * ys[i] + (t3 - 2 * t2 + t) * h * m[i]
            + (-2 * t3 + 3 * t2) * ys[i + 1] + (t3 - t2) * h * m[i + 1])


def _aero_isotime(dist_m, P):
    """Smooth, monotone iso-time (s) at dist_m for score P, interpolating the
    real race anchors (5K/10K/HM/marathon) in log-distance/log-time. Exact at the
    anchors; monotone (no spikes) between and above them."""
    logt = [math.log(_time_at(e, P)) for e in _AERO]
    m = _mono_tangents(_LOG_AERO, logt)
    return math.exp(_hermite(_LOG_AERO, logt, m, math.log(dist_m)))


def wa_points(dist_m, time_s):
    """WA points for a performance (dist_m, time_s). Exact when dist matches a
    tabled distance (<1%). For >5K the iso-time comes from the smooth monotone
    anchor curve; for <5K it's the legacy log-distance interpolation on the track
    tabs."""
    if time_s <= 0 or dist_m <= 0:
        return float('nan')
    if dist_m >= ANCHOR_M:                       # aerobic: smooth monotone curve
        lo, hi = 50.0, 1400.0                    # (exact at the anchors, so no special case)
        for _ in range(60):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if _aero_isotime(dist_m, mid) > time_s else (lo, mid)
        return (lo + hi) / 2
    for ev in _DISTS:                             # sub-5K: exact track tab (<1%)
        if abs(dist_m - ev) / ev < 0.01:
            a, b, c = COEF[ev]
            return a * time_s * time_s + b * time_s + c
    if dist_m <= _DISTS[0]:                       # sub-5K (track tabs, log-linear)
        a, b, c = COEF[_DISTS[0]]
        return a * time_s * time_s + b * time_s + c
    lo_d = max(e for e in _DISTS if e < dist_m)
    hi_d = min(e for e in _DISTS if e > dist_m)
    w = (math.log(dist_m) - math.log(lo_d)) / (math.log(hi_d) - math.log(lo_d))

    def iso_time(P):
        return math.exp((1 - w) * math.log(_time_at(lo_d, P))
                        + w * math.log(_time_at(hi_d, P)))

    lo, hi = 50.0, 1400.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if iso_time(mid) > time_s else (lo, mid)
    return (lo + hi) / 2


def _time_at_dist(dist_m, P):
    """Equivalent time (s) at any distance for score P. Smooth monotone anchor
    curve for >5K; legacy log-linear track-tab interpolation for <5K."""
    if P != P:
        return float('nan')
    if dist_m >= ANCHOR_M:                        # aerobic: smooth monotone curve
        return _aero_isotime(dist_m, P)
    for ev in _DISTS:                             # sub-5K: exact track tab (<1%)
        if abs(dist_m - ev) / ev < 0.01:
            return _time_at(ev, P)
    if dist_m <= _DISTS[0]:
        return _time_at(_DISTS[0], P)
    lo_d = max(e for e in _DISTS if e < dist_m)
    hi_d = min(e for e in _DISTS if e > dist_m)
    w = (math.log(dist_m) - math.log(lo_d)) / (math.log(hi_d) - math.log(lo_d))
    return math.exp((1 - w) * math.log(_time_at(lo_d, P))
                    + w * math.log(_time_at(hi_d, P)))


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
