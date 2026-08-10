"""Per-run elevation / grade enrichment from the rich detail stream (rich>=2).

Produces, per watch day:
  - smoothed total elevation gain / loss (ft),
  - a Minetti per-run grade-cost factor (multiplicative energy-vs-flat ratio,
    used downstream as a t_eff correction), and
  - per-corrected-mile splits (pace_s, gain_ft, loss_ft) for display and a
    future pace-vs-grade physiology model.

Design notes
------------
* Barometric altitude is noisy, so it's resampled onto an even DISTANCE grid
  (robust to pace) and smoothed before differencing. ``GRID_M`` / ``SMOOTH_M``
  are the knobs; pick them by matching the device's own ``summary.elevGain``
  and surveyed hill-loop climb (see scripts/backfill_elevation.py --validate).
* Unlike the distance calibration (paved-gated — GPS corner-cutting under
  tree cover is a route property), barometric altitude is trustworthy on
  trail too, so elevation enrichment is NOT paved-gated.
* The distance axis is rescaled by ``corr_miles / watch_miles`` so split
  boundaries land on watch-corrected miles. Split pace uses moving time
  (stream-gap / pause seconds removed), never the raw ``speed`` field.
* Minetti grade is clipped to the model's fitted domain (|i| <= 0.45) so the
  quintic never extrapolates on a noisy spike.
"""
import numpy as np
import pandas as pd

from src.shared.hill_model import minetti_cost, FT_PER_M
from src.shared.units import METERS_PER_MILE

GRID_M = 10.0      # even-distance resample step (m)
SMOOTH_M = 120.0   # altitude smoothing window (m of distance). Validated
                   # 2026-06-13: at 120 m, computed gain matches the device's
                   # summary.elevGain within ~1-3% on hilly Boulder runs
                   # (east boulder, boulder turnpike); flat runs sit a little
                   # high (barometric noise dominates when real gain is small,
                   # where the Minetti effect is negligible anyway). Smaller
                   # windows over-count gain badly.
SHAPE_SMOOTH_M = 30.0   # finer window for the grade-SHAPE metrics (two-scale
                        # construction, Aug 2026: totals keep the validated
                        # 120 m; the descent-steepness distribution needs ~30 m
                        # or short steep pitches smear into the gentle grades —
                        # a 320 m road pitch vanishes at 120 m)
GRADE_CLIP = 0.45  # Minetti domain bound on |grade|
GRADE_CLIP_SHAPE = 0.20  # per-step |grade| bound for the weighted-grade
                         # metrics (bounds model-input extrapolation). Distinct
                         # from SPIKE_GRADE_CAP below, which bounds the SUMS.
SPIKE_GRADE_CAP = 1.00   # Per-step |grade| ceiling on the gain/loss sums: 45°,
                         # i.e. physically impossible to run, so this removes
                         # only barometric spikes and never real terrain. Aug
                         # 2026: a 0.20 (20%) cap was tried first and reverted —
                         # technical trail genuinely exceeds 20% over a 10 m
                         # step, and the watch's own summary.elevGain confirmed
                         # the vertical it deleted was real (device agreed with
                         # the raw sum to ~2% on trail days while the cap cut
                         # 30%). The corrupt case it must catch is nothing like
                         # steep terrain: east boulder 2024-04-24 summed steps
                         # of ~66 m per 10 m (a 660% "grade") and read 6.4x the
                         # device. Trail days are the exposed ones because
                         # tree-cover GPS dropouts fail the DEM coverage gate,
                         # so they fall back to raw barometric.
STREAM_GAP_S = 10.0  # gap beyond this between samples = a pause (drop its time)


def alt_points(rec):
    """(t_s, dist_m, alt_m) from a rich>=2 record. Keeps the leading 0-distance
    sample; drops points missing altitude or distance."""
    out = []
    for f in rec.get('freq') or []:
        if len(f) < 6 or f[1] is None or f[5] is None:
            continue
        out.append((f[0] / 100.0, f[1] / 100.0, float(f[5])))
    return out


def _gridded_altitude(dist, alt, smooth_m=SMOOTH_M):
    """Resample altitude onto an even GRID_M distance grid and smooth it over
    ``smooth_m``. Returns (grid_dist, grid_alt_smoothed) or (None, None) if
    too short."""
    dist = np.asarray(dist, float)
    alt = np.asarray(alt, float)
    # Enforce strictly increasing distance (GPS/clock jitter can stall it).
    keep = np.concatenate(([True], np.diff(dist) > 0))
    dist, alt = dist[keep], alt[keep]
    if len(dist) < 3 or dist[-1] - dist[0] < GRID_M:
        return None, None
    grid = np.arange(dist[0], dist[-1], GRID_M)
    galt = np.interp(grid, dist, alt)
    win = max(1, int(round(smooth_m / GRID_M)))
    if win > 1 and len(galt) > 1:
        # Centered rolling mean; min_periods shrinks the window at the ends so
        # no index gymnastics and no edge crash on short/sparse segments.
        galt = (pd.Series(galt).rolling(win, center=True, min_periods=1)
                .mean().to_numpy())
    return grid, galt


def shape_steps(dist, alt):
    """(mid_dist_m, dalt_m) per GRID_M step of the SHAPE_SMOOTH_M-smoothed
    curve — the substrate for the weighted-grade metrics. Returns (None, None)
    if the profile is too short."""
    grid, galt = _gridded_altitude(dist, alt, smooth_m=SHAPE_SMOOTH_M)
    if galt is None or len(galt) < 2:
        return None, None
    return (grid[:-1] + grid[1:]) / 2.0, np.diff(galt)


# --- Hill segmentation -------------------------------------------------------
# A run's grade cost is the sum over its real hills, so the geometry layer has
# to identify those hills before it can describe them. The Aug 2026 rebuild
# (validated on before/after panels + the mile-grain fit):
#
#  * NO elevation smoothing. The device stream is already filtered (flat ground
#    holds a straight line to 0.74 ft RMS per 100 m); the old 80 m rolling mean
#    only destroyed shape — it rounded short pitches under the hill floor and
#    deflated grades ~2x (Boulder Turnpike's real 13.3% descent read 3.9%).
#  * Grade sign from a centred LAG difference. The stream is ~1 ft-quantized,
#    so a grade needs a baseline: one quantum over a 10 m step reads 3% (a
#    phantom pitch), over 40 m it reads 0.75% — under threshold. The lag sets
#    the measurement baseline without touching the curve.
#  * Steps beyond 45 degrees are ZEROED (barometric resets — a level shift, so
#    removing the step restores the level; the totals' SPIKE_GRADE_CAP clip
#    would leave a 32.8 ft residue that clears the hill floor as a phantom).
#  * Same-sign pitches merge across short flat gaps, with the absorbed flat
#    capped as a share of the hill — the old iterative gap-closing was
#    unbounded (it bridged up to 2,920 m and halved real grades).
#  * Grade = vertical / PITCHED distance. Absorbed flat never dilutes it.
SEG_LAG_M = 40.0        # centred lag over which the grade sign is measured
SEG_MIN_GRADE = 2.0     # the grade that makes ground a hill rather than flat;
                        # doubles as the steepness selection — sub-grade ground
                        # is the flat baseline, priced at zero
SEG_GAP_M = 60.0        # a flat interruption up to this long stays inside a hill
SEG_FLAT_FRAC = 0.25    # ... but absorbed flat may be at most this share of
                        # the hill's extent (bounds the merge; it cannot chain)
SEG_MIN_VERT_FT = 12.0  # minimum vertical to count as a hill (momentum floor)
SEG_GRADE_MAX = 100.0   # 45 degrees — the physical bound (NOT the old 20%,
                        # which censored ordinary steep hills)


def _lag_grade(g, lag_m):
    """Per-step grade (%) measured over a centred ``lag_m`` baseline."""
    L = max(1, int(round(lag_m / GRID_M)))
    n = len(g) - 1
    gr = np.zeros(n)
    for i in range(n):
        a = max(0, i - L // 2)
        b = min(len(g) - 1, i + L // 2 + L % 2)
        if b > a:
            gr[i] = (g[b] - g[a]) / ((b - a) * GRID_M) * 100.0
    return gr


def hill_segments(dist, alt):
    """Validated climb and descent segments as
    ``[(d0_m, d1_m, vert_ft, grade_pct, kind, pitched_m, flat_m)]``,
    kind in {1, -1}. Pure geometry — no pace, no time, nothing about how the
    run was executed — so nothing here can be moved by running differently."""
    dist = np.asarray(dist, float)
    alt = np.asarray(alt, float)
    # Enforce strictly increasing distance against the RUNNING MAXIMUM:
    # Coros resets the cumulative-distance field to ~0 on trailing samples
    # (the same quirk that broke the race picker), and after a reset the
    # field can resume counting — a diff>0 mask would keep those small
    # values and leave the axis non-monotonic, silently returning zero
    # hills for the whole day.
    run_max = np.maximum.accumulate(dist)
    keep = np.concatenate(([True], dist[1:] > run_max[:-1]))
    dist, alt = dist[keep], alt[keep]
    if len(dist) < 3:
        return []
    grid = np.arange(dist[0], dist[-1], GRID_M)
    if len(grid) < 6:
        return []
    g = np.interp(grid, dist, alt)
    dg = np.diff(g)
    dg[np.abs(dg) > SPIKE_GRADE_CAP * GRID_M] = 0.0     # zero baro resets
    g = np.concatenate(([g[0]], g[0] + np.cumsum(dg)))
    lg = _lag_grade(g, SEG_LAG_M)
    sign = np.where(lg >= SEG_MIN_GRADE, 1,
                    np.where(lg <= -SEG_MIN_GRADE, -1, 0))
    runs, i = [], 0
    while i < len(sign):
        j = i
        while j < len(sign) and sign[j] == sign[i]:
            j += 1
        runs.append((i, j, int(sign[i])))
        i = j
    gap_steps = max(1, int(round(SEG_GAP_M / GRID_M)))
    hills, cur = [], None
    for a, b, s in runs:
        ln = b - a
        if s == 0:
            if cur is not None and ln <= gap_steps:
                cur['pend'] += ln
            elif cur is not None:
                hills.append(cur)
                cur = None
            continue
        if cur is not None and cur['sign'] == s:
            pit = cur['pitched'] + ln
            fl = cur['flat'] + cur['pend']
            if fl <= SEG_FLAT_FRAC * (pit + fl):
                cur.update(i1=b, pitched=pit, flat=fl, pend=0)
                continue
            hills.append(cur)
            cur = None
        elif cur is not None:
            hills.append(cur)
            cur = None
        cur = {'i0': a, 'i1': b, 'sign': s, 'pitched': ln, 'flat': 0, 'pend': 0}
    if cur is not None:
        hills.append(cur)
    out = []
    for h in hills:
        d0, d1 = float(grid[h['i0']]), float(grid[h['i1']])
        vert = float(g[h['i1']] - g[h['i0']]) / FT_PER_M
        pitched = h['pitched'] * GRID_M
        if pitched <= 0 or abs(vert) < SEG_MIN_VERT_FT:
            continue
        grade = min(abs(vert) * FT_PER_M / pitched * 100.0, SEG_GRADE_MAX)
        out.append((d0, d1, vert, grade, h['sign'], pitched,
                    h['flat'] * GRID_M))
    return out


def segment_sums(segs, lo=None, hi=None):
    """((up_vert_ft, up_vert_x_grade), (dn_vert_ft, dn_vert_x_grade)).

    Returned as SUMS rather than grades so they compose: across the windows of
    a run, or across the several activities of a race, adding the pairs and
    dividing at the end gives exactly the same answer as measuring the whole
    thing at once. Restricted to ``[lo, hi)`` when given, with each segment's
    vertical prorated by how much of it falls inside — so windowing cannot
    create or destroy vertical, and a hill spanning a boundary is shared rather
    than double-counted or dropped."""
    up = dn = (0.0, 0.0)
    for seg in segs:
        d0, d1, vert, grade, kind = seg[:5]
        span = d1 - d0
        if span <= 0:
            continue
        frac = 1.0
        if lo is not None or hi is not None:
            a = max(d0, lo if lo is not None else d0)
            b = min(d1, hi if hi is not None else d1)
            if b <= a:
                continue
            frac = (b - a) / span
        v = abs(vert) * frac
        if kind > 0:
            up = (up[0] + v, up[1] + v * grade)
        else:
            dn = (dn[0] + v, dn[1] + v * grade)
    return up, dn


def grade_from_sums(pair):
    """Vertical-weighted mean segment grade (%) from a (vert, vert*grade) pair.

    Vertical weighting is not a preference, it is the identity that makes a
    linear cost model exact: sum over hills of (c0 + c1*g_i)*vert_i equals
    (c0 + c1*g_bar) * total_vert precisely when g_bar is this mean. Any other
    weighting leaves a residual the fitted coefficients have to absorb."""
    vert, vxg = pair
    return float(vxg / vert) if vert > 0 else 0.0


def segment_vertical(segs):
    """(up_ft, dn_ft) — vertical inside validated hills only. This is the
    magnitude that pairs with the segment grades: the cost model prices hills,
    and sub-threshold ground is the flat baseline it is measured against."""
    up, dn = segment_sums(segs)
    return up[0], dn[0]


def weighted_grades(dist, alt):
    """(g_gain_pct, g_loss_pct) — vertical-weighted mean grade of the validated
    climb and descent segments. (0.0, 0.0) when the profile has no hills.

    These are the engine's two steepness inputs, and they are deliberately
    separate quantities: a run's climbs and its descents have their own
    steepness, and nothing forces them to agree."""
    segs = hill_segments(dist, alt)
    if not segs:
        return 0.0, 0.0
    up, dn = segment_sums(segs)
    return grade_from_sums(up), grade_from_sums(dn)


# --- Baro + DEM fusion --------------------------------------------------------
FUSE_WIN_M = 1500.0     # drift-median window. Must be much longer than any
                        # structure (bridges run 100-400 m — a localized
                        # disagreement cannot move a 1.5 km median) and shorter
                        # than ambient pressure change (kilometres of running).
FUSE_EDGE_M = 300.0     # endpoint drift anchor: near an activity's ends the
                        # centred median can only look one-sided (up to half a
                        # window inward), so endpoint drift lags reality and
                        # the residual lands in the run's NET (Run Mag Mile
                        # read −18 ft fused against a DEM-endpoint truth of
                        # −8). The drift at each end is anchored to the local
                        # median of (baro − DEM) over this many metres and
                        # blended linearly to the rolling median half a window
                        # in. NOT a loop-closure assumption — a genuinely
                        # lower finish keeps its real net; only the
                        # instrument's endpoint drift goes.


def fuse_altitude(dist, alt, dem_dist, dem_alt, win_m=None):
    """Complementary filter: baro shape, DEM low-frequency trend.

        drift = rolling_median(baro - DEM, win_m);  fused = baro - drift

    Baro is right at high frequency (structures, pitches; wrong slowly via
    ambient pressure) and DEM is right at low frequency (net trend, loop
    closure; wrong locally via bare-earth structure stripping and GPS wander) —
    disjoint failure bands, so subtraction of the slow difference keeps the
    best of both. Validated Aug 2026: out-and-back self-mismatch 6.6 -> 3.0 ft
    median, loop |net| 9.8 -> 4.1 ft, cross-visit SD 12.8 -> 8.9 ft, and the
    known structures (arch bridges, the post-lidar railbed) preserved within
    ~2 ft of baro where plain DEM-trust reads them near zero.

    ``dem_dist``/``dem_alt`` are on the SAME distance axis as ``dist`` (both
    from the watch's cumulative-distance field). The drift is measured on the
    DEM span and edge-extended beyond it. Returns fused altitude at ``dist``.
    """
    if win_m is None:
        win_m = FUSE_WIN_M
    orig_dist = np.asarray(dist, float)
    orig_alt = np.asarray(alt, float)
    # Trailing distance-reset samples (see hill_segments) would break the
    # axis; guard against the running maximum, not the neighbour.
    run_max = np.maximum.accumulate(orig_dist)
    mono = np.concatenate(([True], orig_dist[1:] > run_max[:-1]))
    md, ma = orig_dist[mono], orig_alt[mono]
    if len(md) < 3 or len(dem_dist) < 3:
        return orig_alt
    grid = np.arange(md[0], md[-1], GRID_M)
    if len(grid) < 30:
        return orig_alt
    baro = np.interp(grid, md, ma)
    lo, hi = dem_dist[0], dem_dist[-1]
    m = (grid >= lo) & (grid <= hi)
    if m.sum() < 30:
        return orig_alt
    diff = baro[m] - np.interp(grid[m], dem_dist, dem_alt)
    w = max(5, int(round(win_m / GRID_M)))
    med = (pd.Series(diff).rolling(w, center=True, min_periods=max(5, w // 6))
           .median().bfill().ffill().to_numpy().copy())
    # Endpoint anchoring (see FUSE_EDGE_M): pin drift at each end to the local
    # difference and ramp to the centred median half a window in.
    half = w // 2
    if len(diff) > 2 * half + 10:
        k = max(5, int(round(FUSE_EDGE_M / GRID_M)))
        a0 = float(np.median(diff[:k]))
        a1 = float(np.median(diff[-k:]))
        med[:half] = np.linspace(a0, med[half], half, endpoint=False)
        med[-half:] = np.linspace(med[-half - 1], a1, half + 1)[1:]
    drift = np.empty_like(baro)
    drift[m] = med
    drift[grid < lo] = med[0]
    drift[grid > hi] = med[-1]
    # Correct at the ORIGINAL sample positions; np.interp clamps at the grid
    # edges, so trailing reset samples inherit the edge drift value.
    return orig_alt - np.interp(orig_dist, grid, drift)



def gain_loss_ft(dist, alt):
    """(gain_ft, loss_ft) from the smoothed gridded profile, each step bounded
    by ``SPIKE_GRADE_CAP`` (45°) so a barometric spike can't inflate the total
    while every runnable grade — including technical trail — passes through
    untouched. Serves both the barometric and the DEM paths."""
    _, galt = _gridded_altitude(dist, alt)
    if galt is None:
        return 0.0, 0.0
    cap = SPIKE_GRADE_CAP * GRID_M
    d = np.clip(np.diff(galt), -cap, cap)
    return float(d[d > 0].sum() / FT_PER_M), float(-d[d < 0].sum() / FT_PER_M)


def minetti_factor(dist, alt):
    """Distance-weighted Minetti energy cost over the smoothed profile,
    divided by flat cost — a multiplicative grade-cost factor (>=1 ~ flat).
    Each GRID_M step is equal-distance, so this is just mean(cost)/cost(0)."""
    grid, galt = _gridded_altitude(dist, alt)
    if galt is None or len(galt) < 2:
        return 1.0
    grade = np.clip(np.diff(galt) / GRID_M, -GRADE_CLIP, GRADE_CLIP)
    return float(minetti_cost(grade).mean() / minetti_cost(0.0))


def _moving_time(t):
    """Elapsed seconds minus stream gaps (pauses) over a (sorted) time array."""
    t = np.asarray(t, float)
    if len(t) < 2:
        return 0.0
    dt = np.diff(t)
    return float(dt[dt <= STREAM_GAP_S].sum())


def mile_splits(pts, k, segs=None):
    """Per-corrected-mile splits from (t_s, dist_m, alt_m) points, with the
    distance axis scaled by ``k = corr_miles / watch_miles``. Returns a list of
    dicts {mile, pace_s, gain_ft, loss_ft, covered, g_up, g_down, seg_up_ft,
    seg_dn_ft} (pace = moving s/mi; covered = fraction of the mile the stream
    actually spans, so partial boundary miles are identifiable downstream).

    The hill segments are found ONCE on the whole-run curve and then prorated
    into each mile, never re-segmented per mile: smoothing and gap-closing must
    not restart at an arbitrary boundary, or a hill straddling mile 3 and mile 4
    would be measured as two shorter, and therefore different, hills.

    g_up/g_down are the vertical-weighted mean grades of that mile's climb and
    descent segments; seg_up_ft/seg_dn_ft are the vertical inside those segments.
    The grades pair with the SEGMENT verticals, not with gain_ft/loss_ft, which
    include the sub-threshold ground the segmentation calls flat — mixing the two
    bases is what leaves a residual for the coefficients to absorb."""
    if len(pts) < 3:
        return []
    t = np.array([p[0] for p in pts])
    d = np.array([p[1] for p in pts]) * k / METERS_PER_MILE  # corrected miles
    a = np.array([p[2] for p in pts])
    if segs is None:
        segs = hill_segments(d * METERS_PER_MILE, a)
    out = []
    last = int(np.floor(d[-1]))
    for mi in range(last + 1):
        m = (d >= mi) & (d < mi + 1)
        if m.sum() < 2:
            continue
        seg_d = d[m]
        covered = seg_d[-1] - seg_d[0]
        if covered <= 0:
            continue
        mov = _moving_time(t[m])
        g, l = gain_loss_ft(seg_d * METERS_PER_MILE, a[m])
        up, dn = segment_sums(segs, mi * METERS_PER_MILE,
                              (mi + 1) * METERS_PER_MILE)
        out.append({'mile': mi,
                    'pace_s': round(mov / covered, 1),
                    'gain_ft': round(g, 1),
                    'loss_ft': round(l, 1),
                    'covered': round(float(covered), 3),
                    'g_up': round(grade_from_sums(up), 2),
                    'g_down': round(grade_from_sums(dn), 2),
                    'seg_up_ft': round(up[0], 1),
                    'seg_dn_ft': round(dn[0], 1)})
    return out


def _stitch(recs, dem_profiles=None):
    """Concatenate multiple activities into ONE monotonic (t, d, a) profile,
    fusing each activity's altitude with its DEM profile where one exists
    (``dem_profiles`` aligns with ``recs``; see fuse_altitude — fusion happens
    per activity because drift and DEM coverage are per-activity).

    Each activity's distance stream restarts at 0, so a naive concatenation
    of a warmup+main+cooldown day yields non-monotonic distance and a
    scrambled altitude profile (np.interp needs sorted x). Offset each
    activity's distance by the cumulative prior distance, and rebase its time
    after a >STREAM_GAP_S marker so split moving-time skips the rest between
    activities. Returns (points, act_bounds) with act_bounds the [d_lo, d_hi)
    RAW-axis span of each rec (None for recs with no stream), so callers can
    map a stitched position back to its activity."""
    out, bounds = [], []
    d_off = t_off = 0.0
    for i, rec in enumerate(recs):
        pts = alt_points(rec)
        if not pts:
            bounds.append(None)
            continue
        prof = dem_profiles[i] if dem_profiles else None
        if prof is not None:
            dd = np.array([q[1] for q in pts], float)
            aa = np.array([q[2] for q in pts], float)
            keep = np.concatenate(([True], np.diff(dd) > 0))
            fused = np.interp(dd, dd[keep],
                              fuse_altitude(dd[keep], aa[keep],
                                            prof[0], prof[1]))
            pts = [(q[0], q[1], f) for q, f in zip(pts, fused)]
        t0 = pts[0][0]
        for (t, d, a) in pts:
            out.append((t - t0 + t_off, d + d_off, a))
        bounds.append((d_off, d_off + pts[-1][1]))
        d_off += pts[-1][1]
        t_off = out[-1][0] + STREAM_GAP_S + 1.0
    return out, bounds


def measure_day_elevation(recs, corr_miles, watch_miles, dem_profiles=None):
    """Aggregate elevation metrics for one day's run activities (``recs`` is a
    list of rich records, time-ordered). ``corr_miles`` is the watch-corrected
    distance (for the split-axis rescale and per-mile pace); ``watch_miles`` is
    the raw measured distance. ``dem_profiles`` (aligned with recs) enables the
    baro+DEM fusion. Returns a dict or None if no altitude stream. ``hills``
    are on the corrected stitched axis; ``act_bounds_raw`` on the raw one."""
    all_pts, bounds = _stitch(recs, dem_profiles)
    if len(all_pts) < 3:
        return None
    # The stream's distance axis is the watch's (it under-reads), so scale it
    # to the corrected distance before computing grade — grade = dAlt/dDist
    # depends on the horizontal axis. Vertical gain/loss is axis-invariant but
    # the rescale keeps everything on one (corrected) axis.
    k = (corr_miles / watch_miles) if (watch_miles and watch_miles > 0) else 1.0
    pts = [(t, d * k, a) for (t, d, a) in all_pts]
    dist = np.array([p[1] for p in pts], float)
    alt = np.array([p[2] for p in pts], float)
    g, l = gain_loss_ft(dist, alt)
    mf = minetti_factor(dist, alt)
    segs = hill_segments(dist, alt)
    up, dn = segment_sums(segs)
    return {
        'elev_gain_ft': round(g, 1),
        'elev_loss_ft': round(l, 1),
        'minetti_factor': round(mf, 5),
        'g_gain_pct': round(grade_from_sums(up), 2),
        'g_loss_pct': round(grade_from_sums(dn), 2),
        'splits': mile_splits(pts, 1.0, segs),  # pts already corrected-axis
        'hills': segs,
        'act_bounds_raw': bounds,
        'n_alt_pts': len(all_pts),
    }
