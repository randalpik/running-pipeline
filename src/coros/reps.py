"""reps.py — watch-derived rep extraction (Phase B of workout enrichment).

Reconstructs per-rep structure (distance, time, pace, rest, HR) for quality
workout days from rich Coros detail records (see build_current_log.
rich_detail), reconciled against the hand log. Writes workout_measured.csv.

The algorithm, validated June 2026 against Max's full watch-era corpus
(59/66 quality days reconstruct to the meter; the rest explicitly
disqualified — see DISQUALIFIED):

 1. Split each activity's per-second stream into moving SEGMENTS at watch
    pauses. A watch stop pins a rep END; a watch start pins nothing.
 2. 100m BINS anchored at each segment start (last bin 50–150m). 2-means
    cluster all bin paces into quality/jog; repair isolated slow bins (the
    track's GPS dead zone fires once per 400m lap); a block starting at
    bin 1 absorbs bin 0 (start line sits in the dead zone).
 3. Contiguous quality bins -> coarse BLOCKS. Refinement proposes
    ALTERNATIVE extents — never replacements — from evidence:
    two-sided changepoint fits at free edges, leading-slow trims,
    trailing-jog trims at CS+20 crossings, valley splits (rep+jog+stride),
    short-tail extensions (trailing dead zone), a sibling-jog prior on
    noise-mangled blocks, GPS-inflation/deflation lap-ratio rescales, and
    GPS-freeze distance restoration.
 4. Extents snap to 100m multiples (±75m), each variant priced by Max's
    rep-length prior: tier1 {200,300,400,800,1600} free, tier2
    {100,500,other ×200} cheap, other ×100 expensive.
 5. RECONCILIATION (the invariant): a segment-grouped DP picks per-block
    candidates + optional fragments so the day total EXACTLY equals the
    hand-logged quality distance (target snapped to 100m), minimizing
    cost (alternatives used, optionals included, tier penalties). No
    solution -> the day is disqualified rather than approximated.
    `f@` days may instead account whole segments as pieces; the cf label
    applies only when the entire workout is one unbroken chunk.

Per-100m pace is a classification signal ONLY — never a measurement
(track mode verifies distance per lap; finer pace is GPS noise). All
reported numbers are block-level aggregates anchored to watch events.

Continuous-hill days (run_type hill_cont) take a separate GPS-anchored path
(see extract_hill_day): find the loop point — the hand log fixes the loop
count, the surveyed loop distance stays authoritative — and measure the
block's exact moving time plus per-loop splits. Statuses hill-exact /
hill-total / hill-no-block; loop rows carry kind='loop'. Flat-workout
consumers filter on status exact|watch-only and never see hill rows.

Reads from DATA_DIR (per-profile via RP_DATA_DIR): daily.csv,
bayes_cs_summary.csv, races.csv (watch-only mode); rich details from
--details-dir. Writes DATA_DIR/workout_measured.csv with one status row
(rep_idx=0) per analyzed day plus one row per reconstructed rep.
"""
import argparse
import bisect
import json
import math
import statistics
import sys
from itertools import combinations, product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from src.coros import mappings as M
from src.coros.build_current_log import Activity
from src.shared.paths import DATA_DIR
from src.shared.workouts import hc_loop_distance, parse_hc

MILE = 1609.344
QUALITY_TYPES = {'tempo', 'interval', 'rep', 'fartlek'}

CHUNK = 20.0          # m, fine evaluation step
FINE_W = 60.0         # m, fine pace window (20m windows are GPS noise)
FIT_W = 100.0         # m, fit window each side of a candidate edge
SEARCH = 160.0        # m, search radius around the grid edge (start edges)
END_SEARCH = 460.0    # m, end edges: must reach back past a full 400m trailing jog
STEP = 10.0           # m, search step
MARGIN = 0.85         # refined score must be <= 85% of grid-edge score
MIN_TRIM = 80.0       # m, minimum leading-jog run to trim at a segment start
MIN_JOG_EST = 60.0    # m, jog gap needed to estimate a local jog level

TIER1 = {200, 300, 400, 800, 1600}
# Max runs 100s only occasionally (descending-ladder tails) and 500s only in
# continuous fartlek (never as structured intervals/reps) — so both are
# plausible-but-non-standard lengths: tier-2, not free. Demoting 500 is what
# lets a spurious 100m float glued onto a 400 (read as a "500") lose to the
# true 400 in reconciliation, instead of forcing the error onto a neighbour.
TIER2 = {100, 500}
ALT_W, OPT_W = 2.0, 0.5

# Coverage bias (soft, applied in reconcile): a watch segment is USUALLY one
# rep, so prefer decompositions that DON'T split a segment into several reps
# when an equal-total alternative keeps them in their own segments. A BIAS,
# not a rule — continuous fartlek is many reps in one segment, and a single
# segment can legitimately hold two; small enough to only break ties.
# Deliberately one-sided: we penalize EXTRA reps per segment but do NOT reward
# "use every segment" — a salvaged jog core (see _salvage_core) is offered but
# left unused unless it genuinely helps hit the total; rewarding coverage would
# pull recovery-jog segments in as phantom reps.
COVERAGE_EXTRA = 0.4
MIN_SALVAGE = 80.0    # m, shortest quality core worth salvaging from a segment
                      # whose block the cs_cut filter dropped
SALVAGE_W = 0.3       # extra cost on a salvaged core so it is a LAST RESORT —
                      # used only when it strictly beats the alternatives (e.g.
                      # avoids splitting a segment), never to manufacture a tie
                      # on a day that already reconciled without it. < COVERAGE_
                      # EXTRA so a salvaged own-segment rep still beats a split.

# Days adjudicated by Max (June 2026) as not worth watch enrichment: the
# recording or the conditions make exact reconstruction impossible. These
# apply to the hand-log corpus; watch-only profiles never consult this.
DISQUALIFIED = {
    '2023-09-10': 'mid-rep watch stops, structure under-determined',
    '2021-04-26': 'XC',
    '2022-07-29': 'TT, partial recording',
    '2023-05-28': 'recording missing ~1500m',
    '2024-01-10': 'road anomaly',
    '2024-02-04': 'snow',
    '2025-03-11': 'road anomaly',
    '2024-05-19': 'hybrid day, ignored per Max',
}


# ---------- rich records / segments ----------

def _freq_points(rec):
    """Scaled (t_s, dist_m, heart) from a rich record, glitch zeros dropped."""
    return [(f[0] / 100.0, f[1] / 100.0, f[2])
            for f in rec.get('freq') or [] if f[1]]


def _freq_gps(rec):
    """Scaled (t_s, dist_m, lat, lon) for points carrying a GPS fix."""
    return [(f[0] / 100.0, f[1] / 100.0, f[3] / 1e7, f[4] / 1e7)
            for f in rec.get('freq') or [] if f[1] and f[3]]


class Segment:
    def __init__(self, pts):
        self.pts = pts
        self.ds = [d for _, d, _ in pts]
        self.ts = [t for t, _, _ in pts]
        self.d0, self.d1 = self.ds[0], self.ds[-1]

    def time_at(self, d):
        i = bisect.bisect_left(self.ds, d)
        if i == 0:
            return self.ts[0]
        if i >= len(self.ds):
            return self.ts[-1]
        da, db = self.ds[i-1], self.ds[i]
        ta, tb = self.ts[i-1], self.ts[i]
        return ta + (d - da) / (db - da) * (tb - ta) if db > da else ta

    def pace(self, a, b):
        a, b = max(a, self.d0), min(b, self.d1)
        if b - a < 1:
            return None
        return (self.time_at(b) - self.time_at(a)) / ((b - a) / MILE)

    def chunk_paces(self, a, b):
        out = []
        x = a
        while x + FINE_W <= b + 0.01:
            p = self.pace(x, x + FINE_W)
            if p is not None:
                out.append(p)
            x += CHUNK
        return out


def moving_segments(rec, gap_s=10):
    pts = _freq_points(rec)
    if not pts:
        return []
    segs, cur = [], [pts[0]]
    for prev, p in zip(pts, pts[1:]):
        if p[0] - prev[0] > gap_s:
            segs.append(cur)
            cur = []
        cur.append(p)
    segs.append(cur)
    return [Segment(s) for s in segs if s[-1][1] - s[0][1] >= 30]


def _hav(a, b):
    la1, lo1, la2, lo2 = map(math.radians, [a[0], a[1], b[0], b[1]])
    h = (math.sin((la2-la1)/2)**2
         + math.cos(la1) * math.cos(la2) * math.sin((lo2-lo1)/2)**2)
    return 2 * 6371000 * math.asin(math.sqrt(h))


def seg_lap_ratios(recs):
    """Reported meters per physical lap, per moving segment (keyed by start time).

    Coros track-mode distance failures are segment-scoped: GPS-jitter
    inflation while the track fix settles (observed 416-444 m/lap, sometimes
    persisting into the second segment) and occasional deflation (<=390).
    Healthy band observed 394-408.
    """
    ratios = {}
    for rec in recs:
        pts = _freq_gps(rec)
        if not pts:
            continue
        segs, cur = [], [pts[0]]
        for prev, p in zip(pts, pts[1:]):
            if p[0] - prev[0] > 10:
                segs.append(cur)
                cur = []
            cur.append(p)
        segs.append(cur)
        for seg in segs:
            if seg[-1][1] - seg[0][1] < 700:        # need >=2 laps to measure
                continue
            ref = (seg[8][2], seg[8][3]) if len(seg) > 8 else (seg[0][2], seg[0][3])
            passes, last_d = [], -1e9
            for t, d, la, lo in seg:
                if _hav(ref, (la, lo)) < 20 and d - last_d > 250:
                    passes.append(d)
                    last_d = d
            laps = [(b - a) / max(1, round((b - a) / 400))
                    for a, b in zip(passes, passes[1:])]
            if laps:
                ratios[round(seg[0][0])] = statistics.median(laps)
    return ratios


# ---------- continuous hills (loop-point detection) ----------
# A hill_cont day is loops of one surveyed circuit. The hand log fixes the
# loop count, the surveyed loop distance stays authoritative (GPS reads ~5%
# short under tree cover) — the watch contributes exact TIME. Max starts and
# stops the hill-block recording at the loop point, so activity bounds pin
# loop 1's start / loop N's end and the anchor's nreps-1 interior crossings
# pin the rest; on merged recordings (warmup/cooldown in the same activity)
# the block bounds are the first/last crossing of a regular run instead.
# Mid-block watch pauses are subtracted from time, nothing more.

HILL_RADIUS = 20.0            # m, anchor pass radius (seg_lap_ratios')
HILL_CAND_STEP = 30           # try an anchor candidate every ~30 GPS fixes
HILL_GAP_BAND = (0.75, 1.10)  # crossing gap as fraction of surveyed loop
HILL_EDGE_EPS = 15.0          # s, crossing this close to a block bound IS it
HILL_MERGE_FRAC = 1.25        # fragment beyond this many loops = merged jog
HILL_SPLIT_TOL = 0.25         # splits within this of their median = per-loop


def _gps_stream(rec, gap_s=10):
    """Flattened (t, d, lat, lon) for one activity; pause-split runs shorter
    than 10 fixes dropped (GPS settle noise)."""
    pts = _freq_gps(rec)
    if not pts:
        return []
    segs, cur = [], [pts[0]]
    for prev, p in zip(pts, pts[1:]):
        if p[0] - prev[0] > gap_s:
            segs.append(cur)
            cur = []
        cur.append(p)
    segs.append(cur)
    return [p for s in segs if len(s) >= 10 for p in s]


def _hav_np(ref, lat, lon):
    la1, lo1 = math.radians(ref[0]), math.radians(ref[1])
    la2, lo2 = np.radians(lat), np.radians(lon)
    h = (np.sin((la2 - la1) / 2) ** 2
         + math.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2)
    return 2 * 6371000 * np.arcsin(np.sqrt(h))


def _anchor_crossings(ref, ts, ds, lat, lon, min_adv):
    """(t, d) passes of ref, deduped by cumulative-distance advance (a runner
    dawdling at the anchor scores one crossing, not many)."""
    idx = np.nonzero(_hav_np(ref, lat, lon) < HILL_RADIUS)[0]
    out, last_d = [], -1e9
    for i in idx:
        if ds[i] - last_d > min_adv:
            out.append((float(ts[i]), float(ds[i])))
            last_d = ds[i]
    return out


def _regular_run(vs, loop_m):
    """Longest consecutive sub-run of crossings whose distance gaps look like
    one loop: median inside HILL_GAP_BAND, every gap within 25% of median.
    Isolates the loop block from stray warmup/cooldown passes."""
    best_len, best = 0, None
    for i in range(len(vs)):
        for j in range(i + 1, len(vs)):
            gaps = [vs[k + 1][1] - vs[k][1] for k in range(i, j)]
            med = statistics.median(gaps)
            if not (HILL_GAP_BAND[0] * loop_m <= med
                    <= HILL_GAP_BAND[1] * loop_m):
                continue
            if any(abs(g - med) > 0.25 * med for g in gaps):
                continue
            if j - i + 1 > best_len:
                best_len, best = j - i + 1, (i, j)
    return best


def _hill_pause_s(rec, a, b):
    """Watch-pause seconds starting inside [a, b] (timestamps epoch-seconds)."""
    return sum(dur / 100.0 for ps, _e, dur in rec.get('pauses') or []
               if ps and a <= ps / 100.0 <= b)


def _hill_eval(rec, pts, rv, nreps):
    """Score one regular run: block bounds, pause-corrected splits, totals.

    A fragment under HILL_MERGE_FRAC loops outside the run belongs to the
    block (a partial whose boundary crossing the GPS missed — activity
    started/stopped at the loop point); a larger fragment is a merged jog
    and the crossing itself is the block bound."""
    t0, t1 = pts[0][0], pts[-1][0]
    d0, d1 = pts[0][1], pts[-1][1]
    gap_m = statistics.median(b[1] - a[1] for a, b in zip(rv, rv[1:]))
    b_start, sk = ((t0, 'act') if rv[0][1] - d0 < HILL_MERGE_FRAC * gap_m
                   else (rv[0][0], 'cross'))
    b_end, ek = ((t1, 'act') if d1 - rv[-1][1] < HILL_MERGE_FRAC * gap_m
                 else (rv[-1][0], 'cross'))
    inner = [v[0] for v in rv if v[0] - b_start > HILL_EDGE_EPS
             and b_end - v[0] > HILL_EDGE_EPS]
    bounds = [b_start] + inner + [b_end]
    splits = [(b - a) - _hill_pause_s(rec, a, b)
              for a, b in zip(bounds, bounds[1:])]
    med = statistics.median(splits)
    consistent = (len(splits) == nreps
                  and all(abs(s - med) <= HILL_SPLIT_TOL * med
                          for s in splits))
    cv = (statistics.pstdev(splits) / statistics.mean(splits)
          if len(splits) > 1 else 9.9)
    return {'rec': rec, 'bounds': bounds, 'nrun': len(rv),
            'total_s': (b_end - b_start) - _hill_pause_s(rec, b_start, b_end),
            'splits': splits, 'consistent': consistent, 'cv': cv,
            'kinds': (sk, ek), 'gap_m': gap_m}


def _hill_annotate(row, rec, a, b):
    """Per-piece HR + standing rest from the activity's (t, d, h) stream."""
    hs = [h for t, _, h in _freq_points(rec) if a <= t <= b and h]
    row['avg_hr'] = round(statistics.mean(hs)) if hs else None
    row['max_hr'] = max(hs) if hs else None
    row['rest_stand_s'] = round(_hill_pause_s(rec, a, b))
    return row


def extract_hill_day(recs, nreps, loop_m):
    """Locate the day's hill block and measure it.

    Returns (loops, status): 'hill-exact' -> one dict per loop (surveyed
    dist, measured moving time); 'hill-total' -> one aggregate dict (block
    bounds had to be activity bounds for the total to be trusted);
    'hill-no-block' -> nothing usable."""
    exact, generic = None, None
    for rec in recs:
        pts = _gps_stream(rec)
        if len(pts) < 60:
            continue
        ts = np.array([p[0] for p in pts])
        ds = np.array([p[1] for p in pts])
        lat = np.array([p[2] for p in pts])
        lon = np.array([p[3] for p in pts])
        cands = [(pts[i][2], pts[i][3])
                 for i in range(0, len(pts), HILL_CAND_STEP)]
        # the block-end position is the strongest loop-point candidate on a
        # standalone block recording (no cold-start GPS drift there)
        cands.append((statistics.median(p[2] for p in pts[-5:]),
                      statistics.median(p[3] for p in pts[-5:])))
        for ref in cands:
            vs = _anchor_crossings(ref, ts, ds, lat, lon,
                                   0.5 * 0.95 * loop_m)
            if len(vs) < 2:
                continue
            run = _regular_run(vs, loop_m)
            if run is None:
                continue
            r = _hill_eval(rec, pts, vs[run[0]:run[1] + 1], nreps)
            if r['consistent'] and (exact is None or r['cv'] < exact['cv']):
                exact = r
            gscore = (r['nrun'], -r['cv'])
            if generic is None or gscore > generic['_score']:
                generic = {**r, '_score': gscore}

    if exact is not None:
        rec, bounds = exact['rec'], exact['bounds']
        loops = [_hill_annotate({'L': loop_m, 't': s}, rec, a, b)
                 for s, a, b in zip(exact['splits'], bounds, bounds[1:])]
        return loops, 'hill-exact'
    if generic is not None and generic['kinds'] == ('act', 'act'):
        rec, bounds = generic['rec'], generic['bounds']
        agg = _hill_annotate({'L': nreps * loop_m, 't': generic['total_s']},
                             rec, bounds[0], bounds[-1])
        return [agg], 'hill-total'
    return [], 'hill-no-block'


# ---------- coarse blocks ----------

def seg_bin_edges(seg, bin_m=100):
    length = seg.d1 - seg.d0
    n_full = int(length // bin_m)
    edges = [seg.d0 + i * bin_m for i in range(n_full + 1)]
    rem = length - n_full * bin_m
    if rem >= 50:
        edges.append(seg.d0 + length)
    elif n_full:
        edges[-1] = seg.d0 + length
    return edges


def coarse_blocks(segments):
    """Cluster bins across the day, return per-segment coarse block extents."""
    per_seg = []
    all_paces = []
    for seg in segments:
        edges = seg_bin_edges(seg)
        paces = [seg.pace(a, b) for a, b in zip(edges, edges[1:])]
        per_seg.append((seg, edges, paces))
        all_paces.extend(paces)
    if not all_paces:
        return []
    c0, c1 = min(all_paces), max(all_paces)
    for _ in range(30):
        g0 = [p for p in all_paces if abs(p - c0) <= abs(p - c1)]
        g1 = [p for p in all_paces if abs(p - c0) > abs(p - c1)]
        if not g0 or not g1:
            break
        c0, c1 = statistics.mean(g0), statistics.mean(g1)
    thresh = (c0 + c1) / 2
    weak = (c1 - c0) < 45      # near-unimodal: continuous quality, no jog cluster

    out = []   # [seg, d_start, d_end, at_seg_start, at_seg_end]
    for seg, edges, paces in per_seg:
        if weak:
            q = [True] * len(paces)
        else:
            q = [p <= thresh for p in paces]
            for i in range(1, len(q) - 1):          # dead-zone repair
                if not q[i] and q[i-1] and q[i+1]:
                    q[i] = True
            if len(q) > 1 and not q[0] and q[1]:    # start line in dead zone
                q[0] = True
        i = 0
        while i < len(q):
            if q[i]:
                j = i
                while j < len(q) and q[j]:
                    j += 1
                out.append([seg, edges[i], edges[j], i == 0, j == len(q)])
                i = j
            else:
                i += 1
    return out


# ---------- refinement (alternative extents, never replacements) ----------

def interior_level(seg, a, b):
    span = b - a
    return statistics.median(seg.chunk_paces(a + span * 0.25, b - span * 0.25)
                             or seg.chunk_paces(a, b))


def jog_level_for_gap(seg, a, b):
    if b - a < MIN_JOG_EST:
        return None
    pad = min(30.0, (b - a - MIN_JOG_EST) / 2 + 1)
    ch = seg.chunk_paces(a + pad, b - pad)
    return statistics.median(ch) if ch else None


def edge_score(seg, s, jog_lvl, int_lvl, mode, lo, hi):
    """Two-sided fit at candidate edge s ('start': jog left / interior right)."""
    left = seg.chunk_paces(max(lo, s - FIT_W), s)
    right = seg.chunk_paces(s, min(hi, s + FIT_W))
    jog_side, int_side = (left, right) if mode == 'start' else (right, left)
    if not int_side:
        return None
    res = sum(abs(p - int_lvl) for p in int_side) / len(int_side)
    if jog_side:
        res += sum(abs(p - jog_lvl) for p in jog_side) / len(jog_side)
    return res


def refine_blocks(blocks, cs_cut):
    """Attach alternative extents to each coarse block, in place."""
    # day-level jog estimate (fallback): all inter-block gaps within segments
    day_jogs = []
    by_seg = {}
    for blk in blocks:
        by_seg.setdefault(id(blk[0]), []).append(blk)
    gaps = {}   # seg-id -> [(a, b)] jog gaps adjacent to blocks
    for sid, blks in by_seg.items():
        blks.sort(key=lambda b: b[1])
        seg = blks[0][0]
        bounds = [seg.d0] + [x for b in blks for x in (b[1], b[2])] + [seg.d1]
        seg_gaps = [(bounds[i], bounds[i+1]) for i in range(0, len(bounds) - 1, 2)]
        gaps[sid] = seg_gaps
        for a, b in seg_gaps:
            lvl = jog_level_for_gap(seg, a, b)
            if lvl is not None:
                day_jogs.append(lvl)
    day_jog = statistics.median(day_jogs) if day_jogs else None

    for blk in blocks:
        seg, d_start, d_end, at_start, at_end = blk
        blk.append([])   # alt extent slots
        if d_end - d_start < 300:
            continue

        # short-tail extension: the segment continues <200m past the block end
        # with no room for a real jog — the tail is the rep's final meters
        # reading slow (trailing dead zone). Offer the full extent.
        if not at_end and 0 < seg.d1 - d_end < 200:
            blk[5].append((d_start, seg.d1))

        # trailing-jog trim: the block's tail is recovery that sneaked under
        # the cluster threshold (rep + jog merged). Cut where two consecutive
        # windows cross CS+20, with a quality-side prefix median and a
        # slow-side suffix median. Suffix >=100m.
        wins = []
        x = d_start
        while x + FINE_W <= d_end:
            wins.append((x, seg.pace(x, x + FINE_W)))
            x += CHUNK
        for i in range(len(wins) - 1):
            e = wins[i][0]
            if e - d_start < 200 or d_end - e < 100:
                continue
            if not (wins[i][1] > cs_cut and wins[i+1][1] > cs_cut):
                continue
            before = [p for wx, p in wins if wx + FINE_W <= e]
            after = [p for wx, p in wins if wx >= e]
            if (len(before) >= 3 and len(after) >= 3
                    and statistics.median(before) <= cs_cut
                    and statistics.median(after) > cs_cut):
                blk[5].append((d_start, e))
                break

        # interior-relative trailing-float trim: the block's tail decelerated
        # into a float — slower than the rep core but UNDER the absolute CS
        # cutoff, so the 2-means glued it on (the trailing-jog trim above,
        # keyed on cs_cut, can't see a sub-cutoff float) — while real jog lies
        # past the block end. Cut at the START of the trailing contiguous run
        # above core+45 (scan back from d_end, NOT the first interior window
        # over it — an interior dead-zone spike followed by clean rep must not
        # move the boundary). Only for blocks that do NOT reach the segment
        # end (a watch stop pins the rep there); over-reaching at_end blocks
        # are handled by the sibling-rep prior below.
        if not at_end and seg.d1 - d_end >= 150:
            bw = seg.chunk_paces(d_start, d_end)
            core = (statistics.quantiles(bw, n=4)[0] if len(bw) >= 4
                    else statistics.median(bw)) if bw else None
            beyond = jog_level_for_gap(seg, d_end, seg.d1)
            if core is not None and beyond is not None and beyond > core + 90:
                fw = []
                x = d_start
                while x + FINE_W <= d_end:
                    fw.append((x, seg.pace(x, x + FINE_W)))
                    x += CHUNK
                thr = core + 45
                if fw and fw[-1][1] is not None and fw[-1][1] > thr:
                    k = len(fw) - 1
                    while k >= 0 and fw[k][1] is not None and fw[k][1] > thr:
                        k -= 1
                    cut = fw[k + 1][0]
                    if (cut - d_start >= 200 and d_end - cut >= 80
                            and abs(cut - d_end) >= 40):
                        blk[5].append((d_start, cut))

        # valley split: a brief jog valley bridged by a trailing stride into
        # the watch stop (rep + jog + stride reads as one fast block).
        if at_end and d_end - d_start <= 800:
            for i in range(3, len(wins) - 1):
                vx = wins[i][0]
                if vx - d_start < (d_end - d_start) / 2 or d_end - vx > 400:
                    continue
                lead_med = statistics.median(p for _, p in wins[:i])
                valley = [p for _, p in wins[i:i+2]]
                if len(valley) >= 2 and all(p > lead_med + 40 for p in valley):
                    blk[5].append((d_start, vx))
                    break

        # leading-slow trim: prefix of windows slower than the CS cutoff
        if at_start and at_end:
            run = 0.0
            x = d_start
            while x + FINE_W <= d_end:
                p = seg.pace(x, x + FINE_W)
                if p is None or p <= cs_cut:
                    break
                run += CHUNK
                x += CHUNK
            if run >= MIN_TRIM and d_end - (d_start + run) >= 100:
                blk[5].append((d_start + run, d_end))

        int_lvl = interior_level(seg, d_start, d_end)
        seg_gaps = gaps[id(seg)]

        # start edge (always refinable: a watch start pins nothing)
        gap = next(((a, b) for a, b in seg_gaps if abs(b - d_start) < 1), None)
        jog_lvl = (jog_level_for_gap(seg, *gap) if gap else None) or day_jog
        if jog_lvl is not None and jog_lvl > int_lvl + 20:
            lo = gap[0] if gap and not at_start else seg.d0
            hi = min(d_start + SEARCH, d_end - 100)
            lo_s = max(lo, d_start - SEARCH) if not at_start else seg.d0
            base = edge_score(seg, d_start, jog_lvl, int_lvl, 'start', lo, d_end)
            best, best_s = base, d_start
            s = lo_s
            while s <= hi:
                sc = edge_score(seg, s, jog_lvl, int_lvl, 'start', lo, d_end)
                if sc is not None and sc < best:
                    best, best_s = sc, s
                s += STEP
            ok = base is not None and best < base * MARGIN
            if ok and at_start and (best_s - seg.d0) < MIN_TRIM:
                ok = False   # standing-start acceleration, not a roll-in
            if ok and abs(best_s - d_start) >= 40:
                blk[5].append((best_s, d_end))

        # end edge (pinned when the block ends at the watch stop)
        if not at_end:
            alt_start = blk[5][-1][0] if blk[5] else d_start
            gap = next(((a, b) for a, b in seg_gaps if abs(a - d_end) < 1), None)
            jog_lvl = (jog_level_for_gap(seg, *gap) if gap else None) or day_jog
            if jog_lvl is not None and jog_lvl > int_lvl + 20:
                hi = gap[1] if gap else seg.d1
                base = edge_score(seg, d_end, jog_lvl, int_lvl, 'end', d_start, hi)
                best, best_e = base, d_end
                e = max(d_start + 100, d_end - END_SEARCH)
                while e <= min(hi, d_end + END_SEARCH):
                    sc = edge_score(seg, e, jog_lvl, int_lvl, 'end', d_start, hi)
                    if sc is not None and sc < best:
                        best, best_e = sc, e
                    e += STEP
                if (base is not None and best < base * MARGIN
                        and abs(best_e - d_end) >= 40):
                    blk[5].append((alt_start, best_e))

    # sibling-jog prior: trailing jogs are consistent within a day; the
    # day-median jog length pins rep extents. Fires on two evidence patterns
    # (gated — ungated it overthrows verified structures with cheap bogus
    # alternatives): (a) noise-mangled blocks (window-pace IQR > 60 s/mi);
    # (b) an at_end block whose proposed rep (d1 - J) matches the day's clean
    # SIBLING rep length — a uniform Nx workout where one segment's recovery
    # jog got bridged onto the rep (a fast dip dropping below the cluster
    # threshold) so the coarse block over-reached to the segment end. The
    # sibling match is robust to a noisy/gentle recovery that jog-depth gates
    # miss (the rep core + the consistent segment length carry it).
    last_in_seg = {}
    for blk in blocks:
        sid = id(blk[0])
        if sid not in last_in_seg or blk[1] > last_in_seg[sid][1]:
            last_in_seg[sid] = blk
    tails = [blk[0].d1 - blk[2] for blk in last_in_seg.values()
             if blk[0].d1 - blk[2] >= 150]
    # clean sibling rep lengths: last-block extents that end before the
    # segment end (a watch stop pinned the rep — not over-reached)
    sib_reps = [blk[2] - blk[1] for blk in last_in_seg.values() if not blk[4]]
    sib_med = statistics.median(sib_reps) if len(sib_reps) >= 2 else None
    if len(tails) >= 3:
        J = statistics.median(tails)
        for blk in last_in_seg.values():
            seg = blk[0]
            alt_e = seg.d1 - J
            if not (alt_e - blk[1] >= 100 and abs(alt_e - blk[2]) >= 40):
                continue
            wins = []
            x = blk[1]
            while x + FINE_W <= blk[2]:
                wins.append(seg.pace(x, x + FINE_W))
                x += CHUNK
            noise_mangled = (len(wins) >= 4
                             and statistics.quantiles(wins, n=4)[2]
                             - statistics.quantiles(wins, n=4)[0] > 60)
            sibling_match = (blk[4] and sib_med is not None
                             and abs((alt_e - blk[1]) - sib_med) <= 75)
            if noise_mangled or sibling_match:
                blk[5].append((blk[1], alt_e))
    return blocks


# ---------- candidates (snap variants + distance-scale corrections) ----------

def tier_penalty(L):
    """Max's rep-length prior: tier1 free, tier2 (100/500/other x200) cheap,
    other x100 dear."""
    if L in TIER1:
        return 0.0
    if L in TIER2 or L % 200 == 0:
        return 0.6
    return 2.5


def _snap_variants(seg, d_start, d_end, at_end, is_alt, t_override=None):
    """Snap an extent to nearby 100m multiples, one variant per L."""
    raw = d_end - d_start
    out = []
    base = int(raw // 100) * 100
    for L in (base, base + 100):
        if L < 100 or abs(L - raw) > 75:
            continue
        if t_override is not None:
            t, t0, t1 = t_override
        else:
            if at_end:                  # watch stop pins the rep end
                win = (d_end - L, d_end)
            else:                       # center the snapped window
                mid = (d_start + d_end) / 2
                win = (mid - L / 2, mid + L / 2)
            a = max(win[0], seg.d0)
            b = min(a + L, seg.d1)
            t, t0, t1 = seg.time_at(b) - seg.time_at(a), seg.time_at(a), seg.time_at(b)
        out.append({'raw': raw, 'L': L, 't': t, 't0': t0, 't1': t1,
                    'tier': tier_penalty(L), 'is_alt': is_alt})
    return out


def _frozen_distance(seg, a, b):
    """Distance swallowed by GPS freezes inside [a, b]: spans where the
    per-second distance stream stalls (<2 m/s for >=8s) while the runner is
    mid-rep. Invisible to lap ratios (the median hides one short lap)."""
    pts = [(t, d) for t, d, _ in seg.pts if a <= d <= b]
    if len(pts) < 10:
        return 0.0
    speeds = [(d1 - d0) / (t1 - t0) for (t0, d0), (t1, d1) in zip(pts, pts[1:])
              if t1 > t0]
    moving = [v for v in speeds if v > 2.0]
    if not moving:
        return 0.0
    v_med = statistics.median(moving)
    swallowed, run_t, run_d = 0.0, 0.0, 0.0
    for (t0, d0), (t1, d1) in zip(pts, pts[1:]):
        dt = t1 - t0
        if dt <= 0 or dt > 10:
            continue
        if (d1 - d0) / dt < 2.0:
            run_t += dt
            run_d += d1 - d0
        else:
            if run_t >= 8:
                swallowed += run_t * v_med - run_d
            run_t = run_d = 0.0
    if run_t >= 8:
        swallowed += run_t * v_med - run_d
    return swallowed


def build_reps(blocks, inflation=None):
    reps = []
    for seg, d_start, d_end, at_start, at_end, alts in blocks:
        cands = _snap_variants(seg, d_start, d_end, at_end, False)
        if not cands:
            continue
        for alt in alts:
            for ac in _snap_variants(seg, alt[0], alt[1], at_end, True):
                if ac['L'] not in [c['L'] for c in cands]:
                    cands.append(ac)
        # GPS-freeze restoration: time is always true; only distance was stolen
        swallowed = _frozen_distance(seg, d_start, d_end)
        if swallowed >= 60:
            t_full = (seg.time_at(d_end) - seg.time_at(d_start),
                      seg.time_at(d_start), seg.time_at(d_end))
            for ac in _snap_variants(seg, d_start, d_end + swallowed, at_end,
                                     True, t_override=t_full):
                if ac['L'] not in [c['L'] for c in cands]:
                    cands.append(ac)
        # lap-ratio rescale (inflation >=412 or deflation <=390 m/lap)
        ratio = None
        if inflation:
            t0 = seg.ts[0]
            near = min(inflation, key=lambda k: abs(k - t0))
            if abs(near - t0) <= 5:
                ratio = inflation[near]
        if ratio and (ratio >= 412 or ratio <= 390):
            raw2 = (d_end - d_start) * (400.0 / ratio)
            t_full = (seg.time_at(d_end) - seg.time_at(d_start),
                      seg.time_at(d_start), seg.time_at(d_end))
            for ac in _snap_variants(seg, d_start, d_start + raw2, at_end,
                                     True, t_override=t_full):
                if ac['L'] not in [c['L'] for c in cands]:
                    cands.append(ac)
        reps.append({'seg': seg, 'cands': cands, 'first': at_start,
                     'last': at_end, **cands[0]})
    reps.sort(key=lambda r: r['t0'])
    for r in reps:
        # rolling-stop fragments (Max accelerating into the watch stop)
        r['optional'] = (r['last'] and not r['first']) or r['raw'] < 150
    return reps


# ---------- reconciliation ----------

def _cand_cost(c):
    return ((ALT_W if c['is_alt'] else 0.0) + c['tier']
            + (SALVAGE_W if c.get('salvage') else 0.0))


def _seg_options(rs, segments_by_id, sid, cf_allowed, target):
    """All ways to account one segment: block-candidate combos x optional
    inclusion, plus (fartlek days) the whole-segment candidate, which is
    mutually exclusive with interior blocks."""
    definite = [r for r in rs if not r['optional']]
    optional = [r for r in rs if r['optional']]
    opts = {}

    def add(sumL, cost, sig, picks):
        if sumL > target:
            return
        # coverage bias: splitting a segment into >1 rep costs a little.
        cost += COVERAGE_EXTRA * max(0, len(picks) - 1)
        key = (sumL, sig)
        if key not in opts or cost < opts[key][0]:
            opts[key] = (cost, picks)

    for pick in (product(*[r['cands'] for r in definite]) if definite else [()]):
        base_L = sum(c['L'] for c in pick)
        base_cost = sum(_cand_cost(c) for c in pick)
        base_picks = tuple(zip(definite, pick))
        for n in range(len(optional) + 1):
            for cmb in combinations(optional, n):
                sumL = base_L + sum(r['L'] for r in cmb)
                cost = base_cost + sum(OPT_W + _cand_cost(r['cands'][0])
                                       for r in cmb)
                sig = tuple(sorted([c['L'] for c in pick]
                                   + [r['L'] for r in cmb]))
                add(sumL, cost, sig,
                    base_picks + tuple((r, r['cands'][0]) for r in cmb))

    if cf_allowed and sid in segments_by_id:
        seg = segments_by_id[sid]
        is_single = len(segments_by_id) == 1   # entire workout = one chunk
        span = seg.d1 - seg.d0
        base = int(span // 100) * 100
        for L in (base, base + 100):
            if L >= 100 and abs(L - span) <= 75:
                cand = {'raw': span, 'L': L,
                        't': seg.time_at(seg.d1) - seg.time_at(seg.d0),
                        't0': seg.time_at(seg.d0), 't1': seg.time_at(seg.d1),
                        'tier': tier_penalty(L), 'is_alt': True,
                        'cf': is_single}
                stub = {'seg': seg, 'cands': [cand], 'optional': False,
                        'first': True, 'last': True, **cand}
                # a blend costs a hair more than an evidence alternative, so
                # a full decomposition wins an otherwise-equal tie
                add(L, ALT_W + 0.2 + cand['tier'], (L,), ((stub, cand),))
    if not definite:
        add(0, 0.0, (), ())      # segment contributes nothing
    return [(k[0], v[0], k[1], v[1]) for k, v in opts.items()]


def reconcile(reps, logged, segments=None, cf_allowed=False):
    """Pick candidates so the day total EXACTLY equals the logged quality
    distance (snapped to 100m). Returns (chosen, status); chosen is None
    when no combination reaches the target."""
    target = round(logged / 100) * 100
    by_seg = {}
    for r in reps:
        by_seg.setdefault(id(r['seg']), []).append(r)
    segments_by_id = {id(s): s for s in (segments or [])}
    seg_ids = list(by_seg)
    if cf_allowed:                       # rep-less segments still offer cf
        seg_ids += [sid for sid in segments_by_id if sid not in by_seg]

    def prune(entries):
        # future additions are independent of the past, so only min-cost
        # prefixes can win — but sig ties must survive for the ambiguity check
        if not entries:
            return entries
        lo = min(e[0] for e in entries)
        kept, sigs = [], set()
        for e in sorted(entries, key=lambda e: e[0]):
            if e[0] > lo + 1e-6:
                break
            if e[1] not in sigs:
                kept.append(e)
                sigs.add(e[1])
            if len(kept) >= 8:
                break
        return kept

    states = {0: [(0.0, (), ())]}
    for sid in seg_ids:
        options = _seg_options(by_seg.get(sid, []), segments_by_id, sid,
                               cf_allowed, target)
        new = {}
        for tot, entries in states.items():
            for oL, ocost, osig, opicks in options:
                t2 = tot + oL
                if t2 > target:
                    continue
                for cost, sig, picks in entries:
                    new.setdefault(t2, []).append(
                        (cost + ocost, tuple(sorted(sig + osig)),
                         picks + opicks))
        states = {t: prune(v) for t, v in new.items()}
        if not states:
            break

    hits = states.get(target)
    if not hits:
        min_def = sum(min(c['L'] for c in r['cands'])
                      for r in reps if not r['optional'])
        return None, ('definite-exceeds' if min_def > target else 'no-subset')
    lo = min(h[0] for h in hits)
    best = [h for h in hits if h[0] <= lo + 1e-6]
    chosen = sorted(({**r, **c} for r, c in best[0][2]), key=lambda x: x['t0'])
    if len({h[1] for h in best}) > 1:
        return chosen, 'ambiguous'
    if any(c['tier'] > 1.0 for _, c in best[0][2]):
        return chosen, 'ambiguous-tier3'   # uses a tier-3 piece: park for review
    return chosen, 'exact'


def annotate(reps, pts):
    """Per-rep HR and the rest (standing vs jog) until the next rep."""
    ts = [p[0] for p in pts]
    for i, r in enumerate(reps):
        hs = [h for t, _, h in pts if r['t0'] <= t <= r['t1'] and h]
        r['avg_hr'] = round(statistics.mean(hs)) if hs else None
        r['max_hr'] = max(hs) if hs else None
        if i + 1 < len(reps):
            a = bisect.bisect_left(ts, r['t1'])
            b = bisect.bisect_right(ts, reps[i+1]['t0'])
            moving = sum(min(ts[k+1] - ts[k], 10) for k in range(a, b - 1))
            total = reps[i+1]['t0'] - r['t1']
            r['rest_jog_s'] = round(min(moving, total))
            r['rest_stand_s'] = round(max(total - moving, 0))
        else:
            r['rest_jog_s'] = r['rest_stand_s'] = None


# ---------- day pipeline ----------

def cs_threshold_fn(cs_path):
    cs = pd.read_csv(cs_path, parse_dates=['date']).sort_values('date')
    days = cs['date'].map(pd.Timestamp.toordinal).values
    paces = cs['cs_pace_med'].values * 60.0

    def cutoff(date):
        return float(np.interp(pd.Timestamp(date).toordinal(), days, paces)) + 20.0
    return cutoff


def _salvage_core(seg, cs_cut):
    """Longest contiguous run of quality windows (<= cs_cut) in a segment whose
    block the cs_cut filter dropped — the rep core, surfaced so a
    contamination-inflated block AVERAGE (a GPS cold-start jog, a freeze/jump
    glitch) doesn't silently lose a real rep. Returns (a, b) or None."""
    runs, s, e = [], None, None
    x = seg.d0
    while x + FINE_W <= seg.d1 + 0.01:
        p = seg.pace(x, x + FINE_W)
        if p is not None and p <= cs_cut:
            s = x if s is None else s
            e = x + FINE_W
        elif s is not None:
            runs.append((s, e))
            s = None
        x += CHUNK
    if s is not None:
        runs.append((s, e))
    if not runs:
        return None
    best = max(runs, key=lambda r: r[1] - r[0])
    return best if best[1] - best[0] >= MIN_SALVAGE else None


def extract_day(recs, cs_cut, logged, cf_allowed):
    """Run the full pipeline for one day's rich records."""
    segments = [s for rec in recs for s in moving_segments(rec)]
    all_pts = sorted((p for s in segments for p in s.pts))
    blocks = coarse_blocks(segments)
    blocks = refine_blocks(blocks, cs_cut)
    reps = build_reps(blocks, inflation=seg_lap_ratios(recs))
    reps = [r for r in reps if r['t'] / (r['L'] / MILE) <= cs_cut]
    if logged is not None:
        # Salvage: a segment whose only block was contamination-filtered still
        # offers its quality core as an OPTIONAL candidate, so the hand total
        # can place it. The coverage bias (see _seg_options) then prefers using
        # it over splitting another segment / leaving it unaccounted — without
        # forcing it (a genuine all-jog segment stays dropped).
        used = {id(r['seg']) for r in reps}
        for seg in segments:
            if id(seg) in used:
                continue
            sv = _salvage_core(seg, cs_cut)
            if not sv:
                continue
            cands = _snap_variants(seg, sv[0], sv[1], False, False)
            if cands:
                for c in cands:
                    c['salvage'] = True
                reps.append({'seg': seg, 'cands': cands, 'first': True,
                             'last': True, 'optional': True, **cands[0]})
        reps.sort(key=lambda r: r['t0'])
    if not reps:
        return [], 'no-blocks'
    if logged is not None:
        chosen, status = reconcile(reps, logged, segments=segments,
                                   cf_allowed=cf_allowed)
        if chosen is None:
            return [], status
    else:
        chosen = [{**r, **min(r['cands'], key=_cand_cost)}
                  for r in reps if not r['optional']]
        status = 'watch-only'
    annotate(chosen, all_pts)
    return chosen, status


WORKOUT_MEASURED_COLS = ['date', 'status', 'logged_qd_m', 'rep_idx', 'kind',
                         'dist_m', 'time_s', 'pace_sec_per_mi', 'rest_stand_s',
                         'rest_jog_s', 'avg_hr', 'max_hr', 'label_ids']


def _extract_day_rows(date, recs, logged, cf, hill, cutoff, label_ids,
                      watch_only):
    """Rows for one analyzed day (status row rep_idx=0 + one row per rep).
    Returns (rows, slim_skipped). `label_ids` is the day's labelId set, stamped
    on the status row as the presence key for incremental reuse."""
    if not watch_only and date in DISQUALIFIED:
        return [{'date': date, 'status': f'disqualified: {DISQUALIFIED[date]}',
                 'logged_qd_m': logged, 'rep_idx': 0, 'label_ids': label_ids}], 0
    rich = [r for r in recs if 'freq' in r]
    if not rich:
        return [{'date': date, 'status': 'no-rich-data', 'logged_qd_m': logged,
                 'rep_idx': 0, 'label_ids': label_ids}], 1
    if hill:
        chosen, status = extract_hill_day(rich, *hill)
        kind = 'loop'
    else:
        chosen, status = extract_day(rich, cutoff(date), logged, cf)
        kind = None
    rows = [{'date': date, 'status': status, 'logged_qd_m': logged,
             'rep_idx': 0, 'label_ids': label_ids}]
    for i, r in enumerate(chosen):
        rows.append({
            'date': date, 'status': status, 'logged_qd_m': logged,
            'rep_idx': i + 1,
            'kind': kind or ('cf' if r.get('cf') else 'rep'),
            'dist_m': r['L'], 'time_s': round(r['t'], 1),
            'pace_sec_per_mi': round(r['t'] / (r['L'] / MILE), 1),
            'rest_stand_s': r['rest_stand_s'],
            'rest_jog_s': r.get('rest_jog_s'),
            'avg_hr': r['avg_hr'], 'max_hr': r['max_hr'],
        })
    return rows, 0


def _day_plan(by_date, daily, watch_only, races_path):
    """{date: {logged, cf, hill, ids}} — which days to analyze and, per day, the
    labelId set to extract from. `by_date` maps date -> [(labelId, sport_type)].
    Pure metadata: no per-second parse. Mirrors the original plan exactly."""
    plan = {}
    if watch_only:
        race_dates = set()
        if races_path and Path(races_path).exists():
            race_dates = set(pd.read_csv(races_path)['date'].astype(str))
        for date, acts in by_date.items():
            track_ids = [l for l, st in acts if st == M.SPORT_TRACK_RUN]
            if track_ids and date not in race_dates:
                plan[date] = {'logged': None, 'cf': False, 'hill': None,
                              'ids': track_ids}
        return plan
    hand = daily.copy()
    hand['date'] = pd.to_datetime(hand['date']).dt.date.astype(str)
    hand = hand.set_index('date')
    for date, acts in by_date.items():
        h = hand.loc[date] if date in hand.index else None
        run_type = h['run_type'] if h is not None else None
        qd = (float(h['quality_distance_m'])
              if h is not None and run_type in QUALITY_TYPES
              and pd.notna(h['quality_distance_m']) else None)
        cf = h is not None and 'f@' in str(h['workout_raw'])
        track_ids = [l for l, st in acts if st == M.SPORT_TRACK_RUN]
        all_ids = [l for l, st in acts]
        if track_ids:
            if run_type == 'race':
                continue
            plan[date] = {'logged': qd, 'cf': cf, 'hill': None, 'ids': track_ids}
        elif run_type in QUALITY_TYPES:
            plan[date] = {'logged': qd, 'cf': cf, 'hill': None, 'ids': all_ids}
        elif run_type == 'hill_cont':
            _min, nreps, loop = parse_hc(h)
            loop_m = hc_loop_distance(loop) if loop else None
            if nreps and loop_m:
                plan[date] = {'logged': float(nreps) * float(loop_m), 'cf': False,
                              'hill': (int(nreps), float(loop_m)), 'ids': all_ids}
    return plan


def build_workout_measured(daily, details_dir, cs_path, *, watch_only=False,
                           races_path=None, activities_path=None,
                           out_path=None, full_regen=False):
    """workout_measured DataFrame, built INCREMENTALLY when the per-activity
    index (watch_activities.csv) is present: candidate days come from the index
    (no parse), and a day is re-extracted only when its labelId set changed (vs.
    the cached status row) or `full_regen` is set (the CS-refit safety valve).
    Without the index it falls back to the full-parse behavior."""
    cutoff = cs_threshold_fn(cs_path)
    details_dir = Path(details_dir)
    indexed = bool(activities_path and Path(activities_path).exists())

    if indexed:
        idx = pd.read_csv(activities_path, dtype={'labelId': str, 'date': str})
        by_date = {}
        for _, r in idx.iterrows():
            by_date.setdefault(r['date'], []).append(
                (r['labelId'], int(r['sport_type'])))

        def load_recs(meta):
            return [json.loads((details_dir / f'{l}.json').read_text())
                    for l in meta['ids']]
    else:
        # Fallback: parse the cache to build the per-day index in memory.
        by_date, rec_by_id = {}, {}
        for p in sorted(details_dir.glob('*.json')):
            if p.stem in M.EXCLUDED_LABEL_IDS:
                continue
            rec = json.loads(p.read_text())
            if (rec.get('summary') or {}).get('sportType') is None:
                continue
            act = Activity(rec)
            if act.sport_type not in M.RUN_SPORTS:
                continue
            by_date.setdefault(act.local_date.isoformat(), []).append(
                (p.stem, act.sport_type))
            rec_by_id[p.stem] = rec

        def load_recs(meta):
            return [rec_by_id[l] for l in meta['ids']]

    plan = _day_plan(by_date, daily, watch_only, races_path)

    # Cached rows keyed by date, with the presence key from the status row.
    existing = {}
    if indexed and out_path and Path(out_path).exists():
        ex = pd.read_csv(out_path)
        if 'label_ids' in ex.columns:
            for date, g in ex.groupby('date'):
                sr = g[g['rep_idx'] == 0]
                key = (str(sr['label_ids'].iloc[0])
                       if len(sr) and pd.notna(sr['label_ids'].iloc[0]) else '')
                existing[str(date)] = (key, g.to_dict('records'))

    rows, skipped, reextracted = [], 0, 0
    for date in sorted(plan):
        meta = plan[date]
        key = ' '.join(sorted(meta['ids']))
        cached = existing.get(date)
        if not full_regen and cached is not None and cached[0] == key:
            rows.extend(cached[1])             # unchanged day — reuse, no parse
            continue
        reextracted += 1
        drows, sk = _extract_day_rows(date, load_recs(meta), meta['logged'],
                                      meta['cf'], meta['hill'], cutoff, key,
                                      watch_only)
        rows.extend(drows)
        skipped += sk
    if indexed:
        print(f'[reps] {len(plan)} workout days, re-extracted {reextracted} '
              f'(reused {len(plan) - reextracted})')
    return pd.DataFrame(rows, columns=WORKOUT_MEASURED_COLS), skipped


def main():
    p = argparse.ArgumentParser(description=(__doc__ or '').split('\n\n')[0])
    p.add_argument('--details-dir', type=Path,
                   default=DATA_DIR / 'details',
                   help='Rich Coros detail cache (see backfill_rich_details).')
    p.add_argument('--watch-only', action='store_true',
                   help='No hand log to reconcile against (watch-import '
                        'profiles): emit definite blocks only.')
    p.add_argument('--activities', type=Path,
                   default=DATA_DIR / 'watch_activities.csv',
                   help='Per-activity index (watch_daily). Enables incremental '
                        'extraction; absent -> full parse.')
    p.add_argument('--full-regen', action='store_true',
                   help='Re-extract every day, ignoring the presence cache '
                        '(CS-refit safety valve).')
    p.add_argument('--out', type=Path, default=DATA_DIR / 'workout_measured.csv')
    args = p.parse_args()

    daily = None
    if not args.watch_only:
        daily = pd.read_csv(DATA_DIR / 'daily.csv')
    df, skipped = build_workout_measured(
        daily, args.details_dir, DATA_DIR / 'bayes_cs_summary.csv',
        watch_only=args.watch_only, races_path=DATA_DIR / 'races.csv',
        activities_path=args.activities, out_path=args.out,
        full_regen=args.full_regen)
    df.to_csv(args.out, index=False)

    days = df[df['rep_idx'] == 0]
    print(f'Wrote {args.out}  ({len(days)} days, '
          f'{(df["rep_idx"] > 0).sum()} reps)')
    print(days['status'].value_counts().to_string())
    if skipped:
        print(f'NOTE: {skipped} days skipped for lack of rich data — run '
              f'scripts/backfill_rich_details.py')


if __name__ == '__main__':
    main()
