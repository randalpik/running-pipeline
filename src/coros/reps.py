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
    rep-length prior: tier1 {200,300,400,500,800,1600} free, other ×200
    cheap, other ×100 expensive.
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

TIER1 = {200, 300, 400, 500, 800, 1600}
ALT_W, OPT_W = 2.0, 0.5

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
    # day-median jog length pins rep extents — but only on noise-mangled
    # blocks (window-pace IQR > 60 s/mi). Ungated, it overthrows verified
    # structures with cheap bogus alternatives.
    last_in_seg = {}
    for blk in blocks:
        sid = id(blk[0])
        if sid not in last_in_seg or blk[1] > last_in_seg[sid][1]:
            last_in_seg[sid] = blk
    tails = [blk[0].d1 - blk[2] for blk in last_in_seg.values()
             if blk[0].d1 - blk[2] >= 150]
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
            if len(wins) >= 4:
                qs = statistics.quantiles(wins, n=4)
                if qs[2] - qs[0] > 60:
                    blk[5].append((blk[1], alt_e))
    return blocks


# ---------- candidates (snap variants + distance-scale corrections) ----------

def tier_penalty(L):
    """Max's rep-length prior: tier1 free, other x200 cheap, other x100 dear."""
    if L in TIER1:
        return 0.0
    if L % 200 == 0:
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
    return (ALT_W if c['is_alt'] else 0.0) + c['tier']


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


def extract_day(recs, cs_cut, logged, cf_allowed):
    """Run the full pipeline for one day's rich records."""
    segments = [s for rec in recs for s in moving_segments(rec)]
    all_pts = sorted((p for s in segments for p in s.pts))
    blocks = coarse_blocks(segments)
    blocks = refine_blocks(blocks, cs_cut)
    reps = build_reps(blocks, inflation=seg_lap_ratios(recs))
    reps = [r for r in reps if r['t'] / (r['L'] / MILE) <= cs_cut]
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


def build_workout_measured(daily, details_dir, cs_path, *, watch_only=False,
                           races_path=None):
    """Returns the workout_measured DataFrame: one status row (rep_idx=0)
    per analyzed day plus one row per reconstructed rep."""
    inv = {}     # date -> [(label, sport_type, rec)]
    skipped_slim = 0
    for p in sorted(Path(details_dir).glob('*.json')):
        rec = json.loads(p.read_text())
        if (rec.get('summary') or {}).get('sportType') is None:
            continue
        act = Activity(rec)
        if act.sport_type not in M.RUN_SPORTS:
            continue
        inv.setdefault(act.local_date.isoformat(), []).append(
            (p.stem, act.sport_type, rec))

    plan = []    # (date, [recs], logged_qd, cf_allowed)
    if watch_only:
        race_dates = set()
        if races_path and Path(races_path).exists():
            race_dates = set(pd.read_csv(races_path)['date'].astype(str))
        for date, acts in inv.items():
            tracks = [r for l, st, r in acts if st == M.SPORT_TRACK_RUN]
            if tracks and date not in race_dates:
                plan.append((date, tracks, None, False))
    else:
        hand = daily.copy()
        hand['date'] = pd.to_datetime(hand['date']).dt.date.astype(str)
        hand = hand.set_index('date')
        for date, acts in inv.items():
            h = hand.loc[date] if date in hand.index else None
            run_type = h['run_type'] if h is not None else None
            qd = (float(h['quality_distance_m'])
                  if h is not None and run_type in QUALITY_TYPES
                  and pd.notna(h['quality_distance_m']) else None)
            cf = h is not None and 'f@' in str(h['workout_raw'])
            tracks = [r for l, st, r in acts if st == M.SPORT_TRACK_RUN]
            if tracks:
                if run_type == 'race':
                    continue
                plan.append((date, tracks, qd, cf))
            elif run_type in QUALITY_TYPES:
                plan.append((date, [r for l, st, r in acts], qd, cf))

    cutoff = cs_threshold_fn(cs_path)
    rows = []
    for date, recs, logged, cf in sorted(plan):
        if not watch_only and date in DISQUALIFIED:
            rows.append({'date': date,
                         'status': f'disqualified: {DISQUALIFIED[date]}',
                         'logged_qd_m': logged, 'rep_idx': 0})
            continue
        rich = [r for r in recs if 'freq' in r]
        if not rich:
            rows.append({'date': date, 'status': 'no-rich-data',
                         'logged_qd_m': logged, 'rep_idx': 0})
            skipped_slim += 1
            continue
        chosen, status = extract_day(rich, cutoff(date), logged, cf)
        rows.append({'date': date, 'status': status, 'logged_qd_m': logged,
                     'rep_idx': 0})
        for i, r in enumerate(chosen):
            rows.append({
                'date': date, 'status': status, 'logged_qd_m': logged,
                'rep_idx': i + 1, 'kind': 'cf' if r.get('cf') else 'rep',
                'dist_m': r['L'], 'time_s': round(r['t'], 1),
                'pace_sec_per_mi': round(r['t'] / (r['L'] / MILE), 1),
                'rest_stand_s': r['rest_stand_s'],
                'rest_jog_s': r['rest_jog_s'],
                'avg_hr': r['avg_hr'], 'max_hr': r['max_hr'],
            })
    cols = ['date', 'status', 'logged_qd_m', 'rep_idx', 'kind', 'dist_m',
            'time_s', 'pace_sec_per_mi', 'rest_stand_s', 'rest_jog_s',
            'avg_hr', 'max_hr']
    return pd.DataFrame(rows, columns=cols), skipped_slim


def main():
    p = argparse.ArgumentParser(description=(__doc__ or '').split('\n\n')[0])
    p.add_argument('--details-dir', type=Path,
                   default=DATA_DIR / 'details',
                   help='Rich Coros detail cache (see backfill_rich_details).')
    p.add_argument('--watch-only', action='store_true',
                   help='No hand log to reconcile against (watch-import '
                        'profiles): emit definite blocks only.')
    p.add_argument('--out', type=Path, default=DATA_DIR / 'workout_measured.csv')
    args = p.parse_args()

    daily = None
    if not args.watch_only:
        daily = pd.read_csv(DATA_DIR / 'daily.csv')
    df, skipped = build_workout_measured(
        daily, args.details_dir, DATA_DIR / 'bayes_cs_summary.csv',
        watch_only=args.watch_only, races_path=DATA_DIR / 'races.csv')
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
