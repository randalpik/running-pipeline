"""
parse_workouts.py — Decompose quality workouts from daily.csv into
                    (date, type, rep_dist, rep_count, pace_per_mile, rest_per_mile).

Output: workout_decomposed.csv (one row per resolved quality workout) lands
in data/ and is consumed by the training-quality plot.

The audit trail of rows that didn't decompose (workout_pruned.csv) is a
diagnostic-only artifact — useful for human inspection but not consumed
anywhere in the pipeline. Pass ``--diagnostics`` to also write
workout_pruned.csv into output/debug/.

Reads daily.csv from data/.
"""
import argparse
import math
import sys
import numpy as np
import pandas as pd
import re
from pathlib import Path
from datetime import date

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.shared.paths import DATA_DIR, DEBUG_DIR
from src.shared.workouts import RECON_TAU_S, g_anaerobic

MILE_M = 1609.344

# ---------- regex for explicit Nx shapes ----------
# Variant A (pre-March 2016): Nx<dist>[tirf]/<rest>s@<pace>  -- pace per-rep, rest per-rep
RX_A = re.compile(r'(\d+)x(\d+)([tirf])/(\d+)s@([\d:.]+)')
# Variant B (post-March 2016):  Nx<dist>[tirf]@<pace>          -- pace per-mile
RX_B = re.compile(r'(\d+)x(\d+)([tirf])@([\d:.]+)')
# Variant C (2016 rep syntax):  Nx<dist>rep@<pace>              -- pace per-mile
RX_C = re.compile(r'(\d+)x(\d+)rep@([\d:.]+)')
# Per-mile rest annotation in continuous form
RX_REST = re.compile(r'\(?([\d:]+)\s*rest/mi\)?')


def parse_time_to_seconds(s):
    """Parse 'M:SS' or 'SS' into float seconds."""
    if s is None or s == '':
        return None
    parts = str(s).split(':')
    try:
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        return float(parts[0])
    except (ValueError, TypeError):
        return None


def match_nx(raw):
    """Try variants A/B/C. Return dict or None."""
    m = RX_A.search(raw)
    if m:
        return {
            'count': int(m.group(1)), 'rep_dist': int(m.group(2)),
            'letter': m.group(3),
            'rest_per_rep_s': int(m.group(4)),
            'pace_raw_s': parse_time_to_seconds(m.group(5)),
            'pace_is_per_rep': True,
        }
    m = RX_B.search(raw)
    if m:
        return {
            'count': int(m.group(1)), 'rep_dist': int(m.group(2)),
            'letter': m.group(3),
            'rest_per_rep_s': None,
            'pace_raw_s': parse_time_to_seconds(m.group(4)),
            'pace_is_per_rep': False,
        }
    m = RX_C.search(raw)
    if m:
        return {
            'count': int(m.group(1)), 'rep_dist': int(m.group(2)),
            'letter': 'r',
            'rest_per_rep_s': None,
            'pace_raw_s': parse_time_to_seconds(m.group(3)),
            'pace_is_per_rep': False,
        }
    return None


def default_rest_per_mile(final_type, rep_dist):
    """Defaults applied only when no explicit rest is given."""
    if final_type == 'tempo':
        return 60       # 1:00/mi
    if final_type == 'interval':
        return 140 if rep_dist == 800 else 180   # 2:20/mi for 800m, 3:00/mi otherwise
    if final_type == 'rep':
        return 420      # 7:00/mi  (best fit per data analysis; was 8:00 in v6)
    if final_type == 'continuous_fartlek':
        return 0
    return None


def reclassify_fartlek(total_m, has_zero_rest):
    """
    Returns (final_type, rep_dist, rep_count) for a fartlek given total distance.
    Continuous fartlek (7000-10000m, 0 rest) handled by the caller before this.
    Returns None for unresolved.
    """
    if total_m is None or pd.isna(total_m):
        return None
    if total_m < 4000:
        return ('rep', 400, round(total_m / 400))
    if total_m <= 6999:
        return ('interval', 800, round(total_m / 800))
    if total_m <= 10000:
        return ('interval', 1600, round(total_m / 1600))
    return None


def _structure_label(dists):
    """Run-length-encoded headline for measured reps, in rep order:
    [1600, 800, 800, 400, 400, 400, 400, 200] -> '1600 + 2×800 + 4×400 + 200m'.
    Uniform days keep the familiar 'N × Dm' form."""
    groups = []
    for d in dists:
        d = int(d)
        if groups and groups[-1][0] == d:
            groups[-1][1] += 1
        else:
            groups.append([d, 1])
    if len(groups) == 1:
        d, n = groups[0]
        return f'{d}m' if n == 1 else f'{n} × {d}m'
    return ' + '.join(f'{n}×{d}' if n > 1 else f'{d}' for d, n in groups) + 'm'


def _connected_core(dists, times, rests):
    """Connected-fatigue (D_eff, t_eff) from per-rep arrays, with the
    short-distance anaerobic correction applied per rep.

    Each rep extends a running "connected" effort by its distance; the rest
    AFTER it dissipates the accumulated connection by exp(-rest_s/RECON_TAU_S).
    Rest is ACTUAL seconds — reconstitution is a wall-clock process, NOT
    per-mile-normalized (per-mile would misweight ladders). D_eff is the
    deepest connected distance reached, bounded in [longest rep, total] with
    no floor (no rest -> total; full recovery -> longest rep).

    The anaerobic correction g_anaerobic(d) slows each rep's pace before the
    mean rep speed is taken, so short reps (sub-~800m) — whose pace the CP
    hyperbola over-credits as CS — contribute a sustainable-equivalent speed.
    Distance-only, so it preserves gain and acts per-segment on ladders. No
    CS enters D_eff, so the implied CS the projection reads off stays an
    independent fitness signal — see [[project-workout-enrichment]].

    ``dists``/``times``: per-rep arrays. ``rests``: rest-after seconds per rep
    (the final entry, if present, is unused — nothing accumulates after it).
    """
    dists = np.asarray(dists, float)
    times = np.asarray(times, float)
    rests = np.asarray(rests, float)
    paces = times * MILE_M / dists                       # s/mi
    times_corr = (paces + g_anaerobic(dists)) * dists / MILE_M
    conn = d_eff = 0.0
    for i in range(len(dists)):
        conn += dists[i]
        if conn > d_eff:
            d_eff = conn
        if i < len(rests):
            conn *= math.exp(-rests[i] / RECON_TAU_S)
    v = dists.sum() / times_corr.sum()                   # corrected mean speed
    return d_eff, d_eff / v


def _measured_d_eff(day):
    """(D_eff, t_eff) for a watch-measured day — see _connected_core. Rest is
    the recorded standing+jog seconds per interval (last rep has none)."""
    day = day.sort_values('rep_idx')
    rest = (day['rest_stand_s'].fillna(0)
            + day['rest_jog_s'].fillna(0)).to_numpy(float)
    return _connected_core(day['dist_m'].to_numpy(float),
                           day['time_s'].to_numpy(float), rest)


# ---------- rest-source policy (per Max's logging-behaviour audit) ----------
# Logged rest is trustworthy in two regimes and lazy/catch-all otherwise (a
# 2026-06 scatter audit of logged rest/mi over time):
#   - tempo / fartlek logged rest is real (0:00 continuous blocks; the ladder's
#     enforced 2:20; deliberate short tempo rest) — always trusted;
#   - interval / rep logged rest is accurate PRE-2020 but lazy/legacy after
#     (Max kept logging the old ballpark while actually resting more). So 2020+
#     interval/rep rest is taken from this profile's OWN watch-measured median
#     (by rep distance); pre-2020 uses the log, or the pre-2020 logged median
#     for that type when a day has no annotation at all.
# This is data-derived per profile (recomputed each run), so it sharpens as the
# watch corpus grows. See [[project-workout-enrichment]].
WATCH_ERA_START_YEAR = 2020
LADDER_4800 = [1600, 800, 800, 400, 400, 400, 400]   # 4800f @ 2:20: exact, hardcoded


def _logged_rest_per_mile(raw):
    """Logged rest/mile from a workout string, or None if not annotated.
    Per-rep `/Ns@` (early format) is converted to per-mile; `(M:SS rest/mi)`
    is taken directly."""
    a = RX_A.search(raw)
    if a:
        return int(a.group(4)) * (MILE_M / int(a.group(2)))
    m = RX_REST.search(raw)
    return parse_time_to_seconds(m.group(1)) if m else None


def build_rest_model(measured, daily):
    """Per-profile rest estimators for days without trustworthy logged rest:
    `watch` = median measured rest (s/rep) by 100m-binned rep distance (2020+
    interval/rep); `early` = pre-2020 logged median rest/mile by type."""
    watch = {}
    if measured is not None and len(measured):
        mr = measured[(measured['rep_idx'] > 0)
                      & measured['status'].isin(['exact', 'watch-only'])
                      & (measured['kind'] != 'cf')].copy()
        mr = mr[mr['rest_stand_s'].notna() | mr['rest_jog_s'].notna()]
        if len(mr):
            mr['rest_s'] = mr['rest_stand_s'].fillna(0) + mr['rest_jog_s'].fillna(0)
            mr['rd'] = (mr['dist_m'] / 100).round() * 100
            watch = {int(rd): float(g['rest_s'].median())
                     for rd, g in mr.groupby('rd') if len(g) >= 5}
    early = {}
    if daily is not None:
        d = daily.copy()
        d['date'] = pd.to_datetime(d['date'])
        for typ in ('interval', 'rep'):
            sub = d[(d['run_type'] == typ) & (d['date'].dt.year < WATCH_ERA_START_YEAR)]
            vals = [_logged_rest_per_mile('' if pd.isna(r) else str(r))
                    for r in sub['workout_raw']]
            vals = [v for v in vals if v is not None and v > 0]
            if vals:
                early[typ] = float(np.median(vals))
    return {'watch': watch, 'early': early}


def _watch_rest_s(rep_dist, watch):
    if not watch:
        return None
    return watch.get(int(rep_dist)) or watch[min(watch, key=lambda k: abs(k - rep_dist))]


def effective_rest_per_mile(rtype, rep_dist, year, logged_rpm, rest_model):
    """Return (rest_per_mile, trusted). `trusted` gates the connected
    projection / TQ inclusion — an untrusted estimate falls back to the legacy
    formula and (for reps) stays excluded."""
    if rtype in ('interval', 'rep') and rest_model is not None:
        if year >= WATCH_ERA_START_YEAR:
            s = _watch_rest_s(rep_dist, rest_model['watch'])
            if s is not None:
                return s * MILE_M / rep_dist, True       # 2020+: watch median
        if logged_rpm is not None:
            return logged_rpm, True                       # pre-2020: trust the log
        early = rest_model['early'].get(rtype)
        if early is not None:
            return early, True                            # pre-2020, no log: early median
    if logged_rpm is not None:                            # tempo/fartlek: real
        return logged_rpm, True
    dflt = default_rest_per_mile(rtype, rep_dist)
    return (dflt if dflt is not None else 0.0), False


def measured_to_decomposed(measured, daily_df):
    """Convert watch-measured reps (workout_measured.csv, written by
    src/coros/reps.py) into decomposed-schema rows.

    Only days the rep-extraction layer trusts are converted:
      - 'exact'      : reconciled to the meter against the hand log;
      - 'watch-only' : no hand log exists (watch-import profiles) — a
        watch-derived Track Run on such a day IS the workout.
    Every other status (disqualified, ambiguous, no-subset, ...) falls back
    to the string parser. Days whose hand log doesn't claim a quality
    workout never appear in workout_measured.csv at all (reps.py only
    analyzes quality days), so a stray watch Track Run on a non-workout
    day is ignored completely — per Max, the hand log wins.

    Each enriched day becomes ONE decomposed row — the projection's CS+D'
    hyperbola needs whole-workout D_eff (per-rep-group rows with measured
    full-recovery rests push D_eff toward D' and the 5K-equivalent
    explodes). Per-rep detail stays in workout_measured.csv for display.

      rep_dist     : distance-weighted mean rep length (sum d_i^2 / sum d_i)
                     rounded to 100m — exact for uniform days, an effective
                     rep for ladders (kept for type inference and as a
                     fallback summary; display and projection use the
                     columns below);
      rep_count    : round(total / rep_dist), >= 1;
      pace_per_mile: time-weighted measured pace across all reps;
      rest_per_mile: MEASURED total rest (standing + jog) per rep-mile,
                     replacing the parser's defaults;
      structure    : run-length-encoded rep layout in rep order
                     ('1600 + 2×800 + 4×400 + 200m') — the hover headline;
      d_eff_m,
      t_eff_s      : whole-workout effective distance/time computed from
                     the per-rep structure and measured rests
                     (_measured_d_eff); project_workouts uses these instead
                     of the uniform-rep formula when present.

    Type honors the hand letter when explicit (t/i/r); `f@` days (and
    watch-only days) become continuous_fartlek when the day is a single cf
    chunk, else interval/rep by effective rep length.

    Returns (rows, enriched_dates).
    """
    daily_types = {}
    if daily_df is not None:
        d = daily_df.copy()
        d['date'] = pd.to_datetime(d['date']).dt.date
        daily_types = dict(zip(d['date'], d['run_type']))

    rows, enriched = [], set()
    reps = measured[(measured['rep_idx'] > 0)
                    & (measured['status'].isin(['exact', 'watch-only']))]
    for dt, day in reps.groupby('date'):
        dt = pd.to_datetime(dt).date()
        day = day.sort_values('rep_idx')
        run_type = daily_types.get(dt)
        total = day['dist_m'].sum()
        pace = day['time_s'].sum() / (total / MILE_M)

        if (day['kind'] == 'cf').all():
            rows.append({'date': dt, 'type': 'continuous_fartlek',
                         'rep_dist': int(total), 'rep_count': 1,
                         'pace_per_mile': pace, 'rest_per_mile': 0,
                         'structure': None, 'd_eff_m': float(total),
                         't_eff_s': day['time_s'].sum()})
            enriched.add(dt)
            continue

        rep_dist = (day['dist_m'] ** 2).sum() / total
        rep_dist = max(100, round(rep_dist / 100) * 100)
        rep_count = max(1, round(total / rep_dist))
        rest = (day['rest_stand_s'].fillna(0) + day['rest_jog_s'].fillna(0))
        has_rest = day['rest_stand_s'].notna() | day['rest_jog_s'].notna()
        rest_total = rest[has_rest].sum()
        # Normalize over the distance of reps that HAD a rest after them (the
        # last rep has none), so N-1 rests divide by N-1 reps' distance:
        # 4x1mi w/ 3:00 between reads 3:00/mi, not 2:15. Display + non-enriched
        # fallback only — the connected D_eff uses raw rest seconds directly.
        rest_denom_m = day.loc[has_rest, 'dist_m'].sum()
        rest_per_mile = rest_total / (rest_denom_m / MILE_M) if rest_denom_m else 0.0
        if run_type in ('tempo', 'interval', 'rep'):
            final_type = run_type
        else:                           # fartlek collision / watch-only
            final_type = 'interval' if rep_dist >= 800 else 'rep'
        d_eff, t_eff = _measured_d_eff(day)
        rows.append({'date': dt, 'type': final_type,
                     'rep_dist': int(rep_dist), 'rep_count': int(rep_count),
                     'pace_per_mile': pace, 'rest_per_mile': rest_per_mile,
                     'structure': _structure_label(day['dist_m']),
                     'd_eff_m': round(d_eff, 1), 't_eff_s': round(t_eff, 1)})
        enriched.add(dt)
    return rows, enriched


def decompose(daily_df, continuous_fartlek_only=False, rest_model=None):
    """
    Filter quality workouts, decompose each row.
    Returns (decomposed_df, pruned_df).

    continuous_fartlek_only: for watch-import profiles whose only quality
    coding is a single continuous fartlek (no rep structure is ever recorded),
    any no-rest fartlek >=4000m is classified continuous_fartlek rather than
    being reclassified into intervals by distance. Matches the importer's
    contract that it never emits interval/rep workouts.
    """
    # Filter on run_type, not quality_segment_type: pre-March 2016 tempos use
    # a per-rep "Nx800t/30s@2:53" format the parser can't extract qd/qp from,
    # so quality_segment_type is NaN even though run_type='tempo'. The Nx
    # regex below handles these correctly.
    qual = daily_df[daily_df['run_type'].isin(
        ['tempo', 'interval', 'rep', 'fartlek']
    )].copy()
    qual['date'] = pd.to_datetime(qual['date']).dt.date

    results = []
    pruned = []

    for _, row in qual.iterrows():
        dt = row['date']
        rtype = row['run_type']
        raw = '' if pd.isna(row['workout_raw']) else str(row['workout_raw'])
        qd = row['quality_distance_m']
        qp = row['quality_pace_sec_per_mi']

        # ---------- pruning rules (in order) ----------
        # (1) Hardcoded anomaly (defensive; current parser excludes from quality anyway)
        if dt == date(2016, 7, 11):
            pruned.append({'date': dt, 'type': rtype, 'reason': 'anomaly per Max', 'raw': raw})
            continue
        # (2) Tempo with @-pace > 600s/mi → @ holds total time, not pace
        if rtype == 'tempo' and not pd.isna(qp) and qp > 600:
            pruned.append({'date': dt, 'type': rtype,
                           'reason': 'pace_suspect (total time in @)', 'raw': raw})
            continue
        # (3) Total distance < 100m → 'N' was minutes, not meters
        if not pd.isna(qd) and qd < 100:
            pruned.append({'date': dt, 'type': rtype,
                           'reason': 'time-based (minutes not meters)', 'raw': raw})
            continue
        # (4) Continuous tempo (0:00 rest): a real sustained block — analyze it
        # like a continuous fartlek (one connected effort, D_eff = total) rather
        # than pruning it. Needs a usable distance/pace.
        has_zero_rest = bool(re.search(r'0:00\s*rest/mi', raw))
        if rtype == 'tempo' and has_zero_rest:
            if pd.isna(qd) or pd.isna(qp) or qd < 100:
                pruned.append({'date': dt, 'type': rtype,
                               'reason': 'continuous tempo, no usable qd/qp', 'raw': raw})
                continue
            results.append({
                'date': dt, 'type': 'tempo',
                'rep_dist': int(qd), 'rep_count': 1,
                'pace_per_mile': qp, 'rest_per_mile': 0,
                'structure': None, 'd_eff_m': float(qd),
                't_eff_s': qp * qd / MILE_M,
            })
            continue

        # ---------- Nx and rest extraction ----------
        nx = match_nx(raw)

        explicit_rest_per_mile = None
        if nx and nx['rest_per_rep_s'] is not None:
            explicit_rest_per_mile = nx['rest_per_rep_s'] * (MILE_M / nx['rep_dist'])
        else:
            rm = RX_REST.search(raw)
            if rm:
                explicit_rest_per_mile = parse_time_to_seconds(rm.group(1))

        # Total distance: from Nx if present, else from quality_distance_m
        total_m = nx['count'] * nx['rep_dist'] if nx else qd

        # ---------- hardcoded ladder: 4800f @ 2:20/mi is ALWAYS 1600,2x800,4x400 ----------
        # Two exact-value criteria (4800m total + 2:20 logged rest) pin the
        # structure of a large 2017-19 body Max ran as this fixed ladder. Without
        # this the parser mis-reads 4800f as 6x800. Per-rep paces are unknown
        # pre-watch, so all reps take the logged average pace; the rest is the
        # enforced 2:20/mi. Decomposes like a watch-measured ladder.
        if (rtype == 'fartlek' and total_m and 4700 <= total_m <= 4900
                and explicit_rest_per_mile is not None
                and 135 <= explicit_rest_per_mile <= 145 and not pd.isna(qp)):
            reps = LADDER_4800
            times = [qp * d / MILE_M for d in reps]
            rests = [explicit_rest_per_mile * d / MILE_M for d in reps[:-1]] + [0.0]
            d_eff, t_eff = _connected_core(reps, times, rests)
            tot = sum(reps)
            eff_rd = max(100, round((sum(d * d for d in reps) / tot) / 100) * 100)
            results.append({
                'date': dt, 'type': 'interval',
                'rep_dist': eff_rd, 'rep_count': max(1, round(tot / eff_rd)),
                'pace_per_mile': qp, 'rest_per_mile': explicit_rest_per_mile,
                'structure': _structure_label(reps),
                'd_eff_m': round(d_eff, 1), 't_eff_s': round(t_eff, 1),
            })
            continue

        # ---------- continuous_fartlek: 6400-10000m with no positive rest signal ----------
        # Either explicit "0:00 rest/mi" or no rest annotation at all.
        # In continuous-fartlek-only mode (watch profiles) the upper band is
        # dropped and the floor lowered to 4000m, so a continuous track effort
        # the importer coded as a fartlek is never split into intervals below.
        no_rest_annotation = explicit_rest_per_mile is None
        cf_lo, cf_hi = (4000, 10**9) if continuous_fartlek_only else (6400, 10000)
        if (rtype == 'fartlek'
                and (has_zero_rest or no_rest_annotation)
                and total_m and cf_lo <= total_m <= cf_hi):
            results.append({
                'date': dt, 'type': 'continuous_fartlek',
                'rep_dist': int(total_m), 'rep_count': 1,
                'pace_per_mile': qp, 'rest_per_mile': 0, 'structure': None,
                'd_eff_m': float(total_m), 't_eff_s': qp * total_m / MILE_M,
            })
            continue

        # ---------- prune sub-4000m fartleks with no usable rest signal ----------
        if rtype == 'fartlek' and total_m and total_m < 4000 and not nx:
            if has_zero_rest:
                pruned.append({'date': dt, 'type': rtype,
                               'reason': 'sub-4000 fartlek, 0 rest (continuous, no shape)',
                               'raw': raw})
                continue
            if explicit_rest_per_mile is None:
                pruned.append({'date': dt, 'type': rtype,
                               'reason': 'sub-4000 fartlek, no rest annotation (ambiguous)',
                               'raw': raw})
                continue

        # ---------- decomposition ----------
        unresolved_reason = None
        final_type = rtype
        rep_dist = rep_count = pace_per_mile = None

        if nx is not None:
            rep_dist = nx['rep_dist']
            rep_count = nx['count']
            pace_per_mile = (
                nx['pace_raw_s'] * (MILE_M / rep_dist)
                if nx['pace_is_per_rep'] else nx['pace_raw_s']
            )

            # Reclassify fartlek by total distance even when Nx was explicit
            if rtype == 'fartlek':
                rc = reclassify_fartlek(total_m, has_zero_rest)
                if rc is None:
                    unresolved_reason = f'fartlek total out of range ({int(total_m)}m)'
                else:
                    final_type, rep_dist, rep_count = rc
        else:
            if pd.isna(qd) or pd.isna(qp):
                unresolved_reason = 'parser_missing_data'
            else:
                pace_per_mile = qp
                if rtype == 'interval':
                    if qd >= 4800:
                        rep_dist, rep_count = 1600, round(qd / 1600)
                    elif qd >= 3200:
                        rep_dist, rep_count = 800, round(qd / 800)
                    else:
                        unresolved_reason = f'interval<3200m ({int(qd)}m)'
                elif rtype == 'rep':
                    rep_dist, rep_count = 400, round(qd / 400)
                elif rtype == 'tempo':
                    if qd < 7000:
                        rep_dist, rep_count = 1000, round(qd / 1000)
                    else:
                        rep_dist, rep_count = 1600, round(qd / 1600)
                elif rtype == 'fartlek':
                    rc = reclassify_fartlek(qd, has_zero_rest)
                    if rc is None:
                        unresolved_reason = f'fartlek_out_of_range ({int(qd)}m)'
                    else:
                        final_type, rep_dist, rep_count = rc

        if unresolved_reason:
            pruned.append({'date': dt, 'type': rtype,
                           'reason': unresolved_reason, 'raw': raw})
            continue

        # ---------- rest (era/type policy) ----------
        # interval/rep: 2020+ -> this profile's watch median; pre-2020 -> logged,
        # else pre-2020 logged median. tempo/fartlek: logged is real. `trusted`
        # gates the connected projection (untrusted -> legacy formula + reps stay
        # excluded). See effective_rest_per_mile / build_rest_model.
        rest_per_mile, trusted = effective_rest_per_mile(
            final_type, rep_dist, dt.year, explicit_rest_per_mile, rest_model)
        # Nx tempo-intervals (e.g. 4x1600t) carry deliberate ~1:00/mi rest even
        # when unlogged (Max: tempo rest is real, either 0:00 or ~1:00). Trust
        # the tempo default for them so they get the connected projection; an
        # un-Nx'd compound tempo (no clean structure) stays on the legacy path.
        if final_type == 'tempo' and not trusted and nx is not None:
            trusted = True

        d_eff_m = t_eff_s = np.nan
        if trusted and rep_count >= 2:
            rep_t = pace_per_mile * rep_dist / MILE_M
            rep_rest_s = rest_per_mile * rep_dist / MILE_M
            d_eff_m, t_eff_s = _connected_core(
                [rep_dist] * rep_count, [rep_t] * rep_count,
                [rep_rest_s] * (rep_count - 1) + [0.0])

        results.append({
            'date': dt, 'type': final_type,
            'rep_dist': rep_dist, 'rep_count': rep_count,
            'pace_per_mile': pace_per_mile, 'rest_per_mile': rest_per_mile,
            'structure': None, 'd_eff_m': d_eff_m, 't_eff_s': t_eff_s,
        })

    # Explicit columns so an empty result (a profile with no quality workouts,
    # e.g. a watch import of all easy runs) is still a well-formed frame the
    # caller can sort/write rather than a column-less DataFrame.
    decomp_cols = ['date', 'type', 'rep_dist', 'rep_count', 'pace_per_mile',
                   'rest_per_mile', 'structure', 'd_eff_m', 't_eff_s']
    return pd.DataFrame(results, columns=decomp_cols), pd.DataFrame(pruned)


def main():
    p = argparse.ArgumentParser(description=(__doc__ or '').split('\n\n')[0])
    p.add_argument('--diagnostics', action='store_true',
                   help='Also write workout_pruned.csv (audit of rows that '
                        'failed to decompose) into output/debug/. Off by '
                        'default — the file is informational and not '
                        'consumed by anything downstream.')
    p.add_argument('--continuous-fartlek-only', action='store_true',
                   help='Watch-import profiles: classify every no-rest fartlek '
                        '>=4000m as continuous_fartlek, never splitting into '
                        'intervals by distance.')
    args = p.parse_args()

    src = DATA_DIR / 'daily.csv'
    if not src.exists():
        raise SystemExit(
            f'Could not find {src}. Run build_dataset.py first to produce daily.csv.'
        )
    print(f'Source: {src}')
    daily = pd.read_csv(src)

    # Rest model (watch-median + pre-2020 logged) drives the era/type rest
    # policy in decompose(); built from this profile's own measured + log data.
    measured_path = DATA_DIR / 'workout_measured.csv'
    measured = pd.read_csv(measured_path) if measured_path.exists() else None
    rest_model = build_rest_model(measured, daily)

    decomposed, pruned = decompose(
        daily, continuous_fartlek_only=args.continuous_fartlek_only,
        rest_model=rest_model)

    # Watch enrichment: days the rep-extraction layer reconstructed exactly
    # replace their parsed rows (real structure + measured rest); watch-only
    # days are added outright. See measured_to_decomposed for the rules.
    if measured is not None:
        m_rows, enriched = measured_to_decomposed(measured, daily)
        if m_rows:
            decomposed = decomposed[~decomposed['date'].isin(enriched)]
            decomposed = pd.concat(
                [decomposed, pd.DataFrame(m_rows)], ignore_index=True)
            print(f'Watch-enriched: {len(enriched)} days '
                  f'({len(m_rows)} decomposed rows) from {measured_path}')

    decomposed = decomposed.sort_values('date').reset_index(drop=True)
    pruned = pruned.sort_values('date').reset_index(drop=True) if len(pruned) else pruned

    decomposed_path = DATA_DIR / 'workout_decomposed.csv'
    decomposed.to_csv(decomposed_path, index=False)
    print(f'Wrote {decomposed_path}  ({len(decomposed)} rows)')

    if args.diagnostics:
        DEBUG_DIR.mkdir(parents=True, exist_ok=True)
        pruned_path = DEBUG_DIR / 'workout_pruned.csv'
        pruned.to_csv(pruned_path, index=False)
        print(f'Wrote {pruned_path}      ({len(pruned)} rows)')

    print()
    print('Type counts:')
    print(decomposed['type'].value_counts().to_string())


if __name__ == '__main__':
    main()
