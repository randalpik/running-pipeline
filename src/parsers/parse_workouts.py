"""
parse_workouts.py — Decompose quality workouts from daily.csv into
                    (date, type, rep_dist, rep_count, pace_per_mile, rest_per_mile).

Output: workout_decomposed.csv  (one row per resolved quality workout)
        workout_pruned.csv      (rows that didn't make it, with reasons)

Reads daily.csv from data/ and writes both outputs alongside it.
"""
import sys
import pandas as pd
import re
from pathlib import Path
from datetime import date

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.shared.paths import DATA_DIR

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


def decompose(daily_df):
    """
    Filter quality workouts, decompose each row.
    Returns (decomposed_df, pruned_df).
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
        # (4) Continuous tempo: rest 0 in string
        has_zero_rest = bool(re.search(r'0:00\s*rest/mi', raw))
        if rtype == 'tempo' and has_zero_rest:
            pruned.append({'date': dt, 'type': rtype,
                           'reason': 'continuous tempo (0 rest)', 'raw': raw})
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

        # ---------- continuous_fartlek: 6400-10000m with no positive rest signal ----------
        # Either explicit "0:00 rest/mi" or no rest annotation at all.
        no_rest_annotation = explicit_rest_per_mile is None
        if (rtype == 'fartlek'
                and (has_zero_rest or no_rest_annotation)
                and total_m and 6400 <= total_m <= 10000):
            results.append({
                'date': dt, 'type': 'continuous_fartlek',
                'rep_dist': int(total_m), 'rep_count': 1,
                'pace_per_mile': qp, 'rest_per_mile': 0,
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

        # ---------- rest ----------
        if explicit_rest_per_mile is not None:
            rest_per_mile = explicit_rest_per_mile
        else:
            rest_per_mile = default_rest_per_mile(final_type, rep_dist)

        results.append({
            'date': dt, 'type': final_type,
            'rep_dist': rep_dist, 'rep_count': rep_count,
            'pace_per_mile': pace_per_mile, 'rest_per_mile': rest_per_mile,
        })

    return pd.DataFrame(results), pd.DataFrame(pruned)


def main():
    src = DATA_DIR / 'daily.csv'
    if not src.exists():
        raise SystemExit(
            f'Could not find {src}. Run build_dataset.py first to produce daily.csv.'
        )
    print(f'Source: {src}')
    daily = pd.read_csv(src)
    decomposed, pruned = decompose(daily)

    decomposed = decomposed.sort_values('date').reset_index(drop=True)
    pruned = pruned.sort_values('date').reset_index(drop=True) if len(pruned) else pruned

    decomposed_path = DATA_DIR / 'workout_decomposed.csv'
    pruned_path = DATA_DIR / 'workout_pruned.csv'
    decomposed.to_csv(decomposed_path, index=False)
    pruned.to_csv(pruned_path, index=False)

    print(f'Wrote {decomposed_path}  ({len(decomposed)} rows)')
    print(f'Wrote {pruned_path}      ({len(pruned)} rows)')
    print()
    print('Type counts:')
    print(decomposed['type'].value_counts().to_string())


if __name__ == '__main__':
    main()
