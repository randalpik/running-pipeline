"""legacy_training.py — grammar and row builders for the snapshot `training`
section (hand-verified pre-log workouts, 2014-15).

The section carries [date, location, type, decomp]; build_dataset joins
location metadata and writes data/training_legacy.csv. These records are
trusted at the watch-verified tier (Max has exact decomps and times), but
they deliberately create NO daily.csv rows: 2014-15 volume is covered by the
UNLOGGED_MILES constants, so nothing here may touch mileage.

Decomp grammar — comma-separated blocks of `[Nx]DIST@TIME`:
  - bare DIST is meters; a `mi` suffix is miles (`1200`, `5mi`);
  - `NxD` runs D N times (`4x1200`), otherwise one rep;
  - TIME is the duration of that block: `M:SS` or bare seconds are the time
    PER REP (`4x1200@4:28` = each 1200 in 4:28; `400@75` = a 75 s 400);
    `Nmin` or `H:MM:SS` is a TOTAL duration (`11mi@80min`, long runs).
  - an optional `/REST r` token per block is reserved for the future
    (`4x1200@4:28/2:30r` = 2:30 rest after each rep); rest was not recorded
    for the legacy era, so the quality builder falls back to
    parse_workouts.effective_rest_per_mile's era model when absent.

The per-rep-time semantics were confirmed by Max (July 2026); if entries
ever turn out per-mile, `block_time_s()` is the single seam to swap.

Types consumed today: tempo / interval / rep / fartlek (quality; fartlek
splits interval-vs-rep by effective rep length, mirroring the measured
hand-log collision rule) and long. Hill types are parsed and stored but not
yet plotted (deferred until the data's shape is known).
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.shared.units import METERS_PER_MILE
from src.shared.paths import DATA_DIR

LEGACY_CSV = 'training_legacy.csv'

# Stable on-disk shape (build_dataset writes these even when empty).
LEGACY_COLUMNS = ['date', 'location', 'type', 'decomp',
                  'display_name', 'city_state', 'terrain_type',
                  'quality_distance_m']

QUALITY_TYPES = {'tempo', 'interval', 'rep', 'fartlek'}
LONG_TYPE = 'long'

_BLOCK_RE = re.compile(
    r'^\s*(?:(\d+)\s*[xX]\s*)?'            # optional N x
    r'(\d+(?:\.\d+)?)\s*(mi)?\s*'          # distance, optional mi suffix
    r'@\s*([\d:.]+\s*(?:min)?)\s*'         # time token
    r'(?:/\s*([\d:.]+)\s*r)?\s*$'          # reserved optional rest token
)


def _parse_time_token(tok):
    """Seconds from a decomp time token. Returns (seconds, is_total):
    `Nmin` and `H:MM:SS` are totals; `M:SS` and bare `SS` are per-rep."""
    tok = tok.strip().lower()
    if tok.endswith('min'):
        return float(tok[:-3]) * 60.0, True
    parts = tok.split(':')
    if len(parts) == 3:
        return (int(parts[0]) * 3600 + int(parts[1]) * 60
                + float(parts[2])), True
    if len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1]), False
    return float(parts[0]), False


def block_time_s(reps, dist_m, time_s, is_total):
    """Per-rep seconds for a block — the pace-semantics seam. Confirmed
    convention: the @value IS the per-rep duration; totals divide out."""
    return time_s / reps if is_total else time_s


def parse_decomp(decomp):
    """Parse a decomp string into block dicts:
    {'reps', 'dist_m', 'time_s' (per rep), 'rest_s' (after each rep, or
    None), 'is_total'}. Raises ValueError on malformed input — the section
    is hand-authored and a silent skip would quietly drop a workout."""
    blocks = []
    for part in str(decomp).split(','):
        if not part.strip():
            continue
        m = _BLOCK_RE.match(part)
        if not m:
            raise ValueError(f'unparseable decomp block {part!r} in {decomp!r}')
        reps = int(m.group(1)) if m.group(1) else 1
        dist_m = float(m.group(2)) * (METERS_PER_MILE if m.group(3) else 1.0)
        time_s, is_total = _parse_time_token(m.group(4))
        rest_s = None
        if m.group(5):
            rest_s, rest_total = _parse_time_token(m.group(5))
            if rest_total:
                raise ValueError(f'rest token must be per-rep in {part!r}')
        blocks.append({'reps': reps, 'dist_m': dist_m,
                       'time_s': block_time_s(reps, dist_m, time_s, is_total),
                       'rest_s': rest_s, 'is_total': is_total})
    if not blocks:
        raise ValueError(f'empty decomp {decomp!r}')
    return blocks


def quality_total_m(decomp):
    """Total quality distance (m) of a decomp — build_dataset stamps this as
    quality_distance_m so the XC 5K-tempo rule can fire for legacy days."""
    return float(sum(b['reps'] * b['dist_m'] for b in parse_decomp(decomp)))


def load_legacy_training(data_dir=None):
    """The training_legacy.csv frame (empty-with-columns when absent). The
    single source for legacy rows and the legacy-date trust set."""
    path = Path(data_dir or DATA_DIR) / LEGACY_CSV
    if not path.exists():
        return pd.DataFrame(columns=LEGACY_COLUMNS)
    df = pd.read_csv(path)
    for col in LEGACY_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    return df


def legacy_dates(df=None):
    """Set of 'YYYY-MM-DD' strings for all legacy training days (any type) —
    the hand-verified trust tier, symmetric with _watch_verified_dates."""
    if df is None:
        df = load_legacy_training()
    if df.empty:
        return set()
    return set(pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d'))


def legacy_quality_rows(df, dp3_at=None, rest_model=None):
    """workout_decomposed-schema dicts for legacy quality days, mirroring
    measured_to_decomposed's contract: one row per day, whole-workout
    d_eff_m/t_eff_s from the per-rep structure via _connected_core, and a
    run-length-encoded `structure` headline. watch_pace_raw stays NaN (there
    was no watch), which also keeps the Workouts hover from rendering a
    phantom "Watch:" line."""
    from src.shared.workouts import _connected_core
    from src.parsers.parse_workouts import (_structure_label,
                                            effective_rest_per_mile)

    rows = []
    if df is None or df.empty:
        return rows
    quality = df[df['type'].isin(QUALITY_TYPES)]
    for _, r in quality.iterrows():
        dt = pd.to_datetime(r['date']).date()
        blocks = parse_decomp(r['decomp'])
        dists = np.concatenate([[b['dist_m']] * b['reps'] for b in blocks])
        times = np.concatenate([[b['time_s']] * b['reps'] for b in blocks])
        total = float(dists.sum())
        pace = float(times.sum()) / (total / METERS_PER_MILE)

        rep_dist = (dists ** 2).sum() / total
        rep_dist = max(100, round(rep_dist / 100) * 100)
        rep_count = max(1, round(total / rep_dist))
        if r['type'] == 'fartlek':
            # Same collision rule as measured hand-log fartleks.
            final_type = 'interval' if rep_dist >= 800 else 'rep'
        else:
            final_type = r['type']

        # Rest: per-block token when present, else the era rest model —
        # never rest-free, which would collapse d_eff to the full total.
        explicit = [b['rest_s'] for b in blocks for _ in range(b['reps'])]
        rest_per_mile, _ = effective_rest_per_mile(
            final_type, rep_dist, dt.year, None, rest_model)
        rests = np.array([e if e is not None
                          else rest_per_mile * d / METERS_PER_MILE
                          for e, d in zip(explicit, dists)], float)

        dp3, vmax = dp3_at(dt) if dp3_at is not None else (None, None)
        d_eff, t_eff = _connected_core(dists, times, rests, dp3=dp3, vmax=vmax)
        rows.append({'date': dt, 'type': final_type,
                     'rep_dist': int(rep_dist), 'rep_count': int(rep_count),
                     'pace_per_mile': pace, 'rest_per_mile': rest_per_mile,
                     'watch_pace_raw': np.nan,
                     'structure': _structure_label(dists),
                     'd_eff_m': round(d_eff, 1), 't_eff_s': round(t_eff, 1)})
    return rows


def legacy_long_run_rows(df=None):
    """Daily-frame-shaped rows for legacy long runs, for injection inside
    project_long_runs ONLY (they must never reach daily.csv — mileage).
    Blank workout_raw/conditions/partners pass the snow/partner gates."""
    if df is None:
        df = load_legacy_training()
    rows = []
    if df.empty:
        return pd.DataFrame(rows)
    for _, r in df[df['type'] == LONG_TYPE].iterrows():
        blocks = parse_decomp(r['decomp'])
        total_m = sum(b['reps'] * b['dist_m'] for b in blocks)
        total_s = sum(b['reps'] * b['time_s'] for b in blocks)
        miles = total_m / METERS_PER_MILE
        terrain = r.get('terrain_type')
        rows.append({
            'date': pd.Timestamp(pd.to_datetime(r['date']).date()),
            'miles': round(miles, 2),
            'minutes': round(total_s / 60.0, 1),
            'recovery_pace_sec_per_mi': round(total_s / miles, 1),
            'pace_sec_per_mi': round(total_s / miles, 1),
            'run_type': 'long',
            'location': r.get('location'),
            'display_name': r.get('display_name'),
            'city_state': r.get('city_state'),
            'terrain_type': terrain if pd.notna(terrain) else 'paved',
            'workout_raw': '', 'conditions': '', 'partners': '',
            'temp_c': np.nan,
            'source_file': 'snapshot:training',
        })
    return pd.DataFrame(rows)
