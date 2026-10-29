"""Shoe roles, physical pairs and per-pair mileage — the single source of shoe facts.

The log records ONE shoe per day. The dashboard's current-training-shoe
mileage and the Shoes tab both need more than that, and both read
``build_shoes`` so they can never disagree about which physical pair a day
belongs to or how many miles it carries. Full rationale and measurements:
docs/shoes-reference.md.

Roles are decided per logged label, from the logged shoes alone:

  spike    worn ONLY on track days (daily ``surface == 'Track'``). XC is out of
           scope (Max, Oct 2026: no XC spikes owned any more).
  racer    any other label with < EASY_SHARE_MIN easy (recovery/trail) days.
  trainer  everything else, trail/snow shoes included.

"Never raced in" does NOT define a trainer: the Clifton 8 raced the 2023-03-04
Tom King HM. The easy-day share separates the groups cleanly (Oct 2026:
trainers >= 61%, racers <= 37%).

Physical pairs: one label is one pair, and an asterisk marks a distinct pair —
from Oct 2026 Max alternates asterisk / no asterisk on consecutive pairs of a
model, even across years. The only exceptions are the historical re-buys that
predate that convention, listed in LEGACY_PAIR_STARTS. Closed list: never
infer splits from gaps in usage.

Mileage: a non-spike day goes wholly to the logged shoe (Max never changes
non-spike shoes mid-run; road racers are worn for warm-up and cool-down too).
A spike day keeps only the spiked part — the race distances plus the infield
tempo if one is logged, or exactly the workout's ``quality_distance_m`` — and
the rest goes to the most recently worn trainer pair BEFORE that day (never a
trainer first worn later: a new trainer isn't debuted on a race warm-up). Those inferred rows carry ``logged=False``: they
count as wear but never earn PR or performance credit.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.shared.units import METERS_PER_MILE

EASY_RUN_TYPES = ('recovery', 'trail')
EASY_SHARE_MIN = 0.5
# Max's same-label re-buys from before the alternating-asterisk convention:
# label -> first wear of each later pair. Set in stone — don't extend; new pairs
# flip the asterisk instead.
LEGACY_PAIR_STARTS = {
    'Hoka Clifton 8':            ['2022-12-28'],
    'Saucony Endorphin Speed 2': ['2025-03-22'],
    'Saucony Endorphin Speed 4': ['2025-06-06', '2026-05-07'],
}
# Track race days: the infield tempo (logged as ``Nrec/t``) is run in spikes.
TRACK_TEMPO_MI = 0.8
# Brand = first word of the label, except multi-word brands.
BRAND_ALIASES = {'New': 'New Balance'}


def scrub_asterisk(name):
    """Model name for display: the label without its pair-disambiguating
    asterisk, or None for a blank/NaN label."""
    if name is None or (isinstance(name, float) and math.isnan(name)):
        return None
    s = str(name).strip()
    if not s or s.lower() in ('nan', 'none'):
        return None
    return s.rstrip('*').strip()


def brand_of(label) -> str:
    word = scrub_asterisk(label).split()[0]
    return BRAND_ALIASES.get(word, word)


def _legacy_pair_starts():
    """LEGACY_PAIR_STARTS for the Max profile, empty otherwise — the labels
    are his physical shoes."""
    if os.environ.get('RP_PROFILE', 'max') != 'max':
        return {}
    return LEGACY_PAIR_STARTS


@dataclass(frozen=True)
class Shoes:
    """``labels``: per logged label — role, easy_share.
    ``pairs``: per physical pair (index = pair id) — label, model, brand, role,
    name, first, last, days, miles.
    ``wear``: one row per (date, pair) — miles, logged. ``logged`` is False on
    the warm-up rows inferred for spike days."""
    labels: pd.DataFrame
    pairs: pd.DataFrame
    wear: pd.DataFrame

    @property
    def empty(self) -> bool:
        return self.wear.empty

    def logged_pair(self) -> pd.Series:
        """Pair id of the shoe actually logged each day (indexed by date) —
        the one that earns that day's PRs and performances."""
        w = self.wear[self.wear['logged']]
        return pd.Series(w['pair'].to_numpy(), index=pd.DatetimeIndex(w['date']))


def build_shoes(daily: pd.DataFrame, races: pd.DataFrame) -> Shoes:
    """Roles, pairs and per-pair mileage from ``daily`` + ``races``.

    Mileage reads ``daily['miles']`` as given — callers pass display mileage
    (``effective_daily_miles``), the same figure every other total uses."""
    daily = daily.sort_values('date').reset_index(drop=True)
    daily['label'] = daily['shoes'].map(
        lambda s: str(s).strip() if scrub_asterisk(s) else None)
    worn = daily.dropna(subset=['label'])
    if worn.empty:
        return Shoes(pd.DataFrame(columns=['role', 'easy_share']),
                     pd.DataFrame(columns=['label', 'model', 'brand', 'role', 'name',
                                           'first', 'last', 'days', 'miles']),
                     pd.DataFrame(columns=['date', 'pair', 'miles', 'logged']))

    labels = _classify(worn)
    worn = worn.assign(role=worn['label'].map(labels['role']),
                       pair=_assign_pairs(worn))
    wear = _allocate(worn, races)
    return Shoes(labels, _pair_table(worn, wear), wear)


def _classify(worn: pd.DataFrame) -> pd.DataFrame:
    track_only = worn.groupby('label')['surface'].agg(
        lambda s: bool((s == 'Track').all()))
    easy = worn['run_type'].isin(EASY_RUN_TYPES).groupby(worn['label']).mean()
    role = pd.Series('trainer', index=track_only.index)
    role[easy < EASY_SHARE_MIN] = 'racer'
    role[track_only] = 'spike'
    return pd.DataFrame({'role': role, 'easy_share': easy})


def _assign_pairs(worn: pd.DataFrame) -> pd.Series:
    """Pair id per worn row: ``label|k``, k = 0 except on the legacy re-buys,
    where k counts the LEGACY_PAIR_STARTS dates on or before the wear."""
    k = pd.Series(0, index=worn.index)
    for label, starts in _legacy_pair_starts().items():
        rows = worn['label'] == label
        starts = pd.to_datetime(starts)
        # A start that isn't a logged wear of the label means the log or the
        # list changed under it — the pairs would silently shift.
        dates = worn.loc[rows, 'date']
        missing = starts[~starts.isin(dates) & (starts <= dates.max())]
        for d in missing:
            print(f'  WARNING: shoes: LEGACY_PAIR_STARTS {label} {d.date()} '
                  f'is not a logged wear of that label')
        k[rows] = np.searchsorted(starts.values, worn.loc[rows, 'date'].values,
                                  side='right')
    return worn['label'] + '|' + k.astype(str)


def _spike_miles(row, race_mi: pd.Series):
    """Miles actually run in spikes on a spike day, or None when the day gives
    no basis for the split (it then stays whole on the spike)."""
    if row.run_type == 'race':
        if row.date not in race_mi.index:
            return None
        tempo = TRACK_TEMPO_MI if 'rec/t' in str(row.workout_raw) else 0.0
        return float(race_mi[row.date]) + tempo
    if pd.isna(row.quality_distance_m):
        return None
    return float(row.quality_distance_m) / METERS_PER_MILE


def _allocate(worn: pd.DataFrame, races: pd.DataFrame) -> pd.DataFrame:
    race_mi = races.groupby('date')['distance_m'].sum() / METERS_PER_MILE
    # The warm-up shoe: the trainer pair worn most recently BEFORE the
    # spike day (strictly — a trainer debuting later that week isn't it).
    trainers = (worn.loc[worn['role'] == 'trainer', ['date', 'pair']]
                .rename(columns={'pair': 'warmup_pair'}).sort_values('date'))
    spikes = worn[worn['role'] == 'spike'].sort_values('date')
    if len(spikes) and len(trainers):
        spikes = pd.merge_asof(spikes, trainers, on='date', direction='backward',
                               allow_exact_matches=False)
    else:
        spikes = spikes.assign(warmup_pair=None)

    rows = [(r.date, r.pair, float(r.miles), True)
            for r in worn[worn['role'] != 'spike'].itertuples()]
    for r in spikes.itertuples():
        spiked = _spike_miles(r, race_mi)
        if spiked is None or pd.isna(r.warmup_pair):
            why = ('no prior daily trainer' if spiked is not None
                   else 'no race rows' if r.run_type == 'race'
                   else 'no quality_distance_m')
            print(f'  WARNING: shoes: {r.date.date()} {r.label}: {why}; '
                  f'whole day ({r.miles:.1f} mi) left on the spike')
            rows.append((r.date, r.pair, float(r.miles), True))
            continue
        spiked = min(spiked, float(r.miles))
        rows.append((r.date, r.pair, spiked, True))
        if r.miles - spiked > 1e-9:
            rows.append((r.date, r.warmup_pair, float(r.miles) - spiked, False))
    wear = pd.DataFrame(rows, columns=['date', 'pair', 'miles', 'logged'])
    return (wear.sort_values(['date', 'logged'], ascending=[True, False])
            .reset_index(drop=True))


def _pair_table(worn: pd.DataFrame, wear: pd.DataFrame) -> pd.DataFrame:
    meta = worn.groupby('pair').agg(label=('label', 'first'), role=('role', 'first'))
    stats = wear.groupby('pair').agg(first=('date', 'min'), last=('date', 'max'),
                                     days=('date', 'nunique'), miles=('miles', 'sum'))
    pairs = meta.join(stats).sort_values('first')
    pairs['model'] = pairs['label'].map(scrub_asterisk)
    pairs['brand'] = pairs['label'].map(brand_of)
    # Display name: the model, numbered by first wear when Max has owned more
    # than one pair of it (asterisked and plain labels alike).
    n = pairs.groupby('model')['first'].rank(method='first').astype(int)
    multi = pairs.groupby('model')['model'].transform('size') > 1
    pairs['name'] = np.where(multi, pairs['model'] + ' (pair ' + n.astype(str) + ')',
                             pairs['model'])
    return pairs[['label', 'model', 'brand', 'role', 'name',
                  'first', 'last', 'days', 'miles']]
