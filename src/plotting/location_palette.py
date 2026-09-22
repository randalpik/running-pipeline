"""Deterministic colour assignment for location groups (states/countries).

Replaces the old hand-maintained hue tables. Nothing here is hand-picked
except four base colours and one step size; every other colour is derived.

Why it is built this way
------------------------
Three properties are required, and they interact:

1. *Stable across tabs.*  Locations and the World Map are separate
   processes, so anything derived from ``hash()`` is out — CPython salts
   string hashing per process.  Everything below is a pure function of the
   data.
2. *Stable across profiles.*  A group's colour is fixed by the **union** of
   every production profile's data, not by one profile's subset, so ``JP``
   is the same colour on Max's map and on Maddy's.
3. *Stable over time.*  A group is ordered by the date it first cleared
   ``QUALIFY_THRESHOLD`` miles inside a calendar year.  That date is a fact
   about the past and cannot change retroactively, and a newly qualifying
   group always qualifies *now* — so it appends to the end of the order and
   never disturbs a colour already in use.

Each group reserves a *track* of ``RESERVED_SHADES`` colours up front — the
base plus the lighter shades its future cities will use.  Tracks are placed
to maximise the minimum perceptual distance (OKLab dE) against every shade
of every group already placed, not just against their bases: two groups with
well-separated bases can still collide once both fan out into their ramps.
Reserving the whole track is what makes growth free — a group gaining its
second, third or fourth city fills a slot that was already held for it, so
no colour is ever reassigned. A group holding fewer traces than it reserved
spreads them over the track's extremes (see ``trace_slots``) rather than
crowding its dark end, which costs nothing and buys contrast.

Placement is greedy in qualification order, which is what buys property 3:
group *k* is placed against groups ``< k`` only.  The cost is myopia — a
globally optimal palette would be tighter, but it could not be grown without
reshuffling.

Measured on the current union (17 groups, 4 shades each): minimum
inter-group dE 5.16, against 3.67 for the hand-tuned palette this replaces —
which reserved nothing.
"""
from __future__ import annotations

import functools
import re

import numpy as np
import pandas as pd

from src.shared.country_codes import country_abbrev
from src.shared.effective_mileage import effective_daily_miles

# ---- the only hand-chosen values ----
# Max's four requested colours. Frozen: these render byte-identical to the
# palette that preceded this module. Their lighter shades are derived.
ANCHOR_BASES = {
    'WA': '#8617be',   # purple
    'TN': '#17be33',   # green
    'CO': '#be6b17',   # orange
    'IL': '#17a2be',   # cyan
}
PERCEPTUAL_STEP = 5.0   # dE between adjacent shades within a group

# ---- derived-design constants ----
QUALIFY_THRESHOLD = 26.2   # miles in one calendar year to earn a colour
RESERVED_SHADES   = 4      # base + 3, reserved up front for every group
_CHROMA_FLOOR     = 0.06   # below this a track reads as gray, not a hue
_BG_MIN_DE        = 18.0   # a base must stand off the #1a1a1a page
_GAMUT_EPS        = 2e-3
_BG_HEX           = '#1a1a1a'

# Candidate grid. Placement quality depends on this being fine enough;
# runtime is linear in its size. Changing it re-derives every non-anchor
# colour, so treat it as part of the palette's definition.
_HUE_STEP   = 4
_L_RANGE    = (0.26, 0.80, 0.02)
_C_RANGE    = (_CHROMA_FLOOR, 0.34, 0.015)


# ---------------------------------------------------------------- colour --
_M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                [0.2119034982, 0.6806995451, 0.1073969566],
                [0.0883024619, 0.2817188376, 0.6299787005]])
_M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                [1.9779984951, -2.4285922050, 0.4505937099],
                [0.0259040371, 0.7827717662, -0.8086757660]])
_M1I, _M2I = np.linalg.inv(_M1), np.linalg.inv(_M2)


def _srgb_to_linear(c):
    c = np.asarray(c, float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c):
    c = np.asarray(c, float)
    return np.where(c <= 0.0031308, c * 12.92,
                    1.055 * np.power(np.clip(c, 0, None), 1 / 2.4) - 0.055)


def rgb_to_oklab(rgb):
    return np.cbrt(_srgb_to_linear(rgb) @ _M1.T) @ _M2.T


def oklab_to_rgb(lab):
    return _linear_to_srgb(((lab @ _M2I.T) ** 3) @ _M1I.T)


def hex_to_oklab(hx):
    return rgb_to_oklab(np.array([int(hx[i:i + 2], 16) / 255
                                  for i in (1, 3, 5)]))


def oklab_to_hex(lab):
    rgb = np.clip(oklab_to_rgb(np.atleast_2d(lab)), 0, 1)
    return ['#{:02x}{:02x}{:02x}'.format(*(int(round(v * 255)) for v in c))
            for c in rgb]


def _lch_to_lab(L, C, H):
    h = np.deg2rad(H)
    return np.stack([L, C * np.cos(h), C * np.sin(h)], axis=-1)


def _lab_to_lch(lab):
    L, a, b = lab
    return L, float(np.hypot(a, b)), float(np.degrees(np.arctan2(b, a)) % 360)


def _in_gamut(lab):
    rgb = oklab_to_rgb(lab)
    return np.all((rgb >= -_GAMUT_EPS) & (rgb <= 1 + _GAMUT_EPS), axis=-1)


def delta_e(a, b):
    """Perceptual distance, OKLab euclidean scaled to a readable range."""
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b))) * 100


_BG_LAB = rgb_to_oklab(np.array([int(_BG_HEX[i:i + 2], 16) / 255
                                 for i in (1, 3, 5)]))


def _track(L0, C0, H, n=RESERVED_SHADES, step=PERCEPTUAL_STEP):
    """n shades at one hue and chroma, each `step` dE lighter than the last."""
    return _lch_to_lab(np.array([L0 + i * step / 100 for i in range(n)]),
                       np.full(n, C0), np.full(n, float(H)))


# ----------------------------------------------------------- group naming --
def group_key(city_state):
    """'Osaka, Japan' and 'Osaka, JP' both -> 'JP'.

    Normalising through country_abbrev first is what lets a watch-sourced
    profile (which stores full country names) and a Drive-sourced one (which
    stores alpha-2) land on the same colour.
    """
    if not isinstance(city_state, str):
        return None
    cs = country_abbrev(city_state)
    m = re.search(r',\s*([A-Z]{2,3})$', cs)
    return m.group(1) if m else cs.strip()


# ------------------------------------------------------ qualification order --
def _qualification_dates(daily_csv):
    """group -> first date its running total in a calendar year hit the
    threshold. Returns an empty Series if the file is missing."""
    try:
        df = pd.read_csv(daily_csv, low_memory=False, parse_dates=['date'])
    except (FileNotFoundError, ValueError):
        return pd.Series(dtype='datetime64[ns]')
    if 'city_state' not in df.columns or df.empty:
        return pd.Series(dtype='datetime64[ns]')
    df = df[df['city_state'].notna()].copy()
    df['miles'] = effective_daily_miles(df)
    df = df[df['miles'] > 0]
    df['group'] = df['city_state'].astype(str).map(group_key)
    df['yr'] = df['date'].dt.year
    df = df.sort_values(['date', 'city_state'])
    df['cum'] = df.groupby(['group', 'yr'])['miles'].cumsum()
    hit = df[df['cum'] >= QUALIFY_THRESHOLD]
    return hit.groupby('group')['date'].min()


def _union_order():
    """Every production profile's qualified groups, earliest-qualifying first.

    Reads the profiles directly rather than the ambient RP_DATA_DIR, because
    the whole point is that the order does not depend on which profile is
    being built.
    """
    from src.profiles import PROFILES
    parts = [_qualification_dates(p.data_dir / 'daily.csv')
             for p in PROFILES if p.prod]
    parts = [s for s in parts if not s.empty]
    if not parts:
        return []
    first = pd.concat(parts).groupby(level=0).min()
    # (date, name) so ties resolve identically in every process
    return [g for g, _ in sorted(first.items(), key=lambda kv: (kv[1], kv[0]))]


# ------------------------------------------------------------- placement --
def _free_candidates(step):
    tracks = []
    for H in range(0, 360, _HUE_STEP):
        for L0 in np.arange(*_L_RANGE):
            for C0 in np.arange(*_C_RANGE):
                t = _track(L0, C0, H, step=step)
                if not _in_gamut(t).all():
                    continue
                if delta_e(t[0], _BG_LAB) < _BG_MIN_DE:
                    continue
                tracks.append(t)
    return np.array(tracks)


def _anchor_track(group, step):
    L, C, H = _lab_to_lch(hex_to_oklab(ANCHOR_BASES[group]))
    t = _track(L, C, H, step=step)
    if not _in_gamut(t).all():
        raise ValueError(
            f'anchor {group} ({ANCHOR_BASES[group]}) leaves sRGB at step '
            f'{step}; the frozen bases cap PERCEPTUAL_STEP at 6')
    return t


@functools.lru_cache(maxsize=4)
def _assign(step=PERCEPTUAL_STEP):
    """group -> (RESERVED_SHADES, 3) OKLab track. Pure; memoised per process."""
    order = _union_order()
    free = None
    placed = {}
    for group in order:
        if group in ANCHOR_BASES:
            placed[group] = _anchor_track(group, step)
            continue
        if free is None:
            free = _free_candidates(step)
        if not placed:
            # nothing to be far from yet: take the most chromatic track
            idx = int(np.argmax(np.hypot(free[:, 0, 1], free[:, 0, 2])))
        else:
            prior = np.concatenate(list(placed.values()), axis=0)
            closest = np.linalg.norm(
                free[:, :, None, :] - prior[None, None, :, :],
                axis=-1).min(axis=(1, 2))
            idx = int(np.argmax(closest))
        placed[group] = free[idx]
    return placed


# ---------------------------------------------------------------- public --
def set_threshold(value):
    """Override the qualification cutoff (the --threshold flag).

    Clears the memoised assignment, because the cutoff decides which groups
    enter the order and therefore every non-anchor color.
    """
    global QUALIFY_THRESHOLD
    QUALIFY_THRESHOLD = float(value)
    _assign.cache_clear()


def palette():
    """group -> list of RESERVED_SHADES hex colours, lightest last."""
    return {g: oklab_to_hex(t) for g, t in _assign().items()}


# Fill order for a group's reserved slots: take the extremes before the
# middle, so a group with fewer traces than RESERVED_SHADES spreads across
# the track instead of crowding its dark end. Two traces land on the
# endpoints (3 steps apart, not 1); the third fills in at index 1 and the
# fourth at index 2. Prefix-ordered, so growing a group only ever adds a
# slot — it never renumbers one already in use.
_SLOT_FILL_ORDER = (0, RESERVED_SHADES - 1, 1, 2)


def trace_slots(n_traces):
    """Reserved slot indices for `n_traces` traces, darkest first."""
    if n_traces <= len(_SLOT_FILL_ORDER):
        return sorted(_SLOT_FILL_ORDER[:n_traces])
    return list(range(n_traces))


def shades(group, count):
    """`count` hex colours for a group, lightest last.

    Past RESERVED_SHADES the track simply keeps stepping — separation is no
    longer guaranteed against other groups, which is the documented cost of
    a group growing beyond its reservation.
    """
    track = _assign().get(group)
    if track is None:
        return None
    if count <= len(track):
        return oklab_to_hex(track[:count])
    L, C, H = _lab_to_lch(track[0])
    return oklab_to_hex(_track(L, C, H, n=count))


def min_separation():
    """Smallest dE between any two shades of any two different groups.

    Exposed so a test can assert the palette has not silently degraded.
    """
    tracks = _assign()
    keys = list(tracks)
    return min(
        float(np.min(np.linalg.norm(
            tracks[a][:, None, :] - tracks[b][None, :, :], axis=-1))) * 100
        for i, a in enumerate(keys) for b in keys[i + 1:])
