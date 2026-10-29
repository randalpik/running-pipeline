"""Shoe-brand colours for the Shoes tab — derived, with optional hand pins.

The location palette's method (docs/location-colour-reference.md), reduced to
one colour per brand: brands are taken in first-wear order and each gets the
candidate colour farthest (OKLab dE) from every colour already placed. A
brand first worn later appends to the order, so no existing brand ever
changes colour; the colour maths is location_palette's.

BRAND_PINS hand-picks a brand's colour (Max, Oct 2026: derive, and pin
individual brands if wanted). Pins are placed first, so every derived colour
keeps its distance from them.
"""
from __future__ import annotations

import numpy as np

from src.plotting.location_palette import (_BG_LAB, _BG_MIN_DE, _in_gamut,
                                           _lch_to_lab, delta_e, hex_to_oklab,
                                           oklab_to_hex)

BRAND_PINS: dict[str, str] = {}

# Candidate grid. Lighter and more chromatic than the location grid: a bar
# can be 1px wide, and it still has to read on the dark page.
_HUE_STEP = 4
_L_RANGE = (0.58, 0.84, 0.02)
_C_RANGE = (0.08, 0.32, 0.015)


def _candidates():
    H, L, C = np.meshgrid(np.arange(0, 360, _HUE_STEP, dtype=float),
                          np.arange(*_L_RANGE), np.arange(*_C_RANGE),
                          indexing='ij')
    lab = _lch_to_lab(L.ravel(), C.ravel(), H.ravel())
    lab = lab[_in_gamut(lab)]
    off_bg = np.linalg.norm(lab - _BG_LAB, axis=1) * 100 >= _BG_MIN_DE
    return lab[off_bg]


def brand_colors(brands) -> dict:
    """brand -> hex for ``brands``, which must be in first-wear order."""
    placed = {b: hex_to_oklab(hx) for b, hx in BRAND_PINS.items()}
    out = dict(BRAND_PINS)
    cand = None
    for b in brands:
        if b in out:
            continue
        if cand is None:
            cand = _candidates()
        if not placed:
            # nothing to be far from yet: take the most chromatic candidate
            idx = int(np.argmax(np.hypot(cand[:, 1], cand[:, 2])))
        else:
            prior = np.array(list(placed.values()))
            closest = np.linalg.norm(cand[:, None, :] - prior[None, :, :],
                                     axis=-1).min(axis=1)
            idx = int(np.argmax(closest))
        placed[b] = cand[idx]
        out[b] = oklab_to_hex(cand[idx])[0]
    return {b: out[b] for b in brands}


def min_separation(colors: dict) -> float:
    """Smallest pairwise dE among ``colors`` (the build prints it)."""
    labs = [hex_to_oklab(h) for h in colors.values()]
    return min((delta_e(a, b) for i, a in enumerate(labs) for b in labs[i + 1:]),
               default=float('inf'))
