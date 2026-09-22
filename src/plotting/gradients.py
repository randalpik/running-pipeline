"""Piecewise-linear colour gradients shared by every plot that encodes a
continuous variable as colour.

Two shapes live here:

* **Anchor gradients** — a list of ``(value, '#hex')`` control points in DATA
  units, evaluated by :func:`gradient_at`. Misc. Trends fills its envelope
  panels with them; Recovery and Long Runs colour their point markers with the
  same anchors so a temperature (or wind, or altitude) reads identically
  wherever it appears.
* **The solar gradient** — the one gradient whose anchor POSITIONS move per
  day, because they are astronomical times rather than fixed data values. See
  :func:`solar_ramp_from_anchors`.

This module is the single definition of both; ``plot_qualitative_trends`` and
``plotting.color_by`` import from here rather than carrying their own copies.
"""
from __future__ import annotations

import numpy as np

# Solar gradient control colors (RGB). The day's anchor MINUTES come from
# solar_anchors_local; these are the colors at each anchor. The purples are kept
# light enough to read against the #1a1a1a panel (a near-black night vanishes).
SOLAR_NIGHT = (48, 32, 82)        # before dawn / after dusk (lightened purple)
SOLAR_TWILIGHT = (96, 58, 140)    # civil twilight (purple)
SOLAR_HORIZON = (240, 138, 61)    # sunrise / sunset (orange)
SOLAR_NOON = (154, 208, 245)      # full daylight, sun >= +18deg (light blue)

# The colour a point/pixel takes when its value is missing. Doubles as the
# "nothing to see here" end of the one-sided ramps (elevation gain at 0 ft, a
# race far enough back to have decayed away), so absent and negligible read the
# same deliberately.
#
# Deliberately close to the #1a1a1a panel background (contrast 2.3, vs 6.7 for
# the #a0a0a0 this replaces). Every ramp that ends here then runs dark -> bright
# rather than light-gray -> mid-colour, which is what makes the coloured end
# read as intensity: it roughly doubles the gray-to-vivid separation on the
# one-sided ramps (Recent race 1.35 -> 2.08) while letting no-data points
# recede instead of glowing. Still light enough to see a dot.
NO_DATA_GRAY = (85, 85, 85)


# ---------- color utilities ----------

def hex_to_rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def rgb_to_hex(c):
    """(r, g, b) -> '#rrggbb'. The wire format for per-point marker colours."""
    return '#%02x%02x%02x' % (int(c[0]), int(c[1]), int(c[2]))


def lerp_rgb(c1, c2, t):
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))


def gradient_at(value, anchors):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return NO_DATA_GRAY
    if value <= anchors[0][0]:
        return hex_to_rgb(anchors[0][1])
    if value >= anchors[-1][0]:
        return hex_to_rgb(anchors[-1][1])
    for i in range(len(anchors) - 1):
        v0, h0 = anchors[i]
        v1, h1 = anchors[i + 1]
        if v0 <= value <= v1:
            t = (value - v0) / (v1 - v0) if v1 > v0 else 0
            return lerp_rgb(hex_to_rgb(h0), hex_to_rgb(h1), t)
    return hex_to_rgb(anchors[-1][1])


def css_gradient(anchors):
    """``linear-gradient(to right, ...)`` for the sidebar legend strip.

    Stop percentages come from the anchor VALUES, not from even spacing — the
    temperature ramp's 22 degC middle stop sits at 64% of the -10..40 span, and
    drawing it at 50% would misreport every colour on the strip. Degenerate
    (zero-width) anchor spans collapse to the first colour.
    """
    lo = float(anchors[0][0])
    hi = float(anchors[-1][0])
    span = hi - lo
    if span <= 0:
        return f'linear-gradient(to right, {anchors[0][1]}, {anchors[-1][1]})'
    stops = ', '.join(
        f'{h} {100.0 * (float(v) - lo) / span:.4g}%' for v, h in anchors)
    return f'linear-gradient(to right, {stops})'


# ---------- gradient ramp (x-independent panels) ----------

def gradient_ramp(anchors, y0, y1, h_px):
    """(h_px, 3) uint8 vertical colour ramp = ``gradient_at`` sampled at each
    output pixel-row's y. The x-independent case of a raster ``column_colors``
    (same ramp every day), for the anchor-based panels (temp/humidity/wind/
    volume/weight/altitude)."""
    yp = y1 - (np.arange(h_px) + 0.5) / h_px * (y1 - y0)
    return np.array([gradient_at(float(v), anchors) for v in yp],
                    dtype=np.uint8)


# ---------- solar raster color ramp ----------

def solar_ramp_from_anchors(a, y_p):
    """(len(y_p), 3) uint8 ramp for one day: night -> twilight -> sunrise
    (orange) -> blue (once the sun reaches +18deg) -> noon -> blue -> +18deg
    -> sunset (orange) -> twilight -> night, anchored at the day's solar times.
    The orange->blue shoulder is the sunrise..+18deg span (astronomically tied,
    wider than a fixed clock offset). ``a`` is a precomputed
    ``solar_anchors_local`` dict; ``y_p`` is the array of pixel-row clock
    minutes (descending, top=late)."""
    noon = a['solar_noon'] if a['solar_noon'] is not None else 720.0
    pts = [(0.0, SOLAR_NIGHT)]  # (minute, color)
    if a['twilight_begin'] is not None:
        pts.append((a['twilight_begin'], SOLAR_TWILIGHT))
    if a['sunrise'] is not None:
        pts.append((a['sunrise'], SOLAR_HORIZON))
    if a['blue_begin'] is not None:
        pts.append((a['blue_begin'], SOLAR_NOON))
    pts.append((noon, SOLAR_NOON))
    if a['blue_end'] is not None:
        pts.append((a['blue_end'], SOLAR_NOON))
    if a['sunset'] is not None:
        pts.append((a['sunset'], SOLAR_HORIZON))
    if a['twilight_end'] is not None:
        pts.append((a['twilight_end'], SOLAR_TWILIGHT))
    pts.append((1440.0, SOLAR_NIGHT))
    # Enforce strictly increasing minutes for np.interp (clamp any disorder).
    xs, last = [], -1.0
    for m, _ in pts:
        m = max(m, last + 1e-6)
        xs.append(m)
        last = m
    xs = np.asarray(xs)
    cols = np.asarray([c for _, c in pts], dtype=float)
    out = np.empty((len(y_p), 3), dtype=np.uint8)
    for ch in range(3):
        out[:, ch] = np.clip(np.interp(y_p, xs, cols[:, ch]), 0, 255)
    return out


# ---------- the equinox demo ramp (legend strip for Time of day) ----------

# An idealized day, symmetric about solar noon: sunrise at 06:00, sunset at
# 18:00, with civil twilight and the +18deg "full daylight" shoulder at the
# spacings a mid-latitude equinox produces. It exists ONLY to draw the sidebar
# legend strip. Point colours always use the run's own date and city, so a real
# 6 am run is night in December and full daylight in June — which is why the
# strip carries qualitative "Dawn"/"Dusk" labels rather than clock times that
# would be wrong for most of the year.
EQUINOX_ANCHORS = {
    'twilight_begin': 360.0 - 27.0,
    'sunrise': 360.0,
    'blue_begin': 360.0 + 96.0,
    'solar_noon': 720.0,
    'blue_end': 1080.0 - 96.0,
    'sunset': 1080.0,
    'twilight_end': 1080.0 + 27.0,
}


def solar_css_gradient(n_stops=64):
    """``linear-gradient(to right, ...)`` across a whole equinox day (00:00 on
    the left, 24:00 on the right), sampled from :data:`EQUINOX_ANCHORS`. Uses
    many stops because the ramp is piecewise over seven anchors and CSS would
    otherwise straight-line across the twilight shoulders."""
    mins = np.linspace(0.0, 1440.0, n_stops)
    cols = solar_ramp_from_anchors(EQUINOX_ANCHORS, mins)
    stops = ', '.join(
        f'{rgb_to_hex(cols[i])} {100.0 * i / (n_stops - 1):.4g}%'
        for i in range(n_stops))
    return f'linear-gradient(to right, {stops})'
