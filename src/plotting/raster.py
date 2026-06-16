"""Generic gradient-raster fill for envelope panels.

A panel's gradient envelope can be drawn two ways: as a stack of horizontal
colour strips (``plot_qualitative_trends.add_envelope_strips`` — one fillcolor
per y-band, so colour can only vary with y) or as a single RGBA raster placed
behind the panel via a layout image (colour can vary with BOTH x and y, and
there is no visible banding).

``render_gradient_raster`` is the raster path. It is deliberately generic so the
same primitive serves:

* **Time** — colour depends on the day's solar anchors (x) *and* the clock
  minute (y); only a raster can express this.
* **Temp / Humidity / Wind / Altitude** — colour depends only on y; the raster
  is the x-independent special case and a drop-in replacement for the strip
  stack (kills the faint horizontal banding) if it proves crisp and cheap.

The PNG is encoded with the stdlib only (``zlib`` + ``struct`` + ``base64``) —
no Pillow/matplotlib dependency, consistent with the project's stdlib-only
non-plotly convention.
"""
from __future__ import annotations

import base64
import struct
import zlib

import numpy as np

# Gradient-raster anti-aliasing. The band mask is rendered at RASTER_SS x the
# output resolution on each axis and box-averaged to fractional alpha, so EVERY
# edge — including steep vertical risers — is anti-aliased (a vertical-only alpha
# edge leaves the steep sides stairstepped).
#
# Output width is a FIXED 4K-class RASTER_W regardless of how many days the
# profile spans; px/day = RASTER_W / n_days falls out of that. This is the right
# way round: a short profile (Maddy's ~200 days, or fewer) gets a dense raster
# (~19 px/day at 200 days) that stays crisp instead of a tiny image the browser
# upscales to mush, while a decade still gets ~1 px/day — plenty for any display
# up to 4K (wider just downscales, which is fine).
RASTER_SS = 3
RASTER_W = 3840


def encode_png(rgba: np.ndarray) -> str:
    """Encode an ``(h, w, 4)`` uint8 RGBA array as a ``data:image/png;base64``
    URI. Uses PNG colour-type 6 (truecolour + alpha), filter 0 on every row."""
    if rgba.dtype != np.uint8:
        rgba = rgba.astype(np.uint8)
    h, w = rgba.shape[:2]
    # Prepend the per-scanline filter byte (0 = None) to each row, vectorised.
    raw = np.zeros((h, w * 4 + 1), dtype=np.uint8)
    raw[:, 1:] = rgba.reshape(h, w * 4)

    def chunk(typ: bytes, data: bytes) -> bytes:
        return (struct.pack('>I', len(data)) + typ + data
                + struct.pack('>I', zlib.crc32(typ + data) & 0xffffffff))

    sig = b'\x89PNG\r\n\x1a\n'
    ihdr = struct.pack('>IIBBBBB', w, h, 8, 6, 0, 0, 0)
    idat = zlib.compress(raw.tobytes(), 9)
    png = sig + chunk(b'IHDR', ihdr) + chunk(b'IDAT', idat) + chunk(b'IEND', b'')
    return 'data:image/png;base64,' + base64.b64encode(png).decode('ascii')


def render_gradient_raster(fig, row, x_dates, env_lo, env_hi, column_colors, *,
                           y_range, h_px=480, layer='below', visible=True):
    """Add a gradient-filled envelope as a layout image on subplot ``row``.

    Parameters
    ----------
    fig : plotly figure (built by make_subplots; row N uses axes x{N}/y{N}).
    row : int, 1-based subplot row.
    x_dates : pd.DatetimeIndex aligned 1:1 with ``env_lo`` / ``env_hi``.
    env_lo, env_hi : array-like, the smoothed envelope edges in y-units; a
        column is fully transparent where either is non-finite or hi <= lo.
    column_colors : callable ``day_index -> (h_px, 3)`` uint8 array — the
        vertical colour ramp for a given day (pixel row 0 = top = highest y).
        Called once per day (cached); interpolated sub-columns use the nearest
        day's ramp. Only invoked for days with a finite envelope.
    y_range : (y0, y1) the data-y span the image covers vertically.
    h_px : vertical pixel resolution.
    layer : 'below' (default) or 'above' the traces.
    visible : initial visibility (the page toggle flips this).

    Returns the index of the added image in ``fig.layout.images``.

    To overlay a crisp trend line on an ``above``-layer raster, add a layout
    ``path`` shape with ``layer='above'`` (shapes render above above-images;
    a baked line would blur under the stretch-resize). See plot_qualitative_trends.
    """
    n = len(x_dates)
    y0, y1 = float(y_range[0]), float(y_range[1])
    lo = np.asarray(env_lo, dtype=float)
    hi = np.asarray(env_hi, dtype=float)

    Wo = RASTER_W if n > 1 else 1     # fixed width; px/day = RASTER_W / n_days
    S = RASTER_SS
    Hf, Wf = h_px * S, Wo * S

    # Supersampled band mask: interpolate the edges across sub-columns, test each
    # sub-pixel centre against [lo, hi], then box-average S x S into fractional
    # alpha. Averaging over BOTH axes anti-aliases every edge — the steep vertical
    # risers a vertical-only coverage left stairstepped. NaN edges (gaps) compare
    # False, so the band ends cleanly. Edge interpolation also makes the contour
    # a smooth diagonal between days.
    fj = np.linspace(0.0, n - 1, Wf) if n > 1 else np.zeros(Wf)
    d0 = np.floor(fj).astype(int)
    d1 = np.minimum(d0 + 1, n - 1)
    t = fj - d0
    lo_f = lo[d0] + (lo[d1] - lo[d0]) * t
    hi_f = hi[d0] + (hi[d1] - hi[d0]) * t
    ys = y1 - (np.arange(Hf) + 0.5) * (y1 - y0) / Hf
    inside = (ys[:, None] >= lo_f[None, :]) & (ys[:, None] <= hi_f[None, :])
    alpha = (inside.reshape(h_px, S, Wo, S).sum(axis=(1, 3), dtype=np.uint16)
             .astype(np.float32) / float(S * S))     # (h_px, Wo) in [0, 1]

    img = np.zeros((h_px, Wo, 4), dtype=np.uint8)
    img[:, :, 3] = np.rint(alpha * 255.0).astype(np.uint8)
    # Per-output-column colour from the nearest day (cached), only where the
    # column has any coverage. RGB is set for the full column so partial-alpha
    # edge pixels composite in the band colour (not a black fringe).
    cj = np.linspace(0.0, n - 1, Wo) if n > 1 else np.zeros(Wo)
    dcol = np.rint(cj).astype(int)
    ramp_cache = {}
    for c in np.nonzero(alpha.max(axis=0) > 0)[0]:
        d = int(dcol[c])
        r = ramp_cache.get(d)
        if r is None:
            r = column_colors(d)                     # (h_px, 3) uint8 for day d
            ramp_cache[d] = r
        img[:, c, :3] = r

    W = Wo
    uri = encode_png(img)
    axn = '' if row == 1 else str(row)
    x_left = x_dates[0]
    span_ms = float((x_dates[-1] - x_dates[0]) / np.timedelta64(1, 'ms'))

    fig.add_layout_image(dict(
        source=uri,
        xref=f'x{axn}', yref=f'y{axn}',
        x=x_left, y=y1,
        xanchor='left', yanchor='top',
        sizex=span_ms, sizey=(y1 - y0),
        sizing='stretch', layer=layer, visible=visible,
    ))
    return len(fig.layout.images) - 1
