# Qualitative Trends Plot — Reference

`plot_qualitative_trends.py` produces `qualitative_trends.html`: three stacked subplots (volume, temperature, weight) showing moving-average trendlines on top of vertical-gradient min-max envelopes.

## Locked windows

| Metric | MA window | Min/max window |
|---|---|---|
| Volume | 56d | 14d |
| Temp   | 28d | 14d |
| Weight | 56d | 14d |

The 14d rolling min/max gets a second 7d centered MA pass for visual smoothness. Hover values are the *raw* 14d min/max (not the smoothed values).

## Rendering: vertical-gradient envelope

Each subplot's envelope is built from N=40 opaque horizontal polygon strips. For strip k:
1. y-band: `[edges[k] - overlap, edges[k+1] + overlap]` where `edges = linspace(y_min, y_max, N+1)`.
2. At each x: `lo[x] = max(strip_bottom, lo_smooth[x])`, `hi[x] = min(strip_top, hi_smooth[x])`. Where `lo >= hi`, both become NaN.
3. Polygon path: walks each contiguous valid x-region, forward along hi then back along lo, with `None` separators for multi-region strips (e.g. temp's 9°C band crosses spring + fall every year → ~30 sub-polygons in one trace).
4. Color: `gradient_at(strip_y_midpoint)`, rendered as opaque rgb.

### Why this approach (and not alternatives we tried)

**Why opaque, not alpha**: With α<1 strip rendering and overlap > 0, the overlap zone has effective α ≈ 1 - (1-α)² ≈ 0.80 for α=0.55, producing visibly darker hairline bands at every strip seam. Pre-compositing the alpha against the bg (`#1a1a1a`) gives identical pixel values to alpha rendering — same washed-out gradient. Pure opaque colors at intentionally darker shades is the only way to get *both* vivid colors *and* no seam artifacts.

**Why a strip stack, not a Heatmap**: Tried `go.Heatmap` with NaN cells outside the envelope. Cells are inherently rectangular, so envelope edges came out stepwise even at high y-resolution. The strip stack uses smooth `lo_smooth`/`hi_smooth` curves to define each strip's clipping boundary, so envelope edges are smooth diagonals.

**Why not SVG `<linearGradient>` injection**: Tried post-render JS that attached a vertical SVG linearGradient to a `fill='tonexty'` envelope path. Plotly's React-based DOM rebuilds invalidated the injection on every redraw. Strip stack is more robust because everything goes through plotly's normal trace mechanism.

**Why `fill='toself'` polygons, not chained `fill='tonexty'` pairs**: Earlier version used pairs of Scatter traces (lo and hi) chained with `tonexty`. With many NaN-broken regions, plotly bridged across gaps creating spurious horizontal connectors. Single-trace `fill='toself'` with explicit polygon paths and `None` separators is unambiguous.

## Gradients

Two-stop gradients looked monochromatic for volume and weight because each metric's envelope at any given x spans only ~25% of the absolute gradient range. The visible color variation across that 25% slice is too small to register as a gradient.

Fix: 3-stop gradients with the middle stop placed at the trendline's typical value, which puts a dramatic color shift across the typical envelope range:

| Metric | Anchors |
|---|---|
| Volume | 0 #3D2208 → 8 #8B5C16 → 28.8 #F2D034 |
| Temp   | -10 #0E7BAA → 22 #5A9E3D → 40 #C82020 (Excel defaults darkened ~30%) |
| Weight | 145 #2D1006 → 156 #8C4F1F → 170 #E89535 |

White trendline contrasts WCAG-AA across the full range of every gradient.

## Weight gap discipline

Weight has multi-day stretches without measurements. `WEIGHT_INTERP_MAX_GAP = 7` controls which gaps get linearly interpolated:
- Gaps < 7d: filled via linear interpolation in `weight_interp` series.
- Gaps ≥ 7d: stay as NaN.

For these long gaps, the trendline AND envelope both vanish on the same days. Implementation gotcha: must mask `lo_raw`/`hi_raw`/`ma` to `series.notna()` *before* the 7d smoothing pass, AND *again after*. The `min_periods=1` in the smoother lets it spread valid neighbour values back into the gap, so masking only once leaves the envelope extending past the trendline at gap boundaries.

## Custom hover (TQ pattern)

All traces have `hoverinfo='skip'` so plotly's native hover events don't fire. Instead, a `mousemove` listener on the plot div:
1. Reads `pdiv._fullLayout.xaxis.p2c(pixel_x)` to convert pixel → ms timestamp → day index.
2. Looks up MA, lo, hi at that index in pre-built JSON payload (`P.ma`, `P.lo`, `P.hi`).
3. Renders a fixed-position tooltip + a 1px full-viewport spike line transformed to the mouse X.
4. Rows where `ma`, `lo`, AND `hi` are all null are hidden entirely (so weight rows disappear before 2022 and during long gaps; the whole tooltip disappears if all three metrics are null).

Range format: `(155.0 to 162.4)` instead of an en-dash, for readability with negative numbers (especially temp).

## Knobs

| Constant | Value | Notes |
|---|---|---|
| `MA_WINDOW` | `{volume: 56, temp: 28, weight: 56}` | per-metric, locked |
| `RANGE_WINDOW` | 14 | rolling min/max window |
| `RANGE_SMOOTH` | 7 | post-pass MA on min/max |
| `N_ENVELOPE_STRIPS` | 40 | gradient resolution |
| `ENVELOPE_X_STRIDE` | 4 | envelope downsample (smoothed → no visual loss) |
| `ENVELOPE_ALPHA` | 1.0 | opaque (alpha at 1 means no compositing) |
| `STRIP_OVERLAP_FRAC` | 0.005 | small overlap to prevent sub-pixel AA gaps |
| `WEIGHT_INTERP_MAX_GAP` | 7 | gaps shorter than this get linearly interpolated |

## Output

- `./output/qualitative_trends.html` (~7.5 MB)
- 123 plotly traces total (40 envelope strips × 3 subplots + 1 trendline × 3 = 123)
