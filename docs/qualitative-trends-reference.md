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

## Time panel — canonical location & timezone

**Principle (since June 2026): the hand-logged `city_state` is the single source of truth for the Time panel's location and timezone. The watch supplies only the *absolute moment* of each run.** Its reported GPS and timezone offset are not trusted for the gradient.

Why: the watch's `tz_min` and `lat`/`lon` each fail on travel days, in ways that previously produced visible sunrise/clock anomalies on the Time band:
- **Failed-GPS home-default.** `daily_envelopes.py` stamps `HOME_LAT/LON` (Boulder) on any day with no usable GPS fix — including a failed-GPS outdoor run while traveling. Because the run still has a valid *time*, the pre-watch bin estimate never fired, so the day rendered at Boulder's solar times (e.g. 2025-10-27, 2025-08-05..07: logged Chicago, drawn Boulder).
- **Stale watch tz.** The watch can report the offset of the zone it last synced in, not where it is now (e.g. 2025-11-01 in Nashville reporting Eastern −240 instead of Central −300 — a +1 h jag in the sunrise band right before the genuine Nov-2 DST fall-back).

These are not solar-model bugs — only ~2/1885 watch days had a truly wrong `tz_min`, and the validated lesson is that the watch offset is essentially authoritative *except* on travel days, while the hand-logged city always knows where you were. Cross-checking the watch `tz_min` against a longitude→offset heuristic is **not** a valid verifier (it mis-flags Hawaii/Korea/EU/Canada and Eastern-zone cities near −84° lon — all no-DST or off-by-a-band); the authoritative check is the city's IANA zone via `zoneinfo`.

Mechanism (all in `plot_qualitative_trends.load_series`):
1. **`city_coords.csv` carries an IANA `tz` per city** (`America/Chicago`, …), auto-resolved from the geocoded lat/lon via keyless Open-Meteo (`timezone=auto`) at cache time — see `src/shared/geocoding.py::_timezone_for`. New cities get their zone for free during geocoding; legacy rows are backfilled. No new Python dependency (timezonefinder would pull a ~50 MB dataset).
2. **Watch era — `reproject_watch_to_canonical`.** `time_daily.csv` stores each run's local-clock minutes + the watch offset it was written with, so the absolute UTC instant is recoverable as `(date 00:00 in watch tz) + start_min`. That instant is projected into the canonical city's IANA zone (DST-correct) and coordinates: `start_min`/`end_min` become canonical local minutes, `lat`/`lon` the canonical city centroid, `tz_min` the canonical offset (which the solar gradient consumes). No change to `daily_envelopes.py` or the `time_daily.csv` schema was needed.
3. **Pre-watch era — `estimate_binned_time`.** Unified onto the same canonical IANA tz (was a longitude + US-DST heuristic).

Consequences and fallbacks:
- Every watch day's coordinates snap to the city centroid rather than the exact run GPS (sub-0.1° → a few seconds of solar difference, invisible).
- A day whose canonical city has no cached zone keeps the watch values (watch era) or the longitude+US-DST estimate (pre-watch) — graceful degradation only.
- Near-midnight runs that cross a local day boundary under the canonical offset are clamped to the day edge (rare; ≤ a few hours of shift).
- **Build ordering:** `run_plots.sh` runs `make_world_map` (whose `ensure_coords` populates lat/lon + tz) **before** `plot_qualitative_trends`, so a newly-added city's zone is cached the same run.

## Weight gap discipline

Weight has multi-day stretches without measurements. `WEIGHT_INTERP_MAX_GAP = 7` controls which gaps get linearly interpolated:
- Gaps < 7d: filled via linear interpolation in `weight_interp` series.
- Gaps ≥ 7d: stay as NaN.

For these long gaps, the trendline AND envelope both vanish on the same days. Implementation gotcha: must mask `lo_raw`/`hi_raw`/`ma` to `series.notna()` *before* the 7d smoothing pass, AND *again after*. The `min_periods=1` in the smoother lets it spread valid neighbour values back into the gap, so masking only once leaves the envelope extending past the trendline at gap boundaries.

## Hover — smart spikeline scaffold

All traces have `hoverinfo='skip'` so Plotly's native hover events don't fire. The plot opts into the shared cursor-tooltip scaffold (`_scaffold/cursor_tooltip.js`) via `cursor_tooltip=CursorTooltip(...)` on `render_plot()`. The scaffold owns the cursor-following spike (`.rp-spike`) and tooltip (`.rp-tooltip`); the plot supplies the per-day data and the `buildTooltip(day)` body.

- **Payload** is a precomputed daily-indexed structure (`P.ma`, `P.lo`, `P.hi` per metric) keyed by `day - first_day`, serialized as `window.__TT_DATA`.
- **`buildTooltip(day)`** looks up MA / lo / hi at the day index and emits one row per metric. Rows where all three are null are omitted (weight rows disappear before 2022 and during long gaps); if every metric is null at the cursor's day, returning `''` suppresses the tooltip entirely.
- **`spike_full_plot=True`** is set on `CursorTooltip` so the spikeline visually spans every stacked subplot rather than per-subplot clipping (default).

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
