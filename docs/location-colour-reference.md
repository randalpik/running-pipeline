# Location colour reference

How a state/country gets its colour on the **Locations** and **World Map**
tabs. Owner: `src/plotting/location_palette.py`. Guard:
`python scripts/check_location_palette.py`.

Read this before touching the palette module, the candidate grid, the
qualification rule, or `get_state`/`group_key`.

## What is hand-chosen

Two things, and nothing else:

| | value |
|---|---|
| Anchor bases | `WA #8617be`, `TN #17be33`, `CO #be6b17`, `IL #17a2be` |
| `PERCEPTUAL_STEP` | 5.0 dE between adjacent shades within a group |

The anchor bases are frozen and render byte-identical to the hand-tuned
palette this replaced. Every other colour is derived.

## The three stability properties

They are the whole point, and they interact — a change that satisfies two
and breaks the third is the usual failure.

**Stable across tabs.** Locations and the World Map are separate processes
(`scripts/run_plots.sh` invokes each as its own `python`), so nothing may
depend on `hash()` — CPython salts string hashing per process. This was the
original bug: `'CN'` missed the old hue table and fell through to
`hash(state) % 360`, so China's colour differed per build *and* between the
two tabs while every other group stayed put.

**Stable across profiles.** Ordering is derived from the **union** of every
`prod` profile's data, read directly from `Profile.data_dir` rather than the
ambient `RP_DATA_DIR`, so the answer does not depend on which profile is
being built. Group keys pass through `country_abbrev` first, so a
watch-sourced profile's `Osaka, Japan` and a Drive-sourced profile's
`Osaka, JP` are one group with one colour.

**Stable over time.** Groups are ordered by the date they first cleared
`QUALIFY_THRESHOLD` (26.2) miles inside a calendar year. That date is a fact
about the past and cannot change retroactively, and a newly qualifying group
always qualifies *now* — so it appends to the end of the order. Placement is
greedy in that order: group *k* is placed against groups `< k` only, and
therefore never moves a colour already in use.

Greedy placement is myopic — a globally optimised palette would be tighter —
but a globally optimised palette cannot be grown without reshuffling, which
is the property being bought.

## Reserved tracks

Each group reserves `RESERVED_SHADES` (4) colours up front: the base plus
the lighter shades its future cities will use. Candidate tracks are scored
against **every shade of every group already placed**, not just their bases —
two groups with well-separated bases still collide once both fan out into
their ramps, which is how the old palette put Verbier CH within 4.97 dE of
Ann Arbor MI.

Reserving the whole track is what makes growth free: a group gaining its
second, third or fourth city fills a slot that was already held for it.

**Slot fill order is `0, 3, 1, 2`** (`trace_slots()`) — extremes before the
middle, so a group holding fewer traces than it reserved spreads across the
track instead of crowding its dark end:

| traces | slots | adjacent separation |
|---|---|---|
| 2 | 0, 3 | ~15 dE |
| 3 | 0, 1, 3 | ~5, ~10 dE |
| 4 | 0, 1, 2, 3 | ~5 dE |

`(other)` always takes the lightest of the chosen slots, so the residual
bucket never reads heavier than a real city.

## Measured

| | min inter-group dE |
|---|---|
| hand-tuned palette this replaced (reserved nothing) | 3.67 |
| hue-only rotation, full reservation | 3.17 |
| **shipped: 3-D packing, full reservation** | **5.16** |

The gain comes from using all of colour space. Fixing base lightness and
saturation confines every track to one thin cylindrical shell, leaving hue
as the only free axis — 360° divided N ways. Letting base lightness and
chroma vary too means two groups can share a hue and still read as plainly
different.

`PERCEPTUAL_STEP` is capped at 6 by the gamut: the frozen anchor bases
stepped up at constant chroma leave sRGB past that (TN starts at OKLab
L 0.698 and hits the ceiling first). Step 6 drops inter-group separation to
3.91, so 5 is the best feasible point. The lever if a larger step is ever
wanted is tapering chroma on the upper shades.

## Deliberate: cities within a state re-shade

Cities inside a group are ranked by **lifetime mileage**, so two can swap
shades as mileage accrues, and because the last trace sits at the light
extreme, a group gaining a trace pulls it inward (IL's Highland Park would
go slot 3 → 1). This is **intended, not a gap** — the colour carries *state*
identity, and mileage order is the more useful ranking within a state,
especially now that within-group shades are close. Ranking cities by
qualification date instead would pin them permanently; it was considered and
rejected.

## Failure mode

A group with no palette entry renders gray, with a printed warning. This can
only happen for a non-`prod` profile holding a group the canonical union
never saw (the `coros` test profile's `KR`, for instance). Inventing a local
colour there would break the cross-tab and cross-profile guarantee, so gray
is the honest answer.

## Gotcha

The candidate grid (`_HUE_STEP`, `_L_RANGE`, `_C_RANGE`) is **part of the
palette's definition**, not a tuning knob — changing it re-derives every
non-anchor colour. The guard's separation floor will catch a degradation but
not a reshuffle, so re-read the assignment after touching it.
