# Shoes reference

How a logged shoe becomes a physical pair with a mileage total. Owner:
`src/shared/shoes.py` (`build_shoes`). Consumers: the dashboard's current
training shoe (`src/plots/dashboard.py`) and the Shoes tab. Both read the same
`Shoes` object, so they cannot disagree about which pair a day belongs to or
how many miles it carries.

Read this before touching shoe roles, pair identity, or the spike/warm-up
mileage split.

## What the log gives us

One shoe per day (`daily.shoes`), Max-only, from 2020-01-01; every running day
since then has one. Watch-sourced profiles log no shoes, so `build_shoes`
returns an empty `Shoes` and both consumers hide their shoe output.

## Roles (per label)

| role | rule |
|---|---|
| spike | every day worn has daily `surface == 'Track'` |
| racer | < 50% of days worn are easy (`recovery` / `trail` run types) |
| trainer | everything else, trail/snow shoes included |

Measured Oct 2026: trainers have >= 61% easy days, racers <= 37%. "Never
raced in" is NOT the trainer rule: the Clifton 8 raced the 2023-03-04 Tom King
HM. `Saucony Endorphin 2` (2020-21 rose park track workouts and time trials)
is a spike by the rule, confirmed by Max.

XC is deliberately out of scope: Max owns no XC spikes any more and races 2-3
casual XC races a year in road shoes. XC race days are ordinary days.

**Trail shoes get no special treatment** — not detected, not hardcoded (Max,
Oct 2026: absolutely no hand list). Nothing in the log separates them from
daily trainers, and a trail shoe worn several days running is worth seeing on
the dashboard anyway.

## Physical pairs

One label = one pair, and an asterisk marks a distinct pair. From Oct 2026 Max
alternates asterisk / no asterisk on consecutive pairs of a model, even across
years, so a new pair can always be told apart from the old one.

The re-buys that predate that convention are hardcoded in `LEGACY_PAIR_STARTS`
(label -> first wear date of each later pair): Clifton 8 (2022-12-28), Speed 2
(2025-03-22), Speed 4 (2025-06-06, 2026-05-07). **Closed list — never extend
it, and never infer splits from usage gaps.** A start date inside a label's
worn span that isn't a logged wear prints a warning (the log changed under the
list).

The display name drops the asterisk and numbers pairs by first wear when a
model has more than one: the plain and asterisked Speed 4 labels together are
"Saucony Endorphin Speed 4 (pair 1-4)".

### Rejected: inferring pairs and trail shoes from usage (Oct 2026)

Each heuristic broke on real data, and Max's conclusion was that a shoe's
usage pattern says nothing reliable about what kind of shoe it is:

- The old dashboard rule (a pair ends after 14 recovery runs in another shoe)
  split the Speedgoat into 7 "pairs", split the Pro 2 around a winter
  Speedgoat stretch, and split two pairs Max bought early on sale and phased
  in slowly (Clifton 7's Sep 2021 tail, Speed 5's Dec 2025 break-in).
- A density threshold for trail shoes (share of running days in the shoe
  between first and last wear) was diluted by re-buys. A version that capped
  each absence at 60 runs still read the Speed 5's break-in as a trail shoe,
  so the dashboard showed the retired Speed 4 from 2026-02-16 to 03-05. A
  minimum-span exemption fixed that window but left other cases. The root
  problem: a break-in looks like a trail shoe and a snow stretch looks like a
  daily trainer.
- A hand list of trail shoes was rejected outright (see Roles).

## Mileage

`daily['miles']` as the caller passes it, i.e. display mileage
(`effective_daily_miles`). Each day's allocated miles add up exactly to that
day's miles.

- **Non-spike day:** the whole day to the logged shoe. Max never changes
  non-spike shoes mid-run; road racers are worn for warm-up and cool-down too.
- **Spike race day:** the spikes get the day's race distances (`races.csv`)
  plus `TRACK_TEMPO_MI` (0.8) only when a `rec/t` tempo is logged.
- **Spike workout day:** the spikes get exactly `quality_distance_m`, with no
  estimate of jogged rest.
- **The remainder** (warm-up, cool-down, jogs between races, doubles) goes to
  the trainer pair worn most recently *before* that day — never a trainer
  first worn later (a new trainer isn't debuted on a race warm-up).

If a spike day has no race rows or no `quality_distance_m`, or no prior
trainer exists, the build warns and leaves the whole day on the spike.

**Credit.** Inferred warm-up rows carry `logged=False`. They count as wear
(days worn, miles, a bar segment) but never earn PR or performance credit;
`Shoes.logged_pair()` gives the shoe that earns each day's results.

Oct 2026 effect (59 warm-up rows, 392.7 mi moved to trainers): ZoomX
Dragonfly 404 -> 148.5 mi, Endorphin 2 109 -> 44.4, Dragonfly 2 Elite 80 ->
29.0, New Balance spikes 33 -> 12.6.

## Dashboard: current training shoe

Unchanged detection: the model (asterisk stripped) worn on the latest 3
consecutive recovery runs. Its mileage is the allocated total of the pair worn
on the latest of those runs, warm-ups included. A trail shoe worn 3 recovery
runs in a row shows here by design (e.g. 2025-01-31, mid snow stretch: Hoka
Speedgoat 5, 309.9 mi).
