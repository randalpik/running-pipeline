#!/usr/bin/env bash
# Run the data pipeline.
# Default: refresh drive_snapshot from Drive, then build_dataset + parse_workouts.
# Flags:
#   --historical  / -h   Run a full historical re-freeze before everything else.
#   --fit         / -f   Run the Bayesian CS fit at the end (slow, ~5-20 min).
#   --diagnostics / -d   Also write diagnostic-only artifacts (workout_pruned,
#                        bayes_cs_residuals/posterior/diagnostics) into
#                        output/debug/. Off by default — these are not
#                        consumed by the plot pipeline.
#   --verbose     / -v   Show full output from every step (default: quiet).
#   --help               Show this help.
#
# The model fit always shows its progress regardless of -v; everything else
# is silenced by default and only printed if a step fails.
set -euo pipefail

cd "$(dirname "$0")/.."

historical=0
fit=0
diagnostics=0
verbose=0

for arg in "$@"; do
  case "$arg" in
    --historical|-h)  historical=1 ;;
    --fit|-f)         fit=1 ;;
    --diagnostics|-d) diagnostics=1 ;;
    --verbose|-v)     verbose=1 ;;
    --help)
      sed -n '2,14p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown argument: $arg" >&2
      echo "Usage: $0 [--historical|-h] [--fit|-f] [--diagnostics|-d] [--verbose|-v]" >&2
      exit 2
      ;;
  esac
done

diag_flag=()
[[ $diagnostics -eq 1 ]] && diag_flag+=(--diagnostics)

# quiet_step: hide stdout/stderr unless --verbose; on failure, dump the captured
# log so the error is still visible. Use for chatty steps where progress is noise.
quiet_step() {
  local label="$1"; shift
  echo "==> $label"
  local t0=$SECONDS
  if [[ $verbose -eq 1 ]]; then
    "$@"
  else
    local tmplog
    tmplog=$(mktemp)
    if ! "$@" >"$tmplog" 2>&1; then
      echo "FAILED: $label" >&2
      cat "$tmplog" >&2
      rm -f "$tmplog"
      exit 1
    fi
    rm -f "$tmplog"
  fi
  echo "    done in $((SECONDS - t0))s"
}

# loud_step: never suppresses output. Use for steps whose progress is the signal
# (the bayes fit's NUTS sampler bar).
loud_step() {
  local label="$1"; shift
  echo "==> $label"
  local t0=$SECONDS
  "$@"
  echo "    done in $((SECONDS - t0))s"
}

if [[ $historical -eq 1 ]]; then
  quiet_step "freeze_historical"  python src/parsers/freeze_historical.py
fi

quiet_step "drive_fetch (snapshot)"  python src/parsers/drive_fetch.py snapshot --year "$(date +%Y)"
# Unified watch-derived per-day table (incremental, presence-based; parses only
# new activities). Projects weather_measured.csv, which build_dataset joins.
# Must precede build_dataset. Skipped when the cache is absent (graceful no-op).
if [[ -d data/profiles/coros/details ]]; then
  quiet_step "watch_daily"           python -m src.coros.watch_daily
fi
quiet_step "build_dataset"           python src/parsers/build_dataset.py
# Long-run + recovery watch reconciliation: measurement rows + the log/watch
# distance calibration. Reads the precomputed watch_daily.csv (no per-second
# parse); doesn't depend on CS, so it runs before the fit. Without the details
# cache, project_long_runs/recovery fall back to logged values + route-era rules.
if [[ -d data/profiles/coros/details ]]; then
  quiet_step "long_runs (watch)"     python src/coros/long_runs.py --details-dir data/profiles/coros/details
fi
# CS fit BEFORE reps: reps reconciles against the CS timeline, so on a refit
# (e.g. a new race added) it must see the fresh CS. The fit only needs races,
# so this order is safe. When the fit runs, the watch producers do a full
# re-extract (--full-regen safety valve) so they refresh against the new CS /
# calibration; otherwise they're incremental against their presence caches.
watch_regen=()
if [[ $fit -eq 1 ]]; then
  loud_step "bayes_cs_fit"  python src/models/bayes_cs_fit.py "${diag_flag[@]}"
  watch_regen=(--full-regen)
fi
# v_max CS-multiples (k_evid, k_pred) derived from this profile's races + CS
# fit -> data/vmax_ratios.csv, which cs_projection reads (the hardcoded
# registry / defaults are only the fallback). Mirrors build_coros_data: after
# the fit, before reps/parse_workouts (workout deflation uses k_evid). Without
# this step the max profile's production builds silently ran on the frozen
# registry snapshot instead of the live calibration.
if [[ -f data/bayes_cs_summary.csv ]]; then
  quiet_step "calibrate_vmax"        python scripts/calibrate_vmax.py --write
fi
# Watch-derived rep extraction: reconciles the Coros per-second stream against
# the hand log (parse_workouts consumes the result). Incremental via the
# watch_activities index. Skipped when the details cache or CS timeline is
# absent — parse_workouts then falls back to string parsing.
if [[ -d data/profiles/coros/details && -f data/bayes_cs_summary.csv ]]; then
  quiet_step "reps"                  python src/coros/reps.py --details-dir data/profiles/coros/details "${watch_regen[@]}"
fi
# Elevation enrichment (gain/loss, Minetti grade, per-mile splits). Incremental
# via the watch_activities index + day-level failure memo. The barometric path
# needs no network (the sync caches outdoor runs rich), but the DEM augment
# looks GPS tracks up against the public DEM API — a warm run is cache-served
# and instant, the one-time cold seed runs for hours at 1 req/s.
# NO --full-regen on fit runs (Aug 2026): elevation depends on nothing a CS fit
# changes (verified byte-equivalent under full regen); the pinned distance
# calibration is its only soft input, and a calibration ADOPTION self-heals via
# the corrected-distance staleness check inside backfill_elevation.
# loud_step (not quiet): during the cold seed its incremental-save progress IS
# the signal, so stream it live; warm runs print only a few summary lines.
if [[ -f data/watch_activities.csv ]]; then
  loud_step "elevation"              python scripts/backfill_elevation.py
fi
# Live calibration of the fractional grade engine (k_up / refund curve) from
# the freshly-written mile splits; consumers read data/elevation_calibration.csv
# via elevation_cost.engine_params(). Thin corpora write defaults (no-op safe).
if [[ -f data/elevation_splits.csv ]]; then
  quiet_step "calibrate_climb"       python scripts/calibrate_climb.py
fi
# Cross-profile physical-beta ratios (footing/altitude as fractions of pace)
# from this profile's pooled fit — consumed by label-less watch profiles via
# recovery_model.shared_beta_ratios. After calibrate_climb (the betas consume
# the fresh grade calibration); refuses to export a non-genuine fit.
if [[ -f data/bayes_cs_summary.csv ]]; then
  quiet_step "calibrate_physical"    python scripts/calibrate_physical.py
fi
# Per-day altitude + local-time envelopes for the Misc. Trends plot, from the
# rich detail cache (no network). Same lifecycle as elevation_measured.csv:
# built wherever the details cache is present, gitignored, and the plot renders
# those panels empty when the CSVs are absent.
if [[ -d data/profiles/coros/details && -f data/watch_activities.csv ]]; then
  quiet_step "daily_envelopes"       python -m src.coros.daily_envelopes
fi
quiet_step "parse_workouts"          python src/parsers/parse_workouts.py "${diag_flag[@]}"

echo
echo "Pipeline complete."
