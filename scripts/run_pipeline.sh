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
# so this order is safe. When the fit runs, reps does a full re-extract
# (--full-regen safety valve) to rebuild against the new CS; otherwise reps is
# incremental against the watch_activities presence cache.
reps_regen=()
if [[ $fit -eq 1 ]]; then
  loud_step "bayes_cs_fit"  python src/models/bayes_cs_fit.py "${diag_flag[@]}"
  reps_regen=(--full-regen)
fi
# Watch-derived rep extraction: reconciles the Coros per-second stream against
# the hand log (parse_workouts consumes the result). Incremental via the
# watch_activities index. Skipped when the details cache or CS timeline is
# absent — parse_workouts then falls back to string parsing.
if [[ -d data/profiles/coros/details && -f data/bayes_cs_summary.csv ]]; then
  quiet_step "reps"                  python src/coros/reps.py --details-dir data/profiles/coros/details "${reps_regen[@]}"
fi
quiet_step "parse_workouts"          python src/parsers/parse_workouts.py "${diag_flag[@]}"

echo
echo "Pipeline complete."
