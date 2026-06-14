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
# Watch-derived rep extraction for Max's hybrid profile: the hand log stays
# the source of truth; reps.py reconciles the Coros per-second stream against
# it (src/coros/reps.py) and parse_workouts consumes the result. Skipped when
# the rich details cache or CS timeline isn't present (e.g. CI before the
# coros sync is wired) — parse_workouts then falls back to string parsing.
if [[ -d data/profiles/coros/details && -f data/bayes_cs_summary.csv ]]; then
  quiet_step "reps"                  python src/coros/reps.py --details-dir data/profiles/coros/details
fi
# Long-run + recovery watch reconciliation: measurement rows for both run
# types + the log/watch distance calibration (src/coros/long_runs.py). Same
# graceful degradation as reps — without the details cache,
# project_long_runs and the recovery fit fall back to logged values plus
# the route-era mislogged-distance rules.
if [[ -d data/profiles/coros/details ]]; then
  quiet_step "long_runs (watch)"     python src/coros/long_runs.py --details-dir data/profiles/coros/details
fi
quiet_step "parse_workouts"          python src/parsers/parse_workouts.py "${diag_flag[@]}"

if [[ $fit -eq 1 ]]; then
  loud_step "bayes_cs_fit"  python src/models/bayes_cs_fit.py "${diag_flag[@]}"
fi

echo
echo "Pipeline complete."
