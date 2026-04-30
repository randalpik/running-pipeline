#!/usr/bin/env bash
# Run the data pipeline.
# Default: refresh drive_snapshot from Drive, then build_dataset + parse_workouts.
# Flags:
#   --historical / -h   Run a full historical re-freeze before everything else.
#   --fit        / -f   Run the Bayesian CS fit at the end (slow, ~5-20 min).
#   --verbose    / -v   Show full output from every step (default: quiet).
#   --help              Show this help.
#
# The model fit always shows its progress regardless of -v; everything else
# is silenced by default and only printed if a step fails.
set -euo pipefail

cd "$(dirname "$0")/.."

historical=0
fit=0
verbose=0

for arg in "$@"; do
  case "$arg" in
    --historical|-h) historical=1 ;;
    --fit|-f)        fit=1 ;;
    --verbose|-v)    verbose=1 ;;
    --help)
      sed -n '2,11p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown argument: $arg" >&2
      echo "Usage: $0 [--historical|-h] [--fit|-f] [--verbose|-v]" >&2
      exit 2
      ;;
  esac
done

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
quiet_step "build_dataset"           python src/parsers/build_dataset.py
quiet_step "parse_workouts"          python src/parsers/parse_workouts.py

if [[ $fit -eq 1 ]]; then
  loud_step "bayes_cs_fit"  python src/models/bayes_cs_fit.py
fi

echo
echo "Pipeline complete."
