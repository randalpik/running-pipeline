#!/usr/bin/env bash
# Render every plot from the existing data/ artifacts.
# Assumes daily.csv, races.csv, workout_decomposed.csv, and bayes_cs_*.csv
# are already present in data/. Run scripts/run_pipeline.sh first if not.
#
# Flags:
#   --diagnostics / -d   Also write per-plot diagnostic artifacts (currently:
#                        route_betas.csv from the recovery plot) into
#                        output/debug/. Off by default.
#   --verbose     / -v   Show full output from every plot script (default: quiet).
#   --help               Show this help.
set -euo pipefail

cd "$(dirname "$0")/.."

diagnostics=0
verbose=0

for arg in "$@"; do
  case "$arg" in
    --diagnostics|-d) diagnostics=1 ;;
    --verbose|-v)     verbose=1 ;;
    --help)
      sed -n '2,11p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown argument: $arg" >&2
      echo "Usage: $0 [--diagnostics|-d] [--verbose|-v]" >&2
      exit 2
      ;;
  esac
done

diag_flag=()
[[ $diagnostics -eq 1 ]] && diag_flag+=(--diagnostics)

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

quiet_step "bayes_cs_plot"           python src/plots/bayes_cs_plot.py
quiet_step "plot_training_quality"   python src/plots/plot_training_quality.py
quiet_step "plot_workouts"           python src/plots/plot_workouts.py
quiet_step "plot_long_runs"          python src/plots/plot_long_runs.py
quiet_step "plot_qualitative_trends" python src/plots/plot_qualitative_trends.py
quiet_step "make_recovery_plots"     python src/plots/make_recovery_plots.py "${diag_flag[@]}"
quiet_step "make_race_plots"         python src/plots/make_race_plots.py
quiet_step "make_geography_plot"     python src/plots/make_geography_plot.py
quiet_step "make_world_map"          python src/plots/make_world_map.py
quiet_step "dashboard"               python src/plots/dashboard.py

echo
echo "All plots written to output/."
