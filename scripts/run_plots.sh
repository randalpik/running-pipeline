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
no_shell=0
keep_going=0

for arg in "$@"; do
  case "$arg" in
    --diagnostics|-d) diagnostics=1 ;;
    --verbose|-v)     verbose=1 ;;
    --no-shell)       no_shell=1 ;;
    --keep-going)     keep_going=1 ;;
    --help)
      sed -n '2,11p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown argument: $arg" >&2
      echo "Usage: $0 [--diagnostics|-d] [--verbose|-v] [--no-shell] [--keep-going]" >&2
      exit 2
      ;;
  esac
done

diag_flag=()
[[ $diagnostics -eq 1 ]] && diag_flag+=(--diagnostics)

# quiet_step: hide a plot's output unless --verbose. On failure, dump the log;
# with --keep-going (used for sparse non-default profiles) a failed plot is
# skipped — its HTML simply won't exist, so the shell auto-hides that tab —
# rather than aborting the whole build.
quiet_step() {
  local label="$1"; shift
  echo "==> $label"
  local t0=$SECONDS
  if [[ $verbose -eq 1 ]]; then
    if ! "$@"; then
      echo "FAILED: $label" >&2
      [[ $keep_going -eq 1 ]] && { echo "    (skipped, --keep-going)"; return 0; }
      exit 1
    fi
  else
    local tmplog
    tmplog=$(mktemp)
    if ! "$@" >"$tmplog" 2>&1; then
      echo "FAILED: $label" >&2
      cat "$tmplog" >&2
      rm -f "$tmplog"
      [[ $keep_going -eq 1 ]] && { echo "    (skipped, --keep-going)"; return 0; }
      exit 1
    fi
    rm -f "$tmplog"
  fi
  echo "    done in $((SECONDS - t0))s"
}

# Training quality runs FIRST: it persists training_quality_corpus.csv,
# which bayes_cs_plot consumes for the performance-frontier line.
quiet_step "plot_training_quality"   python src/plots/plot_training_quality.py
quiet_step "bayes_cs_plot"           python src/plots/bayes_cs_plot.py
quiet_step "plot_workouts"           python src/plots/plot_workouts.py
quiet_step "plot_long_runs"          python src/plots/plot_long_runs.py
# World map first: its ensure_coords() call geocodes any new cities AND resolves
# their IANA timezone into data/city_coords.csv, which qualitative_trends then
# reads to drive the Time panel's canonical-tz solar gradient (same run, no lag).
quiet_step "make_world_map"          python src/plots/make_world_map.py
quiet_step "plot_qualitative_trends" python src/plots/plot_qualitative_trends.py
# Standalone full-page versions of each Misc. Trends panel (linked from the
# panel ↗ insets; not shell tabs). Auto-staged to site/dist root + auto-gated.
for panel in conditions temp humidity wind volume altitude time weight; do
  quiet_step "plot_qualitative_trends --panel $panel" \
    python src/plots/plot_qualitative_trends.py --panel "$panel"
done
quiet_step "make_annual_plot"        python src/plots/make_annual_plot.py
quiet_step "make_recovery_plots"     python src/plots/make_recovery_plots.py "${diag_flag[@]}"
quiet_step "make_race_plots"         python src/plots/make_race_plots.py
quiet_step "make_geography_plot"     python src/plots/make_geography_plot.py
quiet_step "dashboard"               python src/plots/dashboard.py
# The profile-aware orchestrator (build_profiles.py) builds the shell itself
# with the profile switcher; --no-shell skips this single-profile default.
if [[ $no_shell -eq 0 ]]; then
  quiet_step "shell (with admin tab)"  python src/plots/build_shell.py --admin
fi

echo
echo "All plots written to ${RP_OUTPUT_DIR:-output}/."
