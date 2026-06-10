# Running pipeline

Personal training-log pipeline. Parses Max's daily running log out of Google Drive into analysis-ready CSVs and an interactive site at **[running.maxrandalmusic.com](https://running.maxrandalmusic.com)**.

The site is gated by Google OAuth + an allowlist; deployments run automatically on every Workout-column edit in the source spreadsheet, via GitHub Actions → Netlify.

## Quickstart

```bash
# Install
pip install -r requirements.txt              # data + plots
pip install -r requirements-fit.txt          # adds PyMC/ArviZ for the Bayesian CS fit

# Build data → output/
./scripts/run_pipeline.sh                    # refresh + parse, ~30s
./scripts/run_pipeline.sh --fit              # also rerun the Bayesian CS fit (~5-20m)

# Build all plots → output/*.html
./scripts/run_plots.sh

# Iterate on plots locally with livereload
python scripts/dev.py                        # http://localhost:5500
```

`scripts/dev.py` watches `src/plots/*.py` and reruns only the affected plot on save (plus any plots that import a changed shared helper). Edits to `src/plotting/**` or `data/*.csv` rerun everything.

## Profiles (multi-runner builds)

`scripts/run_pipeline.sh` builds **only the default profile** (Max, sourced from Drive). Additional runners — e.g. Maddy (Coros) — are built by `scripts/build_profiles.py`, which routes each profile to its own data source, renders its plots, and stages it under `site/dist/profiles/<id>/`. Profiles are declared in `src/profiles.py`.

```bash
# Build one profile end-to-end (data sync → dataset → plots → stage)
python scripts/build_profiles.py --only maddy            # reuse cached CS fit
python scripts/build_profiles.py --only maddy --fit      # also rerun the Bayesian CS fit
python scripts/build_profiles.py --fit                   # every profile, with fit
```

Coros profiles need their watch credentials in the environment (`COROS2_EMAIL` / `COROS2_PASSWORD` for Maddy — see `src/profiles.py`) and the race-sheet sync needs the Drive service account (`GOOGLE_APPLICATION_CREDENTIALS`); without them the profile is skipped cleanly. Locally, put credentials in the gitignored `.env` at the repo root (`KEY=VALUE` lines, loaded by `src/shared/env.py`; real environment variables take precedence) — CI supplies the same names via GHA repo secrets.

To **re-run only the CS fit** for one profile against its existing local data — no Drive/Coros sync, no credentials, useful for verifying a model change — point the fit at that profile's data dir:

```bash
RP_DATA_DIR=data/profiles/maddy python src/models/bayes_cs_fit.py
```

It reads that dir's `races.csv` + `daily.csv` and writes its `bayes_cs_*.csv` back into it. `RP_DATA_DIR` / `RP_OUTPUT_DIR` reroute every script per profile (see `src/shared/paths.py`), so the same flag works for plots too (e.g. `RP_DATA_DIR=data/profiles/maddy RP_OUTPUT_DIR=output/profiles/maddy python src/plots/make_recovery_plots.py`).

## Repo layout

```
data/         truth-source CSVs (daily.csv, races.csv, bayes_cs_*.csv, drive_snapshot.csv)
src/
  parsers/    Drive → CSV pipeline (build_dataset, parse_workouts, …)
  shared/     analysis helpers shared across plots
  models/     Bayesian CS fit (bayes_cs_fit.py)
  plotting/   shared rendering layer — see CLAUDE.md § "Plot conventions"
  plots/      one .py per plot (+ sibling .js / .css for plot-specific behavior)
scripts/      run_pipeline.sh, run_plots.sh, dev.py, sheets_trigger.gs
output/       generated HTML + plotly.min.js (copied to site/dist/ at deploy)
site/         Netlify functions, edge gate, deployed dist
docs/         reference docs per analytical layer (see map below)
.github/      build-and-deploy.yml CI workflow
```

## Where to find what

| If you want to … | Read |
|---|---|
| Understand the project end-to-end | `docs/ARCHITECTURE.md` |
| Work on plot code (CSS/JS conventions, widgets, scaffold) | `CLAUDE.md` § "Plot conventions" |
| Understand the CS Bayesian fit | `docs/cs-model-reference.md` |
| Understand training-quality residuals | `docs/training-quality-reference.md` |
| Understand the recovery-runs regression | `docs/recovery-runs-reference.md` |
| Understand the locations-sheet flow | `docs/route-normalization-reference.md` |
| Understand the volume/temp/weight panel | `docs/qualitative-trends-reference.md` |
| Understand the schema, conventions, and workout-string coding | `CLAUDE.md` |
| See the live deployment, auth model, and CI triggers | `docs/ARCHITECTURE.md` § "Hosting layer" + `CLAUDE.md` § "Hosting / auth model" |

## Pipeline triggers

Three ways the pipeline runs:

| Trigger | When |
|---|---|
| `repository_dispatch` (`pipeline-run`) | Auto, on every Workout-column edit in the source workbook (60s debounce via `scripts/sheets_trigger.gs`) |
| `repository_dispatch` (`pipeline-run-fit`) | Auto, additionally on race days (workout text contains `race@`) |
| `workflow_dispatch` | Manual via the GitHub UI or the admin button on the live site |

All three flow into `.github/workflows/build-and-deploy.yml`.

## License

Personal project. All data is Max's. Code is shared as-is, no warranty.
