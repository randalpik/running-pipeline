#!/usr/bin/env python3
"""Build every dashboard profile and stage site/dist.

For each profile in src/profiles.py:
  1. build its dataset into the profile's data dir
       - source="drive": scripts/run_pipeline.sh (Drive -> build_dataset)
       - source="coros": incremental Coros sync -> snapshot -> build_dataset
  2. render all plots into the profile's output dir (scripts/run_plots.sh)
  3. write the profile-aware shell (tab bar + profile switcher)
  4. copy the output dir into site/dist (default profile -> root, others -> /<id>/)

The whole pipeline reroutes per profile purely via the RP_DATA_DIR /
RP_OUTPUT_DIR env vars (see src/shared/paths.py), so the plot and parser
scripts need no profile awareness.

Flags:
  --only a,b        build only these profile ids
  --skip-data       reuse existing datasets (just plots + shell + stage)
  --rebuild-coros   reprocess Coros profiles from the cached details
  --fit             run the Bayesian CS fit (default off reuses cached outputs)
  --historical      freeze historical data before the Drive build
  --no-stage        don't copy into site/dist
  --strict          fail the build if any plot fails (default: keep going)

RP_PRODUCTION=1 in the environment hides profiles flagged prod=False (the
Coros test profile) from the build and the switcher.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.parsers import snapshot as snap          # noqa: E402
from src.plots.build_shell import write_index      # noqa: E402
from src.profiles import PROFILES, get_profile     # noqa: E402
from src.shared.env import load_env_file           # noqa: E402
from src.shared.paths import REPO_ROOT             # noqa: E402

load_env_file()

SITE_DIST = REPO_ROOT / "site" / "dist"


class ProfileSkip(Exception):
    """Raised to skip a profile (e.g. credentials not yet provided)."""


def _is_available(profile):
    """Whether a profile can currently be built (and so belongs in the
    profile switcher). Drive profiles always; Coros profiles only once their
    credentials are present in the environment. Profiles flagged ``prod=False``
    (e.g. the Coros test profile) are excluded when RP_PRODUCTION is set."""
    if not profile.prod and os.environ.get("RP_PRODUCTION"):
        return False
    if profile.source == "coros":
        cfg = profile.coros
        return bool(os.environ.get(cfg.get("email_env", "COROS_EMAIL"))
                    and os.environ.get(cfg.get("password_env", "COROS_PASSWORD")))
    return True


def _env_for(profile):
    env = os.environ.copy()
    env["RP_DATA_DIR"] = str(profile.data_dir)
    env["RP_OUTPUT_DIR"] = str(profile.output_dir)
    env["RP_PROFILE"] = profile.id
    return env


def _run(cmd, env, label):
    print(f"\n=== [{label}] {' '.join(str(c) for c in cmd)} ===", flush=True)
    subprocess.run(cmd, env=env, cwd=REPO_ROOT, check=True)


def build_drive_data(profile, env, *, fit=False, historical=False):
    cmd = ["./scripts/run_pipeline.sh"]
    if fit:
        cmd.append("--fit")
    if historical:
        cmd.append("--historical")
    _run(cmd, env, f"{profile.id}: pipeline")


def sync_coros_cache(profile, *, rebuild):
    """Sync a Coros profile's detail cache + current_log and return the log df.

    The detail cache (``<data_dir>/details``) is the shared resource that both
    the watch profile's own build AND the Max drive build's enrichment read
    (run_pipeline.sh reads data/profiles/coros/details). ``--sync-only`` calls
    just this — populating the cache without rendering the profile — so CI can
    refresh Max's watch cache without building the hidden coros test profile."""
    from src.coros.sync import sync_current_log

    cfg = profile.coros
    email = os.environ.get(cfg.get("email_env", "COROS_EMAIL"))
    password = os.environ.get(cfg.get("password_env", "COROS_PASSWORD"))
    if not email or not password:
        raise ProfileSkip(f"missing Coros credentials "
                          f"({cfg.get('email_env', 'COROS_EMAIL')} / "
                          f"{cfg.get('password_env', 'COROS_PASSWORD')})")

    data_dir = profile.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    return sync_current_log(
        email=email, password=password, region=cfg.get("region", "us"),
        current_log_path=data_dir / "coros_current_log.csv",
        details_dir=data_dir / "details",
        token_cache=data_dir / "coros_token.json", rebuild=rebuild,
        start_day=cfg.get("start_day"),
    )


def build_coros_data(profile, env, *, rebuild, fit=False):
    cfg = profile.coros
    data_dir = profile.data_dir
    df = sync_coros_cache(profile, rebuild=rebuild)

    # Assemble a snapshot. current_log carries the data; the reverse-geocoded
    # "City, ST" already in its `location` column is also the city_state, so we
    # synthesize an identity `locations` section (log_location -> city_state)
    # to drive the location-metadata join (Locations / World Map). Races come
    # from `races_csv` (the watch feed has no race flags) as additions.
    locations_df = pd.DataFrame({
        "log_location": sorted(df["location"].dropna().unique()),
    })
    # city_state and display_name both default to the reverse-geocoded
    # "City, ST" — the watch feed has no finer location metadata, and some
    # plots (e.g. workouts via project_workouts) require display_name.
    locations_df["city_state"] = locations_df["log_location"]
    locations_df["display_name"] = locations_df["log_location"]

    additions_df = _race_additions(cfg)
    print(f"[{profile.id}] race additions: {len(additions_df)}")

    snapshot_path = data_dir / "drive_snapshot.csv"
    current_year = int(pd.to_datetime(df["date"]).dt.year.max())
    empty = pd.DataFrame()
    snap.write_snapshot(
        str(snapshot_path), current_year=current_year, current_log_df=df,
        changes_df=empty, additions_df=additions_df, locations_df=locations_df,
        hills_df=empty, coordinates_df=empty, historical_df=empty,
    )
    print(f"[{profile.id}] wrote snapshot {snapshot_path}")

    _run(["python", "src/parsers/build_dataset.py", "--no-historical",
          "--no-fetch", "--snapshot", str(snapshot_path),
          "--out-dir", str(data_dir), "--current-year", str(current_year)],
         env, f"{profile.id}: build_dataset")
    # CS fit before workouts: the rep-extraction layer (reps.py) needs the
    # CS timeline for its quality cutoff, and parse_workouts consumes its
    # output. The fit itself only needs races, so the order is safe.
    if profile.fit and fit:
        try:
            _run(["python", "src/models/bayes_cs_fit.py"], env,
                 f"{profile.id}: bayes_cs_fit")
        except subprocess.CalledProcessError:
            # A profile with very few races may not support the CS fit. Don't
            # abort the build — the CS-dependent plots just won't render (and
            # so their tabs auto-hide), leaving the daily-only tabs.
            print(f"[{profile.id}] WARNING: CS fit failed (too few races?) — "
                  f"CS-dependent tabs will be omitted")
    elif profile.fit:
        print(f"[{profile.id}] no --fit: reusing existing bayes_cs_* outputs")
    # Watch-derived rep extraction (workout_measured.csv). Watch profiles
    # have no hand log to reconcile against -> --watch-only. Needs the rich
    # details cache and a CS fit; skipped (with the tab unaffected) if absent.
    if (data_dir / "details").exists() and (data_dir / "bayes_cs_summary.csv").exists():
        _run(["python", "src/coros/reps.py", "--watch-only"],
             env, f"{profile.id}: reps")
    else:
        print(f"[{profile.id}] reps: no details cache or CS fit — skipped")
    _run(["python", "src/parsers/parse_workouts.py", "--continuous-fartlek-only"],
         env, f"{profile.id}: parse_workouts")
    # Per-day altitude + local-time envelopes (altitude_daily.csv /
    # time_daily.csv) for the Misc. Trends Altitude/Time panels, mined from the
    # rich detail cache (no network). Graceful no-op without rich details.
    if (data_dir / "details").exists():
        _run(["python", "-m", "src.coros.daily_envelopes"],
             env, f"{profile.id}: daily_envelopes")
    else:
        print(f"[{profile.id}] daily_envelopes: no details cache — skipped")


def _race_additions(cfg):
    """Build the additions DataFrame for a Coros profile.

    Two sources (a real second runner uses the sheet; the Max test profile
    reuses races.csv):
      - races_sheet: a single-tab Google Sheet already in additions schema,
        fetched via the Drive service account.
      - races_csv: an existing races.csv, projected to the additions schema.
    races_since (optional) filters to on/after that date. Empty if neither.
    """
    cols = ["date", "distance_m", "time_sec", "surface", "location", "event"]
    if cfg.get("races_sheet"):
        from src.parsers import drive_fetch
        svc = drive_fetch.get_drive_service()
        races = drive_fetch.fetch_sheet_as_df(svc, cfg["races_sheet"])
        races["date"] = pd.to_datetime(races["date"])
    elif cfg.get("races_csv"):
        races = pd.read_csv(REPO_ROOT / cfg["races_csv"], parse_dates=["date"])
    else:
        return pd.DataFrame(columns=cols)
    since = cfg.get("races_since")
    if since:
        races = races[races["date"] >= pd.Timestamp(since)]
    out = races[cols].copy()
    out["date"] = out["date"].dt.date.astype(str)
    return out


def build_plots(profile, env, *, strict):
    cmd = ["./scripts/run_plots.sh", "--no-shell"]
    if not strict:
        cmd.append("--keep-going")
    _run(cmd, env, f"{profile.id}: plots")


def build_shell(profile, switcher_profiles):
    profile_list = [(p.id, p.label, p.url_base) for p in switcher_profiles]
    wrote = write_index(profile.output_dir, include_admin=profile.admin,
                        profiles=profile_list, current_id=profile.id)
    print(f"[{profile.id}] shell {'written' if wrote else 'unchanged'}: "
          f"{profile.output_dir / 'index.html'}")


def stage(profile):
    dest = SITE_DIST / profile.site_subdir if profile.site_subdir else SITE_DIST
    dest.mkdir(parents=True, exist_ok=True)
    src = profile.output_dir
    if not src.exists():
        print(f"[{profile.id}] nothing to stage (no {src})")
        return
    for item in src.iterdir():
        if item.name == "debug":
            continue
        target = dest / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)
    print(f"[{profile.id}] staged -> {dest}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", help="comma-separated profile ids to build")
    p.add_argument("--sync-only", action="store_true",
                   help="for Coros profiles in --only, sync just the detail "
                        "cache + current_log and exit (no dataset/plots). Used "
                        "in CI to refresh Max's watch cache — which his Drive "
                        "build enriches from — without rendering the test "
                        "profile.")
    p.add_argument("--skip-data", action="store_true")
    p.add_argument("--rebuild-coros", action="store_true")
    p.add_argument("--fit", action="store_true",
                   help="run the Bayesian CS fit (Drive: pass --fit to the "
                        "pipeline; Coros: re-fit fit=True profiles). Default "
                        "off reuses cached bayes_cs_* outputs.")
    p.add_argument("--historical", action="store_true",
                   help="freeze historical data before the Drive build")
    p.add_argument("--no-stage", action="store_true")
    p.add_argument("--strict", action="store_true")
    args = p.parse_args(argv)

    if args.only:
        profiles = [get_profile(pid.strip()) for pid in args.only.split(",")]
    else:
        profiles = PROFILES

    if args.sync_only:
        for profile in profiles:
            if profile.source != "coros":
                print(f"[{profile.id}] --sync-only: not a Coros profile — skipped")
                continue
            print(f"\n{'#' * 64}\n# Sync watch cache: {profile.id} "
                  f"({profile.label})\n{'#' * 64}")
            try:
                sync_coros_cache(profile, rebuild=args.rebuild_coros)
            except ProfileSkip as e:
                print(f"[{profile.id}] SKIPPED: {e}")
        return

    # The switcher lists every profile that can currently be built, regardless
    # of which subset --only is rebuilding, so cross-profile navigation always
    # points at something that exists.
    switcher = [p for p in PROFILES if _is_available(p)]

    for profile in profiles:
        print(f"\n{'#' * 64}\n# Profile: {profile.id} ({profile.label}) "
              f"[{profile.source}]\n{'#' * 64}")
        env = _env_for(profile)
        try:
            if not args.skip_data:
                if profile.source == "drive":
                    build_drive_data(profile, env, fit=args.fit,
                                     historical=args.historical)
                elif profile.source == "coros":
                    build_coros_data(profile, env, rebuild=args.rebuild_coros,
                                     fit=args.fit)
                else:
                    raise SystemExit(f"unknown source: {profile.source!r}")
        except ProfileSkip as e:
            print(f"[{profile.id}] SKIPPED: {e}")
            continue
        build_plots(profile, env, strict=args.strict)
        build_shell(profile, switcher)
        if not args.no_stage:
            stage(profile)

    print("\nAll profiles built.")


if __name__ == "__main__":
    main()
