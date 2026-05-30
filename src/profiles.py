"""Profile registry for the multi-profile dashboard.

Each profile is an independent dataset + plot site, built into its own data/
and output/ directories and staged under its own path in site/dist. The tab
bar carries a profile dropdown that navigates between them.

Adding another runner is just another entry here: a Coros profile needs only
a label, a region, and a pair of env vars holding that account's credentials —
"swap = change credentials". Exactly one profile is the default; it builds to
the repo-root data/ + output/ (unchanged paths) and is served at the site
root, so the existing URLs and the admin page keep working.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.shared.paths import REPO_ROOT


@dataclass(frozen=True)
class Profile:
    id: str
    label: str
    source: str                 # "drive" | "coros"
    default: bool = False
    admin: bool = False          # include the (gated) admin tab
    fit: bool = False            # run the bayes CS fit during the data build
    prod: bool = True            # build/show in production (RP_PRODUCTION set)
    coros: dict = field(default_factory=dict)   # region, start_day, env var names

    # ---- per-profile directories ----
    @property
    def data_dir(self):
        return (REPO_ROOT / "data" if self.default
                else REPO_ROOT / "data" / "profiles" / self.id)

    @property
    def output_dir(self):
        return (REPO_ROOT / "output" if self.default
                else REPO_ROOT / "output" / "profiles" / self.id)

    @property
    def site_subdir(self):
        """Path under site/dist/ for this profile ('' = site root).

        Non-default profiles live under ``profiles/<id>`` so the served URL
        layout matches the on-disk output/ layout (output/profiles/<id>/) —
        the dev server (rooted at output/) and the prod site (site/dist/) then
        resolve the same ``url_base`` with no special routing.
        """
        return "" if self.default else f"profiles/{self.id}"

    @property
    def url_base(self):
        """URL path the profile's shell is served at."""
        return "/" if self.default else f"/profiles/{self.id}/"


PROFILES = [
    Profile(
        id="max",
        label="Max",
        source="drive",
        default=True,
        admin=True,
    ),
    Profile(
        id="coros",
        label="Max (Coros)",
        source="coros",
        fit=True,                            # run the bayes CS fit (needs races)
        prod=False,                          # dev/test only — hidden in production
        coros={
            "region": "us",
            "start_day": "20201201",        # bound the first backfill
            "email_env": "COROS_EMAIL",
            "password_env": "COROS_PASSWORD",
            # Races for this profile reuse Max's race history (the watch feed
            # itself carries no race flags) as additions, filtered to the
            # Coros era so they align with the daily data and feed the CS fit.
            "races_csv": "data/races.csv",
            "races_since": "2020-12-01",
        },
    ),
    # The real second runner. Races come from a single-tab Google "additions"
    # sheet (shared with the Drive service account); daily data will come from
    # this runner's Coros account once COROS2_EMAIL/COROS2_PASSWORD are set —
    # until then the build skips this profile cleanly. Rename label/id freely.
    Profile(
        id="maddy",
        label="Maddy",
        source="coros",
        fit=True,
        coros={
            "region": "us",
            "email_env": "COROS2_EMAIL",
            "password_env": "COROS2_PASSWORD",
            "races_sheet": "1aIOyV0klbvUPZgU8Mb0dhyV1mKnjVNpyu1mhr462l5s",
        },
    ),
]


def get_profile(profile_id: str) -> Profile:
    for p in PROFILES:
        if p.id == profile_id:
            return p
    raise KeyError(f"unknown profile: {profile_id!r}")


def default_profile() -> Profile:
    for p in PROFILES:
        if p.default:
            return p
    return PROFILES[0]
