"""Load a gitignored ``.env`` file at the repo root into the environment.

Local-dev convenience for secrets that live in GHA repo secrets in CI
(Coros credentials, etc.). Real environment variables always win — the
file only fills in what's unset — so CI and one-off ``VAR=... python``
invocations behave identically with or without a ``.env`` present.

Format: one ``KEY=VALUE`` per line; blank lines and ``#`` comments are
ignored; optional single/double quotes around VALUE are stripped. No
interpolation, no ``export`` keyword.
"""
from __future__ import annotations

import os

from src.shared.paths import REPO_ROOT


def load_env_file(path=None):
    """Set unset environment variables from ``.env``. Returns the names set."""
    path = path or REPO_ROOT / ".env"
    loaded = []
    if not path.exists():
        return loaded
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value
            loaded.append(key)
    return loaded
