#!/usr/bin/env python3
"""Invariant guard for the location colour palette.

The palette's whole value is that a place's colour never moves — not
between the Locations and World Map tabs, not between profiles, and not
when a new place earns a colour. Those properties are easy to break with
an innocent-looking edit (a changed candidate grid, a new anchor, a tweak
to the qualification rule), and the breakage is invisible until someone
notices a colour shifted. This asserts them.

Run directly:  python scripts/check_location_palette.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.plotting import location_palette as lp   # noqa: E402
from src.profiles import PROFILES                 # noqa: E402

# The approved palette measures 5.157 dE; this floor sits just under it.
# Raising the bar is fine; dropping below it means an edit degraded
# separation and needs a second look before it ships.
APPROVED_MIN_SEPARATION = 5.15
APPROVED_ANCHOR_BASES = {
    'WA': '#8617be', 'TN': '#17be33', 'CO': '#be6b17', 'IL': '#17a2be',
}

_SNIPPET = (
    'import sys; sys.path.insert(0, %r)\n'
    'from src.plotting.location_palette import palette\n'
    'import json; print(json.dumps(palette(), sort_keys=True))\n' % str(REPO)
)


def _palette_in_subprocess(env=None):
    out = subprocess.run([sys.executable, '-c', _SNIPPET],
                         capture_output=True, text=True, env=env, cwd=REPO)
    if out.returncode != 0:
        raise SystemExit(f'palette failed to build:\n{out.stderr}')
    return json.loads(out.stdout)


def main():
    failures = []
    pal = lp.palette()

    # 1. Anchors are frozen. These four are the only hand-picked colours.
    for group, expected in APPROVED_ANCHOR_BASES.items():
        got = pal.get(group, [None])[0]
        if got != expected:
            failures.append(f'anchor {group}: expected {expected}, got {got}')

    # 2. Determinism. str hashing is salted per process, so anything that
    #    leaks hash ordering into the assignment shows up here and nowhere
    #    else — this is the check that would have caught the China bug.
    runs = [_palette_in_subprocess() for _ in range(3)]
    if any(r != runs[0] for r in runs[1:]):
        failures.append('palette differs between processes')
    if runs[0] != pal:
        failures.append('palette differs between subprocess and in-process')

    # 3. Cross-profile: the assignment must not depend on which profile is
    #    being built, so rerun it under each profile's RP_DATA_DIR.
    import os
    for profile in PROFILES:
        env = dict(os.environ, RP_DATA_DIR=str(profile.data_dir),
                   RP_OUTPUT_DIR=str(profile.output_dir))
        if _palette_in_subprocess(env) != pal:
            failures.append(f'palette changes under profile {profile.id!r}')

    # 4. Separation has not silently degraded.
    sep = lp.min_separation()
    if sep < APPROVED_MIN_SEPARATION:
        failures.append(f'min separation {sep:.2f} below approved '
                        f'{APPROVED_MIN_SEPARATION:.2f}')

    # 5. Every group any production profile colours has a track.
    for profile in PROFILES:
        if not profile.prod:
            continue
        daily = profile.data_dir / 'daily.csv'
        if not daily.exists():
            continue
        for group in lp._qualification_dates(daily).index:
            if group not in pal:
                failures.append(f'{profile.id}: qualified group {group!r} '
                                f'has no palette entry')

    print(f'groups: {len(pal)}   min separation: {sep:.2f} dE   '
          f'shades reserved each: {lp.RESERVED_SHADES}')
    if failures:
        print('\nFAILED:')
        for f in failures:
            print(f'  - {f}')
        return 1
    print('all palette invariants hold')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
