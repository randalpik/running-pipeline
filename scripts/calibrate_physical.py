#!/usr/bin/env python3
"""Write the cross-profile physical-beta ratios artifact.

The pooled recovery+long fit (recovery_model.physical_route_betas) prices
footing and altitude in s/mi for the profile it ran on. Those constants don't
transfer between runners — a 15 s/mi trail penalty is a different fractional
effort at a different speed — so the cross-profile currency is the RATIO:
beta divided by the fitting corpus's mean pace.

This script derives the ratios from the ACTIVE profile's live fit and writes
them to data/physical_beta_ratios.csv at the repo root (a shared artifact,
like dem_cache.json — profiles read it via recovery_model.shared_beta_ratios
when their own corpus has no off-road terrain labels to fit from). It runs on
the max/drive pipeline (scripts/run_pipeline.sh) right after calibrate_climb;
it refuses to export when the fit produced no genuine footing beta, so a thin
or label-less corpus can never overwrite the artifact with borrowed or zero
values.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.shared.recovery_model import (BETA_RATIOS_PATH,  # noqa: E402
                                       physical_beta_ratios)


def main():
    ratios = physical_beta_ratios()
    if ratios is None:
        print('[calibrate_physical] no genuine fitted footing beta on this '
              'profile — artifact left untouched')
        return
    import os
    row = {'profile': os.environ.get('RP_PROFILE', 'max'), **ratios}
    BETA_RATIOS_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(BETA_RATIOS_PATH, index=False)
    print(f"[calibrate_physical] wrote {BETA_RATIOS_PATH}: "
          f"footing_frac={ratios['footing_frac']:.5f} "
          f"alt_frac_per_kft={ratios['alt_frac_per_kft']:.5f} "
          f"(ref pace {ratios['ref_pace_s_per_mi']:.0f} s/mi)")


if __name__ == '__main__':
    main()
