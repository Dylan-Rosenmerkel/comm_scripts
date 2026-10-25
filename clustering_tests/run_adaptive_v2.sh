#!/bin/bash
# fixed adaptive (per-pixel dark charge in the occupancy): Fe-55 validation + wide test
cd "$(dirname "$0")"
P=/opt/homebrew/Caskroom/miniforge/base/envs/damic/bin/python
METHODS=MHC,adaptive VALIDATE_OUT=results_adaptive_v2 NPROC=2 $P adaptive_validate.py fe55
WIDE_OUT=results_wide_adaptive_v2 NPROC=3 $P wide_adaptive.py
