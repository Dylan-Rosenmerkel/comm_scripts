#!/bin/bash
# validation of GMX and adaptive+GMX (tracks, few-electron, Fe-55) -> results_adaptive_gmx/
cd "$(dirname "$0")"
P=/opt/homebrew/Caskroom/miniforge/base/envs/damic/bin/python
export METHODS=GMX,adaptive+GMX VALIDATE_OUT=results_adaptive_gmx NPROC=4
$P adaptive_validate.py tracks
$P adaptive_validate.py lowE
$P adaptive_validate.py fe55
