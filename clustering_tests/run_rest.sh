#!/bin/bash
# remaining validations: fixed adaptive (tracks, lowE, crowded) and adaptive+GMX (crowded)
cd "$(dirname "$0")"
while kill -0 90042 2>/dev/null; do sleep 30; done
P=/opt/homebrew/Caskroom/miniforge/base/envs/damic/bin/python
METHODS=adaptive VALIDATE_OUT=results_adaptive_v2 NPROC=4 $P adaptive_validate.py tracks
METHODS=adaptive VALIDATE_OUT=results_adaptive_v2 NPROC=4 $P adaptive_validate.py lowE
METHODS=adaptive VALIDATE_OUT=results_adaptive_v2 NPROC=4 $P adaptive_validate.py crowded
METHODS=adaptive+GMX VALIDATE_OUT=results_adaptive_gmx NPROC=4 $P adaptive_validate.py crowded
