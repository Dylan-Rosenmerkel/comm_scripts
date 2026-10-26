"""Shared by the injection tests (wide_test.py, tracks_test.py, mhc_tracks.py): where an injected object may be placed,
and the number of electrons of an injected muon."""
import os
import numpy as np
from scipy import ndimage

# default since 2026-10-25: the whole footprint of an injected object must avoid real events (before: only its
# centre was checked, so a 15-35 px track could cross a real event and every method 'failed' on it).
# INJECT_FOOTPRINT=0 reproduces the earlier placement (and the results saved before that date).
FOOTPRINT_CHECK = os.environ.get("INJECT_FOOTPRINT", "1") != "0"


def footprint_busy(busy, iy, ix, grow=2):
    """True if the object's pixels (iy, ix), grown by `grow` px, touch the busy mask"""
    y0, y1 = max(iy.min() - grow, 0), min(iy.max() + grow + 1, busy.shape[0])
    x0, x1 = max(ix.min() - grow, 0), min(ix.max() + grow + 1, busy.shape[1])
    fp = np.zeros((y1 - y0, x1 - x0), bool)
    fp[iy - y0, ix - x0] = True
    fp = ndimage.binary_dilation(fp, iterations=grow)
    return bool((fp & busy[y0:y1, x0:x1]).any())


# muons: MUON_PHYS=1 (since 2026-10-26) gives minimum-ionising muons; the default (0) keeps the earlier faint test muon
# (250 e- per projected pixel, 0.94 keV/px) so that saved results stay comparable
MUON_PHYS = os.environ.get("MUON_PHYS", "0") != "0"
PIX_UM, THICK_UM = 15.0, 675.0      # pixel pitch and fully depleted thickness [um]
DEDX_MP = 0.27                      # most probable energy loss of a minimum-ionising particle in silicon [keV/um]
XI_KEV = 12.0                       # Landau width xi for 675 um of Si (0.1535 MeV cm2/g * Z/A * rho * x) [keV]


def muon_electrons(rng, L, e_per_e):
    """electrons deposited by a muon with projected length L [px]: MUON_PHYS -> 3-D path sqrt((15 L)^2 + 675^2) um at the
    most probable 0.27 keV/um plus a Moyal (Landau-like) fluctuation of width xi (scaled with the path); else L * 250"""
    if not MUON_PHYS:
        return int(rng.poisson(L * 250))
    path = np.hypot(PIX_UM * L, THICK_UM)
    from scipy.stats import moyal
    E = DEDX_MP * path + XI_KEV * (path / THICK_UM) * float(moyal.rvs(random_state=rng))
    return int(rng.poisson(max(E, 0.0) / e_per_e))
