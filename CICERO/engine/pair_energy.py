"""Mean energy per electron-hole pair in silicon, epsilon(T), used to convert electrons to energy.

The pair creation energy grows as silicon is cooled (the band gap widens). comm_scripts uses

    epsilon(T) = 3.752 eV x (1 - 1.31e-4 / K x (T - 123 K))

  3.752 +- 0.002 eV at 5.9 keV and 123 K: skipper-CCD measurement, Rodrigues et al., NIM A 1010 (2021) 165511
  -0.0131 +- 0.0004 % / K: Lowe & Sareen, NIM A 576 (2007) 367 (5.9 keV X-rays, 80-270 K)

The CCDs are assumed to run at 130 K (epsilon = 3.7486 eV); another temperature is given with the environment
variable COMM_TEMPERATURE_K (run_WADERs --temperature sets it) or the `T` argument. panaSKImg (pysimdamicm) converts
with a fixed 3.74 eV (PANASKIMG_EV): its own cluster energies stay on that scale, and code that reads them back to
electrons must use PANASKIMG_EV, not pair_energy_eV().
"""
import os

EPS_REF_EV = 3.752          # eV per pair at T_REF_K (5.9 keV)
T_REF_K = 123.0
SLOPE_PER_K = -1.31e-4      # relative change per kelvin
T_DEFAULT_K = 130.0         # assumed operating temperature
PANASKIMG_EV = 3.74         # what panaSKImg / pysimdamicm uses (units.e2eV)


def temperature_K():
    """Operating temperature [K]: $COMM_TEMPERATURE_K if set, else T_DEFAULT_K (130 K)."""
    v = os.environ.get("COMM_TEMPERATURE_K", "").strip()
    return float(v) if v else T_DEFAULT_K


def pair_energy_eV(T=None):
    """epsilon(T) [eV per electron]; T [K] defaults to temperature_K()."""
    T = temperature_K() if T is None else float(T)
    return EPS_REF_EV * (1.0 + SLOPE_PER_K * (T - T_REF_K))
