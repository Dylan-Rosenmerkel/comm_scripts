""":mod:`pysimdamicm.processes.gain_nonlinearity`
    Correction of the charge dependence of the gain, g(n) = dADU/dn (the electron "comb" spacing at n
    electrons), measured with ``gain_vs_charge`` (pysimdamicm/comm/gain_vs_charge.py, bin/gain_vs_charge).

    GainNonlinearityProcess runs after CalibrationProcess and before ClusterFinder. It reads the g(n) model of
    every extension from ``model_file`` (the *_gain_vs_charge.json written by ``gain_vs_charge measure``) and
    attaches it to the image. ClusterFinder then

      - (algorithm "waders") corrects the energies of the clusters it found exactly as ``gain_vs_charge apply``
        does for cluster ROOT files: every pixel's ADU (pixel energy / e2eV x the calibration gain) is converted
        to electrons with n = A^-1(ADU), A(n) = integral of g, and back to energy with epsilon(T);
      - (algorithm "agmx") converts the calibrated image to electrons with the same curve before clustering, as
        comm_scripts' run_WADERs --gain-correction --clusterer agmx.

    The ADU of a pixel is recovered with CalibrationProcess's own curve (calibration_adu): the constant gain, or
    the polynomial ADU = offset + g e + c2 e^2 + ... of a polynomial calibration (get_calibration_constant). With a
    polynomial calibration g(n) therefore REPLACES it for the cluster energies; it is never applied on top of it.

    The pixels and the threshold cut of the WADERs clusters are unchanged (the cut is on the image calibrated
    with the fitted gain), so the cluster content is identical to the uncorrected run; only pixels_E and Energy
    change, as with ``gain_vs_charge apply``.

.. moduleauthor:: comm_scripts (D. Rosenmerkel)
"""
import json
import os

import numpy as np

from pysimdamicm.processes.absp import SKImageProcess
from pysimdamicm.utils.units import Units
u = Units()


class GainNonlinearityProcess(SKImageProcess):
    """Attach the g(n) curve of each extension to the image, for ClusterFinder (see the module documentation).

    Parameters (JSON block "GainNonlinearityProcess")
    ----------
    model_file : str
        *_gain_vs_charge.json from ``gain_vs_charge measure`` (one model per extension, "EXTn").
    anchor : "comb" | "fit"
        comb (default): ADU -> electrons with the comb's g(n) including its low-charge gain;
        fit: keep the image's calibration gain at low charge and correct only the charge dependence.
    temperature : float or null
        CCD temperature [K] for the energy per electron epsilon(T) of the corrected energies
        (default: $COMM_TEMPERATURE_K, else 130 K -> 3.7486 eV).
    """
    __sequence_id__ = 26
    __name__ = 'GainNonlinearityProcess'

    def __init__(self):
        super().__init__()
        self._per_amp = False
        self.model_file = ""
        self.anchor = "comb"
        self.temperature = -1
        self.__units__.update({'model_file': 1, 'anchor': 1, 'temperature': 1})

    def execute_process(self, rawdata):
        from pysimdamicm.comm import gain_vs_charge as GVC
        from pysimdamicm.comm.pair_energy import pair_energy_eV
        print("Process <GainNonlinearityProcess> INFO.")
        if not self.model_file or not os.path.exists(self.model_file):
            raise IOError(f"GainNonlinearityProcess: model_file '{self.model_file}' not found "
                          f"(make it with: gain_vs_charge measure IMAGES -o DIR)")
        if self.anchor not in ("comb", "fit"):
            raise ValueError(f"GainNonlinearityProcess: anchor must be 'comb' or 'fit', not {self.anchor!r}")
        with open(self.model_file) as fh:
            models = json.load(fh)["extensions"]
        temperature = None if self.temperature is None or self.temperature < 0 else float(self.temperature)
        eps = pair_energy_eV(temperature)
        curves = {}
        for amp in rawdata.amplifier.keys():
            model = GVC.load_model(models, amp)
            gain = u._tohdr.get(f"GAIN_{amp}", (None,))[0]
            if model is None or gain is None:
                why = "no reliable g(n) curve in the model file" if model is None else "no calibration gain"
                print(f"    {amp}: not corrected ({why})")
                continue
            curves[amp] = dict(model=model, gain=float(gain), anchor=self.anchor, eps=eps, amp=amp)
            print(f"    {amp}: g(n) {GVC.model_form(model)} model, calibration gain {float(gain)} ADU/e-, "
                  f"anchor {self.anchor}, {eps:.4f} eV/e- for the corrected energies")
            if calibration_is_polynomial(amp):
                print(f"    {amp}: NOTE CalibrationProcess used a polynomial calibration; the cluster energies are "
                      f"converted back to ADU with that polynomial and then to electrons with g(n), i.e. g(n) "
                      f"replaces the polynomial for the energies (it is not applied twice)")
        rawdata.gain_curve = curves
        u._tohdr.update({"GNLFILE": (os.path.basename(self.model_file)[:60], "g(n) model (gain_vs_charge)")})


def calibration_adu(e, amp):
    """ADU above the zero-electron peak of pixels that CalibrationProcess converted to e (the inverse of its
    conversion). CalibrationProcess records its curve ADU = offset + g e + c2 e^2 + ... in u._tohdr (CAL<amp>k,
    k >= 1, offset CAL<amp>0, valid range CAL<amp>R0/R1 with linear continuation outside); a linear calibration
    only GAIN_<amp>. Returns g e + c2 e^2 + ... (the offset is the zero of the electron comb, as in g(n))."""
    from numpy.polynomial import polynomial as P
    e = np.asarray(e, float)
    hdr = u._tohdr
    coeffs = []
    k = 1
    while f"CAL{amp}{k}" in hdr:
        coeffs.append(float(hdr[f"CAL{amp}{k}"][0]))
        k += 1
    if len(coeffs) < 2:                                  # linear calibration: ADU = gain x e
        return e * float(hdr[f"GAIN_{amp}"][0])
    c = np.concatenate(([0.0], coeffs))
    out = P.polyval(e, c)
    if f"CAL{amp}R0" in hdr and f"CAL{amp}R1" in hdr:
        dc = P.polyder(c)
        for edge, side in ((float(hdr[f"CAL{amp}R0"][0]), e < float(hdr[f"CAL{amp}R0"][0])),
                           (float(hdr[f"CAL{amp}R1"][0]), e > float(hdr[f"CAL{amp}R1"][0]))):
            out[side] = P.polyval(edge, c) + (e[side] - edge) * P.polyval(edge, dc)
    return out


def calibration_is_polynomial(amp):
    """True if CalibrationProcess used a polynomial (more than one coefficient) for this amplifier."""
    return f"CAL{amp}2" in u._tohdr


def electrons(adu, curve):
    """ADU -> electrons with the g(n) curve attached by GainNonlinearityProcess (array in, array out)."""
    from pysimdamicm.comm import gain_vs_charge as GVC
    a = np.asarray(adu, float)
    return np.asarray(GVC.electrons_with_anchor(a.ravel(), curve["model"], curve["gain"], curve["anchor"]),
                      float).reshape(a.shape)


def correct_clusters(clusters, curve):
    """pixels_E [keV] and Energy of pysimdamicm Cluster objects corrected with g(n), as gain_vs_charge.apply:
    ADU = pixels_E / e2eV x gain, n = A^-1(ADU), pixels_E = n x epsilon(T). Returns the median ratio
    new / old Energy of the clusters above 1 keV (nan if none)."""
    ratios = []
    for cls in clusters:
        v = np.asarray(cls.pixels_E, float)                     # [keV]
        adu = calibration_adu(v * u.keV / u.e2eV, curve["amp"])  # back to ADU with CalibrationProcess's own curve
        new = electrons(adu, curve) * curve["eps"] * u.eV / u.keV
        old = float(cls.Energy)
        cls.pixels_E = new
        cls.Energy = float(new.sum())
        if old > 1.0:
            ratios.append(cls.Energy / old)
    return float(np.median(ratios)) if ratios else float("nan")
