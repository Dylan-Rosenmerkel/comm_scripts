#!/usr/bin/env python3 

import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Mapping, Any
import json
import pandas as pd
from scipy.interpolate import interp1d
from pysimdamicm import __path__ as __module_path__

# ---------------------------------------------------------------- METADATA
@dataclass
class MetaData:
    tag     : str
    run     : int
    module  : int
    ccd     : int
    image   : int
    file    : str
    rows    : int
    cols    : int
    skips   : int
    npbin   : int
    nsbin   : int
    rolevel : int
    start   : Optional[datetime] = None
    end     : Optional[datetime] = None
    texp    : Optional[float] = None
    tread   : Optional[float] = None
    tzero   : datetime = datetime(2025,1,1)
    
    # Derived image observables
    gain    : Optional[float] = None
    noise   : Optional[float] = None
    dc      : Optional[float] = None

    # Quality flags: 1=good, 2=warning, 3=bad
    gain_quality: Optional[int] = None
    noise_quality: Optional[int] = None
    dc_quality: Optional[int] = None
    image_quality: Optional[int] = None

    def __str__(self):
        title = "\n================== MetaData =================="
        items = []
        for k, v in self.__dict__.items():
            items.append(f"  {k:<8}: {v}")
        items.append("\n")
        return "\n".join([title] + items)

    __repr__ = __str__
    
    def set_detector_observables(self,gain: float, noise: float, dc: float) -> None:
        """
        Store the derived image observables used for DQM and quality evaluation.
        """
        if gain <= 0:
            raise ValueError("gain must be positive")
        if noise < 0:
            raise ValueError("noise must be non-negative")
        if dc < 0:
            raise ValueError("dc must be non-negative")

        self.gain  = gain
        self.noise = noise
        self.dc    = dc
        
    @staticmethod
    def evaluate_observable_quality(observed: float, expected: float, 
            allowed_sigma: float, relative_dispersion: float, me: str = "observable",
            warning_nsigma: float = 3.0, 
            bad_nsigma: float = 5.0) -> int:
        """
        Evaluate the quality flag of a single observable.

        Returns:
            0 -> good
            1 -> warning
            2 -> bad
        """
        if allowed_sigma <= 0:
            raise ValueError("allowed_sigma must be positive")
        if bad_nsigma < warning_nsigma:
            raise ValueError("bad_nsigma must be greater than or equal to warning_nsigma")

        zscore = abs(observed - expected) / allowed_sigma

        sigma_im = np.abs(expected) * relative_dispersion
        var = allowed_sigma**2 + sigma_im**2
        zscore = abs(observed - expected) / np.sqrt(var)
        if me == "gain":
            if np.abs(zscore) <= warning_nsigma:
                return 0
            if np.abs(zscore) <= bad_nsigma:
                return 1
            return 2
        elif me == "noise":
            if np.abs(zscore) <= warning_nsigma:
                return 0
            if np.abs(zscore) <= bad_nsigma:
                return 1
            return 2
        elif me == "dc":
            if np.abs(zscore) <= warning_nsigma:
                return 0
            if np.abs(zscore) <= bad_nsigma:
                return 1
            return 2

    def evaluate_image_quality_from_config(self, quality_config: Mapping[str, Any], npix: int,
        warning_nsigma: float = 3.0,  bad_nsigma: float = 5.0) -> int:
        """
        Evaluate the image quality flag using reference values and tolerances
        loaded from a configuration dictionary, typically parsed from JSON.

        The configuration is expected to contain one entry per (module, ccd),
        using keys of the form 'module:ccd'.
        """

        if self.gain is None or self.noise is None or self.dc is None:
            raise ValueError("gain, noise, and dc must be set before evaluating image quality")

        key = f"{self.module}:{self.ccd}"
        if key not in quality_config:
            raise KeyError(f"Missing quality configuration for module={self.module}, ccd={self.ccd}")

        cfg = quality_config[key]

        try:
            gain_expected = float(cfg["gain"]["expected"])
            gain_sigma = float(cfg["gain"]["sigma"])
            noise_expected = float(cfg["noise"]["expected"])
            noise_sigma = float(cfg["noise"]["sigma"])
            dc_expected = float(cfg["dc"]["expected"])
            dc_sigma = float(cfg["dc"]["sigma"])
        
        except KeyError as exc:
            raise KeyError(
                f"Incomplete quality configuration for key '{key}': missing {exc}"
            ) from exc
        
        N1_exp = dc_expected * npix*np.exp(-dc_expected)
        relative_dispersions = {}
        Adata = pd.read_csv(f'{__module_path__[0]}/data/Astd_noise_scan.csv')
        true_noise = Adata['noise_true'].values
        A_N1 = Adata['A_std_N1'].values
        parameter = Adata['parameter'].values
        keys = ['dc','noise','gain']
        for param in keys:
            interp = interp1d(true_noise[parameter == param], A_N1[parameter == param], kind='linear', fill_value='extrapolate')
            relative_dispersions[param] = interp(noise_expected)/np.sqrt(N1_exp)

        self.gain_quality = self.evaluate_observable_quality(
            observed=self.gain,
            expected=gain_expected,
            allowed_sigma=gain_sigma,
            relative_dispersion=relative_dispersions['gain'],
            me="gain",
            warning_nsigma=warning_nsigma,
            bad_nsigma=bad_nsigma,
        )

        self.noise_quality = self.evaluate_observable_quality(
            observed=self.noise,
            expected=noise_expected,
            allowed_sigma=noise_sigma,
            relative_dispersion=relative_dispersions['noise'],
            me="noise",
            warning_nsigma=warning_nsigma,
            bad_nsigma=bad_nsigma,
        )

        self.dc_quality = self.evaluate_observable_quality(
            observed=self.dc,
            expected=dc_expected,
            allowed_sigma=dc_sigma,
            relative_dispersion=relative_dispersions['dc'],
            me="dc",
            warning_nsigma=warning_nsigma,
            bad_nsigma=bad_nsigma,
        )
        
        self.image_quality = max(self.gain_quality, self.noise_quality, self.dc_quality)
        return self.image_quality

# ---------------------------------------------------------------- MONITOR ELEMENTS
@dataclass
class ME:
    name    : str
    kind    : str
    title   : str
    info    : str
    xlabel  : str
    ylabel  : str
    level   : str = "L0"
    counter : int = 1
    aggregation : str = "MEAN"
    x       : Optional[list] = None
    xerror  : Optional[list] = None
    y       : Optional[list] = None
    yerror  : Optional[list] = None
    yfit    : Optional[list] = None

    def pretty(self) -> str:
        lines = [self.name]
        for k, v in self.__dict__.items():
            lines.append(f"  {k:<8}: {v}")
        lines.append("\n")
        return "\n".join(lines)

    __str__  = pretty
    __repr__ = pretty

# ------------------------------------------------------------------------------------------- ME FACTORY
# ---------------------------------------------------------------- TEMPORAL
@dataclass
class MERowPedestalMean(ME):
    """
    Dark current distribution per pixel or per sensor (choose your binning).
    """
    def __init__(self):
        super().__init__(
            name="MERowPedestalMean",
            kind="TEMPORAL",
            title="Mean of the Per-Row Pedestals",
            info="Mean value of the row-by-row pedestal estimates for a given image, where each row pedestal is obtained from a Gaussian fit to the zero-electron peak of the row pixel-charge distribution. This monitor is useful to track baseline drifts with time.",
            xlabel="Time [days]",
            ylabel="Mean Per-row Pedestal [ADU]",
        )

@dataclass
class MEDarkCurrent(ME):
    """
    Dark current distribution per pixel or per sensor (choose your binning).
    """
    def __init__(self):
        super().__init__(
            name="MEDarkCurrent",
            kind="TEMPORAL",
            title="Poisson Dark Current",
            info="Dark Current estimated from the Poisson-Gaussian fit to the masked pixel-charge distribution. It corresponds to the mean thermally generated charge rate per pixel, after excluding clustered pixels (but not hot regions), and is used to monitor the stability of the sensor leakage current with time.",
            xlabel="Time [days]",
            ylabel="Dark Current Rate [e-/pixel/image]",
        )

@dataclass
class MEGain(ME):
    """
    Gain distribution (ADU → e⁻).
    """
    def __init__(self):
        super().__init__(
            name="MEGain",
            kind="TEMPORAL",
            title="Single-Electron Gain",
            info="Effective gain obtained from the spacing between adjacent electron peaks in the Poisson-Gaussian fit to the pixel-charge distribution. It defines the conversion between the digitized signal and collected charge, and is used to monitor the stability of the skipper readout chain.",
            xlabel="Time [days]",
            ylabel="Gain [ADU/e-]",
        )

@dataclass
class MEReadoutNoise(ME):
    """
    Single-Electron Response (SER) or Resolution
    """
    def __init__(self):
        super().__init__(
            name="MEReadoutNoise",
            kind="TEMPORAL",
            title="Readout Noise",
            info="Readout noise extracted from the Gaussian width of the electron-peak response in the Poisson-Gaussian fit to the pixel-charge distribution. It quantififies the charge resolution achieved by the skipper readout and is a key monitor of low-noise performance.",
            xlabel="Time [days]",
            ylabel="Readout Noise [e- RMS]",
        )

@dataclass
class MEZeroPeakCenter(ME):
    """
    Zero-electron peak center in the noise/charge distribution.
    """
    def __init__(self):
        super().__init__(
            name="MEZeroPeakCenter",
            kind="TEMPORAL",
            title="Residual Pedestal",
            info="Residual pedestal level inferred fro mthe fitted position of the zero-electron peak in the pixel-charge distribution. Deviations from zero indicate imperfect pedestal subtraction or slow drifts in the electronic baseline.",
            xlabel="Time [days]",
            ylabel="Residual Pedestal [e-]",
        )

@dataclass
class MECTI(ME):
    """
    Charge Transfer Inefficiency (CTI) vs. transfer index (row/column).
    Typical usage: x = transfer index (e.g., row #), y = CTI value per transfer.
    """
    def __init__(self):
        super().__init__(
            name="MECTI",
            kind="TEMPORAL",
            title="Charge Transfer Inefficiency",
            info="CTI estimated from each individual image using the overscan-based definition: (S_OS - S_baseline)/(S_active - S_baseline). It monitors charge deferred into the first overscan pixel due to impoerfect charge transfer and is sensitive to trapping and transfer losses during readout.",
            xlabel="Time [days]",
            ylabel="CTI [dimensionless]",
        )

# ---------------------------------------------------------------- STATIC
@dataclass
class MEPCD(ME):
    """
    Pixel Charge Distribution (PCD).
    Typical usage: x = pixel charge [e⁻], y = counts; yfit can be a model including 0e⁻ and 1e⁻ components.
    """
    def __init__(self):
        super().__init__(
            name="MEPCD",
            kind="STATIC",
            title="Pixel Charge Distribution",
            info="Distribution of pixel charge values. It is used to monitor low multiplicity peaks, and the overall response of the sipper readout, including pedestal, gain, noise, and dark current.",
            xlabel="Pixel Charge [e-]",
            ylabel="Pixels / bin",
            aggregation="SUM"
        )

@dataclass
class MESpectrum(ME):
    """
    Energy spectrum of events.
    Typical usage: x = energy bin centers, y = counts per bin, yfit = model spectrum.
    """
    def __init__(self):
        super().__init__(
            name="MESpectrum",
            kind="STATIC",
            title="Energy Spectrum",
            info="Energy spectrum of reconstructed clusters. It is used to monitor the distribution of reconstructed events energies for background purposes.",
            xlabel="Cluster Energy [keV]",
            ylabel="Counts / bin",
            aggregation="SUM"
        )

@dataclass
class MEPerRowPedestal(ME):
    """
    """
    def __init__(self):
        super().__init__(
            name="MEPerRowPedestal",
            kind="STATIC",
            title="Per-Row Pedestals",
            info="Row-by-row pedestal values obtained from the fitted position of the zero-electron peak in each row pixel-charge distribution. This monitor is used to identigy row-dependent baseline structure Per-row offsets values derived by Gaussian fitting (mu) of the zero-electron (first) peak in each row's pixel-charge distribution.",
            xlabel="Row Number",
            ylabel="Per-row Pedestal [ADU]",
        )

@dataclass
class MEPerRowResolution(ME):
    """
    """
    def __init__(self):
        super().__init__(
            name="MEPerRowResolution",
            kind="STATIC",
            title="Per-Row Readout Noise",
            info="Row-by-row readout noise estimated as the starndard deviation of the pixel-charge distribution in each row. This monitor is useful to identify row-dependent variations in readout performance, exces noise, or residual baseline structure.",
            xlabel="Row Number",
            ylabel="Readout Noise [e- RMS]",
        )

@dataclass
class MEColumnOccupancy(ME):
    """
    Distribution of the number of clusters per column.
    Typical usage: x = multiplicity (clusters/column), y = number of columns with that multiplicity.
    """
    def __init__(self):
        super().__init__(
            name="MEColumnOccupancy",
            kind="STATIC",
            title="Pixel Occupancy Above Threshold",
            info="Number of pixels above a fixed charge threshold in each column. For a spatially uniform dark current, this observable is expected to be approximatelly flat across columns; deviations can reveal column-dependent inhomogeneities, hot columns, or defect-related excess charge.",
            xlabel="Column Number",
            ylabel="Pixels Above Zero",
            aggregation="SUM"
        )


# ---------------------------------------------------------------- IMAGE
@dataclass
class MEMeanImage(ME):
    """
    Pixel Charge Distribution (PCD).
    Typical usage: x = pixel charge [e⁻], y = counts; yfit can be a model including 0e⁻ and 1e⁻ components.
    """
    def __init__(self):
        super().__init__(
            name="MEMeanImage",
            kind="IMAGE",
            title="Mean Image",
            info="Two-dimensional mean image obtained by averaging the pixel charge over all images for a given modula and CCD. It is used to reveal stable spatial structures such as hot pixels, hot columns, residual pedestals patterns, or other persistent sensor and readout features.",
            xlabel="Column Number",
            ylabel="Row Number",
        )


# ---------------------------------------------------------------- MECollection
@dataclass
class MECollection:
    metadata: MetaData
    me_list: List[ME] = field(default_factory=list)

    def save(self, path):
        import pickle
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)


