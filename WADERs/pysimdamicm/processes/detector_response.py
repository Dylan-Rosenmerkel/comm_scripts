""":mod:`pysimdamicm.detector_response`

A package to handle different digitilize process involved in the response
of the DAMIC-M detector (process involved either in sensor or electronics
response).

Note
----
Geant4 system units as default, see :obj:`pysimdamicm.io.G4utils.Units`

.. moduleauthor:: Nuria Castello-Mor
"""
from abc import ABCMeta, abstractmethod

import numpy as np
import pandas as pd
from glob import glob
from astropy.io import fits
import warnings
#warnings.simplefilter('ignore', category=fits.verify.VerifyWarning)
import os
import random

from scipy.interpolate import RegularGridInterpolator, interp1d, PchipInterpolator

from pysimdamicm import __path__ as __module_path__
from pysimdamicm.utils.units import Units
u = Units()

##### module DEBUG VARIABLES
__verbose__ = False
__debugs__ = False

from time import time
import random

###############################################################################################
#####       ABSTRACT CLASS TO DEFINE THE DIFFERENT PROCESS
#####           PART OF THE DETECTOR RESPONSE
###############################################################################################
class DigitizeProcess(metaclass=ABCMeta):
    """Abstract Class for the digitilize process.

    Each process involved in the response of the DAMIC-M detector (based on CCDs) will
    be created from this abstract class.

    Example
    -------

    1. Diffusion process
        >>> import pysimdamicm as dam
        >>> diff = dam.detector_response.Diffusion()
        >>> diffusion.__name__
        'Diffusion'
        >>> diffusion.__sequence_id__
        10
        >>> diffusion.__units__
        {'z_offset': 0.001, 'B': 1000.0, 'A': 1e-06, 'alpha': 1.0, 'fanofactor': 1}
        >>> diff_params = {'A':159.0, 'B': 0.0001, 'fanofactor':0.10 }
        >>> diff.set_parameters(**diff_params)
        >>>

    2. DarkCurrent process
        >>> import pysimdamicm as dam
        >>> dc = dam.detector_response.DarkCurrent()
        >>> dc.__name__
        'DarkCurrent'
        >>> dc.__sequence_id__
        40
        >>> dc.__units__
        {'img_size': 1, 'mode': 1, 'darkcurrent': 1.1574074074074073e-05, 'exp_time': 86400.0,'min_enlarge': 1}
        >>> dc_params = {'exp_time':8/24., 'darkcurrent': 0.001 , 'mode':''}
        >>> dc.set_parameters(**dc_params)
        Running on mode: full-image
    >>> dc.dark_current_image
    array([[0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.],
           ...,
           [0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.]])
    >>> dc.dark_current_image.mean()
       0.0012602795833333822

    """
    __seed__ = 321
    __units__ = {}

    def __init__(self,seed=None):
        if seed is not None:
            DigitizeProcess.__seed__ = int(seed)

        ## Set the seed for the random generation process
        ## np.random.seed(DigitizeProcess.__seed)
        self.__verbose__ = False
        self.__display__ = False
        self.__DEBUG__   = False
        self.save_image  = False
        self.save_plots  = False

        self.__units__ = {
                "__display__":1,
                "__DEBUG__"  :1,
                "__verbose__":1,
                "save_image" :1,
                "save_plots" :1,
                }


    #def __str__(self):
    #    """String shown when you use the print() function
    #    """
    #    msg = '<Digitize Process: {}>\n'.format(self.__class__.__name__)
    #    # FIXME Problem with attr.find('_') != 0 --> RecursionError WTF??
    #    for attribute in filter(lambda attr: attr. find('_') == -1,dir(self)):
    #        try:
    #            msg += ' {}: {}\n'.format(attribute,getattr(self,attribute))
    #        except AttributeError:
    #            pass
    #    return msg[:-1]

    #def __repr__(self):
    #    """String shown when you push return over an instance of this class
    #    """
    #    return self.__str__()

    def set_parameters(self, **model_parameters):
        """Runtime method to set all atributes of the digitize process listed on the configuration
        json file.

        Those process's parameters not listed on the configuration json file will set to the default
        values.

        Parameters
        ----------
        **model_parameters : dict
            Keyword arguments for all attributes realted with the digitizer process (see examples).

        """
        for keyword,val in model_parameters.items():
            if keyword in ["units","unit"]:
                continue
            if hasattr(self,keyword):
                if type(val)==dict:
                    # some parameters are more complicates than single values, or list, and units can
                    # not be added XXX Case of the Signal Patter Recognition Process XXX
                    setattr(self,keyword,val)
                else:
                    setattr(self,keyword,val*self.__units__[keyword])
            else:
                raise(AttributeError('"{}" invalid attribute for class {}'.format(keyword,self.__class__.__name__)))

    @property
    def mode(self):
        """Property **mode** to define how the intrinsic detector should be simulated.
        It is mainly related to the size of the noise image, and if this image should be created
        for each pair of values (event,ccd) or not. There are two modes:

            **full-image**

                at the beginning of the process one image noise (DarkCurrent and/or ElectronicNoise) will be
                created and used for all the pairs (event,ccd). The shape of this image corresponds to the
                shape of the sensitive detector (defined at :obj:`Units.ccd_shape`)

            **cropped-image**

                the default mode. For each event an image will be re-created with a shape smaller than the
                sensitive detector. This mode is faster than the former, the filtering before ClusterFinder has less
                pixels to evaluate.
        """
        return self.__mode
    @mode.setter
    def mode(self,val):
        """Setter for mode property
        """
        self.recreate_image = False
        self.full_image_counter = 0

        if not self.__name__ in ['DarkCurrent', 'ElectronicNoise']:
            msm="The attribure 'mode' is only implemented for process related with the intrinsice detector noise"
            raise AttributeError(msm)

        if val == 'cropped-image':
            print('     Set mode to `cropped-image`.')
            self.execute_process = self.__execute_process_cropped_image__
        elif val == 'full-image':
            self.execute_process = self.__execute_process_full_image__
            print('     Set mode to `full-image`.')
        else:
            msm = "Mode not implemented. Possible modes are: \n"+\
                    "- full-image: process that will be executed once during reconstruction,"+\
                    " so all events will share its output (i.e. noise)\n"+\
                    "- cropped-image: process will be executed for each hit collection"
            raise NotImplementedError(msm)
        self.__mode = val

    @abstractmethod
    def execute_process(self):
        """Metaclass abstract method to be implemented for each specific
        process.
        """
        raise NotImplementedError()

    def __del__(self,hits=None):
        """
        """
        pass

###############################################################################################
#####       ELECTRON DIFFUSION MODEL
###############################################################################################
class Diffusion(DigitizeProcess):
    """Class to generate the creation of electron-hole pairs and its diffusion,
    for silicon detector.

    **Charge e-h pair Creation**

    A delta-function energy lost at :math:`p(x,y,z)`, within the silicon bulk of the CCD, will
    generate :math:`N_{Q}` e-h pairs, which is given by


    .. math::

       N_{Q} = integer \left[ \mathcal{G}(\mu,\sigma) \\right]


    where

    .. math::
       :nowrap:

       \\begin{equation}
         \mu = \delta E/ \epsilon_{m}
       \end{equation}


    .. math::
       :nowrap:

       \\begin{equation}
           \sigma = \sqrt{(F * \mu)}
       \end{equation}

    where :math:`F` is the fano factor and :math:`\epsilon_{m}` is the average
    energy loss required to produce a charge pair (e-h pair).

    **Charge diffusion**

    The transverse diffusion of delta-function charge deposition in
    the bulk of the CCD is described by a Gaussian with
    :math:`\sigma_{xy}=\sqrt{2 D_{p} t}` where :math:`D{p}` is the
    diffusion coefficient of the holes and :math:`t` is the time
    elapssed since creation. By using the linear relationship between drift velocity of charge
    carriers and electric field in the semiconductor, :math:`\sigma_{xy}`
    measured by the detector is
    just a function of the starting z-position in the bulk

    .. math::

       \sigma_{xy}^{2}(z,E=0)  = - A \ln{( 1-B(z- z_{offset}) )}

    Where :math:`A` and :math:`B` are coefficients based on the
    physical parameters of the silicon detector (permitivity of
    silicon, donor charge density, operating temperature, bias across
    the CCD, and the thickness of the active region).


    It is obvious to think that the transversal diffusion has an energy dependence: a higher
    energy deposition, stronger the created local electric field. The nal diusion model is therefore

    .. math::
       :nowrap:

       \\begin{equation}
         \sigma_{xy}(z,E) = \sigma_{xy}(z,E=0) \left( \\alpha_{0} + \\frac{\\alpha E}{ \sigma_{xy}(z=z_{offset}, E=0)} \\right)
       \end{equation}

    where :math:`\sigma_{xy}(z,E=0)` is shown in the previous equation.

    **See more details** at `DAMIC-100 notes <https://www.overleaf.com/3446968864nchrgrjbmccq>`_


    **List of Atributes:**

    Attributes
    ----------

        A
            amplitude of the :math:`\sigma_{xy}-z` relationship (depends on the silicon bulk parameters: silicon permeability, donor charge density, ...)

        B
            shape of the :math:`\sigma_{xy}-z` relationship

        fanofactor

            value related with the conversion of absorbed energy into signal quanta, related with the fluctuation of the signal for a given absorbed energy

        z_offset

            thickness of the active region (fully depleted region)

    only_pair_creation

        variable to activate/deactive diffusion. 0: diffusion on, 1: diffusion off

.. warning::

    * y-axis as the columns (horitzontal register),
    * x-axis as the rows (serial register),
    * z-axis as the direction of the electric field in the dopped region of the silicon detector.
    * Geant4 units: mm, eV, second (see :obj:`pysimdamicm.io.G4utils.Units`)
    * 3.77eV as the average energy lost required to produce a charge pair in the bulk of the CCD, which can be changed from :obj:`pysimdamicm.io.G4utils.Units.e2eV`

    """
    __sequence_id__ = 10
    __name__ = 'Diffusion'

    __E_in_eV__ = True

    def __init__(self):
        super().__init__()

        self.model = "gauss"
        self.only_pair_creation = 0


        #### Parameters of the model
        self.A = (float(216.2)*u.um*u.um)
        self.B = (float(0.000886)*1/u.um)
        self.z_offset = float(0.0)*u.um
        self.fanofactor = float(0.129)
        self.alpha = float(0.00597*1e-3)*u.pixel/u.eVee
        self.alpha0 = 1.0
        self.z_max = float(669.0)*u.um
        # parameters for PCC and charge-repulsion corrections
        self.apply_pcc = 0
        self.pcc_debug = 0
        self.repulsion = 0                      # 0 = off, 1 = on
        self.sigma_map_file = None              # path to sigma map file (.xlsx or .csv)
        self.sigma_interpolator = None          # store interpolation object
        self.sigma_map_loaded = False           # flag when map successfully loaded
        # parameters for nuclear-recoil quenching
        self.quenching = 0                      # 0 = off, 1 = on
        self.prod_cut_eV = 10.0                 # eV, production cut used to classify low-energy n/p as NR
        self.quenching_debug = 0                # 1 = print per-hit quenching diagnostics


        ###Pixel Charge Collection function
        thick= u.ccd_thickness/u.um   # CCD thickness from json file in microns
        z_um_data=np.array([thick+0.0,thick-0.5,thick-1.0,thick-1.5,thick-2.0,thick-2.5,
                            thick-3.0,thick-3.5,thick-4.0,thick-4.5,thick-5.0,thick-5.5,thick-6.0,
                            thick-6.5, thick-7.0, thick-7.5])
        # dead layer
        eps_vals = np.array([0.00781727, 0.027572, 0.0789126, 0.157685, 0.247863, 0.34214,
                             0.437597, 0.533334, 0.629258, 0.725383, 0.821692, 0.918136,
                             0.991682, 0.999804, 0.999997, 1.0 ])
        self.eps_func = interp1d(z_um_data, eps_vals, kind='linear', bounds_error=False, fill_value=(1.0, 0.0))
        ###PCC FUNCTION ENDS

        self.__units__={'A':u.um**2.0,'B':1/u.um,'z_offset':u.um,'fanofactor':1,'alpha':u.pixel/u.eVee,'model':1,
                        'z_max':u.um,'alpha0':1, 'only_pair_creation':1,'apply_pcc': 1, 'pcc_debug':1, 'repulsion': 1,
                        'sigma_map_file': 1, 'quenching': 1, 'prod_cut_eV': 1, 'quenching_debug': 1
                        }

        datafile = "{}/data/p100K.npz".format(__module_path__[0])
        ehdata = np.load(datafile)['data']
        self.ehenergies = ehdata[:,0].flatten()
        self.ehprobs = ehdata[:,1::]


    def load_sigma_map_lazy(self):
        """Load sigma map on first use (called inside execute_process)."""
        if getattr(self, 'repulsion', 0) != 1:
            return

        if self.sigma_map_loaded:
            return

        try:
            datafile_1 = self.sigma_map_file
            if not os.path.isabs(datafile_1):
                datafile_1 = os.path.join(__module_path__[0], 'data', datafile_1)

            if not os.path.exists(datafile_1):
                print(f"[WARNING] Sigma map file not found: {datafile_1}")
                self.repulsion = 0
                return

            if pd is None:
                raise RuntimeError("pandas required to read Excel/CSV sigma_map_file")

            print(f"[Diffusion] Lazy-loading sigma map from {datafile_1}")
            df = pd.read_excel(datafile_1, index_col=0)
            sigma_map = df.values.astype(float)
            energy_vals = df.index.to_numpy(dtype=float)
            depth_vals  = df.columns.to_numpy(dtype=float)

            self.sigma_interpolator = RegularGridInterpolator((energy_vals, depth_vals), sigma_map, bounds_error=False, fill_value=None)
            self.sigma_map_loaded = True

            print(f"[Diffusion] Loaded σ_map: shape={sigma_map.shape}, ")

        except Exception as e:
            print(f"[Diffusion]  Could not load sigma map: {e}")
            self.sigma_map_loaded = False
            self.repulsion = 0

    def _yield_model(self, E_eV):
        # ------------------------------------------------------------
        # Lindhard analytic model (for high E_nr)
        # ------------------------------------------------------------
        def _lindhard_yield_keV(E_keV, k=0.146, Z=14):
            E = np.asarray(E_keV, dtype=float)
            eps = 11.5 * E * (Z ** (-7 / 3))
            g = 3 * eps ** 0.15 + 0.7 * eps ** 0.6 + eps
            Y = (k * g) / (1 + k * g)
            return np.clip(Y, 0.0, 1.0)

        # ------------------------------------------------------------
        # Julian data (low energy)
        # ------------------------------------------------------------
        def _julian_yield_data():
            E_nr = np.array([0.26309589206684196, 0.263842387, 0.281081795, 0.296305363, 0.308713703, 0.321641665,
                             0.331206078, 0.349144408, 0.365903602, 0.377520970, 0.403289571,
                             0.425505878, 0.453028012, 0.490486034, 0.537127309, 0.577952603,
                             0.620061283, 0.681016763, 0.739248738, 0.820290870, 0.904368537,
                             1.019750083, 1.121456769, 1.250288171, 1.359850957, 1.511140681,
                             1.719092637, 1.955661397, 2.187435462, 2.437137512 ])
            E_e = np.array([0.0, 0.00264591, 0.003257533, 0.003948283, 0.004609659, 0.005608478,
                            0.006758963, 0.008003881, 0.009609197, 0.011574066, 0.014185168,
                            0.018201961, 0.022774882, 0.027242729, 0.033800141, 0.039856799,
                            0.048986441, 0.059157668, 0.071572365, 0.086255487, 0.100207339,
                            0.122212224, 0.141907774, 0.169819043, 0.195046149, 0.228951380,
                            0.268841298, 0.313355092, 0.369172124, 0.417528129 ])
            Y = E_e / E_nr
            return E_nr, Y

        # ------------------------------------------------------------
        # Continuation (mid–high energy)
        # ------------------------------------------------------------
        def _julian_continuation():
            E_nr = np.array([1.689362199, 2.279741960, 3.143986763, 4.289035859, 5.851115156,
                             7.982108263, 10.88921524, 15.00564742, 20.26536914, 27.41289925,
                             38.02032329, 51.45071611, 71.73027368, 91.85809041,
                             124.7679463, 162.0850415, 229.3218139, 300.6573847, 384.8076072,
                             497.0689288, 690.9873042, 942.6468841, 1294.3676, 1754.312561,
                             2366.362375, 3042.359973 ])

            Y = np.array([0.149453776, 0.164908033, 0.187554809, 0.206268134, 0.227996335,
                          0.248026409, 0.268391388, 0.293009196, 0.333633676, 0.358319140,
                          0.380443103, 0.415602556, 0.453049800, 0.487049575,
                          0.535393409, 0.574791550, 0.625640502, 0.658673971, 0.700786621,
                          0.741695410, 0.789117254, 0.824282933, 0.854276605, 0.891248281,
                          0.918043155, 0.920322982 ])
            return E_nr, Y

        # ------------------------------------------------------------
        # Build combined interpolator
        # ------------------------------------------------------------
        def _ensure_julian_interp():
            Ej, Yj = _julian_yield_data()
            Ec, Yc = _julian_continuation()
            mask = Ec > Ej[-1]
            # x = np.concatenate([Ej, Ec])
            # y = np.concatenate([Yj, Yc])
            x = np.concatenate([Ej, Ec[mask]])
            y = np.concatenate([Yj, Yc[mask]])
            interp = PchipInterpolator(x, y, extrapolate=False)
            return(interp,x,y)

        # ------------------------------------------------------------
        # Piecewise yield (Julian + continuation + Lindhard)
        # ------------------------------------------------------------
        def _yield_julian_piecewise(E_keV):
            #interp, x, y = _ensure_julian_interp()
            E = np.atleast_1d(E_keV).astype(float)
            Y = np.zeros_like(E)

            # WADERS region cuts
            JULIAN_MAX = 2.5      # keV
            SARKIS_MAX = 3042.0   # keV

            m_lo  = E < JULIAN_MAX
            m_mid = (E >= JULIAN_MAX) & (E < SARKIS_MAX)
            m_hi  = E >= SARKIS_MAX

            if np.any(m_lo):
                interp_J, x, y = _ensure_julian_interp()
                Y[m_lo] = interp_J(E[m_lo])     # your PCHIP Julian is fine

            # --- Sarkis (2.5–3042 keV) ---
            if np.any(m_mid):
                Ec, Yc = _julian_continuation()  # these are the Sarkis data points
                Y[m_mid] = np.interp(E[m_mid], Ec, Yc)   # MATCHES THEIR CODE EXACTLY

            if np.any(m_hi):
                Y[m_hi]  = _lindhard_yield_keV(E[m_hi])
            return Y if Y.size > 1 else float(Y)

        E_keV = np.asarray(E_eV, dtype=float) * 1e-3
        return _yield_julian_piecewise(E_keV)

    def _fano_factor_NR(self, n_bar):
        """
        SuperCDMS spline for nuclear Fano factor F_nr(n_bar),
        where n_bar is mean electrons. Clipped to [1, 20].
        - flat (clamped) below/above endpoints
        - safe against NaN/inf
        """
        n_bar = np.asarray(n_bar, dtype=float)
        n_bar = np.where(np.isfinite(n_bar), n_bar, 0.0)
        n_bar = np.maximum(n_bar, 0.0)
        # Data digitized from Fig. 5.47 (mean e- vs Fano)
        F_data = np.array([0.791871095, 1.205274901, 1.825605643, 2.341040948, 2.99389592, 3.422788444,
                           3.871395291, 4.372565278, 4.896991926, 5.390427447, 5.701593709, 6.122261894,
                           6.570333239, 6.972367771, 7.314686045, 7.490754229, 7.764699443, 8.093277907,
                           8.368853063, 8.557417471, 8.781267112, 8.974027113, 8.982331977, 9.116444239,
                           9.334347608, 9.445640678, 9.442234104, 9.413134811, 9.413954013, 9.374350647,
                           9.326344022, 9.310139383, 9.257303299, 9.175780232, 8.96769315, 8.715227368,
                           8.505482172, 8.441915551 ])
        n_e_data = np.array([4.094734282, 9.787541693, 15.19146262, 21.42690986, 27.70819494, 34.29950304,
                             41.3439746, 47.08825806, 54.29912135, 60.85802039, 67.75580559, 74.14568978,
                             80.78678235, 87.42941153, 94.0740339, 100.7242053, 107.3711099, 114.0161909,
                             120.663041, 127.3127953, 133.9613719, 140.6109862, 147.2667573, 153.9183291,
                             160.5671042, 167.2194377, 173.8755996, 180.5326191, 187.18864, 193.8460101,
                             200.5036606, 207.1602497, 213.8180615, 220.4768308, 227.1398245, 233.8042995,
                             240.4673485, 244.4025897 ])

        spline = PchipInterpolator(n_e_data, F_data, extrapolate=True)
        F = spline(n_bar)  # NaN outside range
        # Clamp to endpoints (flat tails)
        x0, x1 = n_e_data[0], n_e_data[-1]
        y0, y1 = F_data[0], F_data[-1]
        F = np.where(n_bar < x0, y0, F)
        F = np.where(n_bar > x1, y1, F)
        # Final safety
        F = np.where(np.isfinite(F), F, y1)
        return np.clip(spline(n_bar), 1.0, 20.0)

    def _sample_nr_energy_poisson(self, E_nr_eV):
        """Julian/CDMS Poisson scaling: λ = n_bar/F, draw k~Pois(λ), n = F*k."""

        Y = self._yield_model(E_nr_eV)                # quench to ionization mean
        E_ion_mean = E_nr_eV * Y
        n_bar = E_ion_mean /u.e2eV
        F = self._fano_factor_NR(n_bar)
        lam = np.maximum(n_bar / F, 1e-6)
        k = np.random.poisson(lam)
        n = F * k
        E_ion = np.clip(n * u.e2eV, 0.0, None)
        return E_ion, F, n_bar

    def _sample_nr_energy_gauss(self, E_nr_eV):
        """Gaussian with Var(E) = F * E_mean * eps_eh (same mean/variance, continuous)."""
        Y = self._yield_model(E_nr_eV)
        E_ion_mean = E_nr_eV * Y
        n_bar = E_ion_mean / u.e2eV
        F = self._fano_factor_NR(n_bar)
        sigma_E = np.sqrt(F * E_ion_mean *u.e2eV)
        E_ion = np.random.normal(E_ion_mean, sigma_E)
        E_ion = np.clip(E_ion, 0.0, None)
        return E_ion, F, n_bar

    def execute_process(self,hits):

        if self.model == "gauss":
            self.execute_process_gauss(hits)
        else:
            self.execute_process_karthik(hits)

    def execute_process_gauss(self,hits):
        """:obj:`Diffusion.execute_process` is a method to model the charge pair creation
        and charge diffusion for each energy loss given by the numpy.ndarray `hits.Edep`.

        While the parameters of the model :math:`A`, :math:`B` and :math:`z_{offset}` were obtained by
        assuming units of **um**, the positions for hits are in **mm** (default units of geant4).
        **Internally the positions will be converted to um, but stored in mm.**

        Parameters
        ----------
        hits : :obj:`pysimdamicm.utils.data_foramts.G4HitCollection`
            A collection of hits (subset of hits from the geant4
            simulations) with at least the following attributes: time, and
            `__column_axis__`.

        Returns
        -------
        hits : :obj:`pysimdamicm.utils.data_foramts.G4HitCollection`
            with actualized values of x,y,z, Edep, time and pdg

            Units (as the inputs ones):
                * Edep in eV
                * position in mm
                * time in seconds
                * pdg adimensional

        Note
        ----
        **hits** must have **x**, **y**, **z** and **Edep** as attributes, if not an
        **AttributeError** will be raised.

        """
        self.load_sigma_map_lazy()

        ###Kush Quenching Patch##
        E = np.asarray(hits.Edep, float)
        E_eff = E.copy()
        n_obs_e = np.zeros_like(E, dtype=float)
        pdg = np.asarray(getattr(hits, 'pdg', np.zeros_like(E, int))).astype(int)

        # -----------------------------
        # Identify nuclear recoils
        # -----------------------------
        is_ion = ((pdg // 10) >= 100014028) & ((pdg // 10) <= 100014030)
        is_neutron = pdg == 2112
        is_proton  = pdg == 2212
        is_NR = is_ion | ((is_neutron | is_proton) & (E <= self.prod_cut_eV))
        mask_ER = ~is_NR
        has_NR, has_ER = np.any(is_NR), np.any(mask_ER)

        # -----------------------------
        # Apply quenching + Fano to NR
        # -----------------------------
        if self.quenching and has_NR:
            E_ion_NR, F_NR, nbar_NR = self._sample_nr_energy_gauss(E[is_NR])
            E_eff[is_NR] = E_ion_NR
            n_obs_e[is_NR] = np.round(E_ion_NR / u.e2eV).astype(int)
            n_obs_e[n_obs_e < 0] = 0

        # -----------------------------
        # Apply standard Gaussian to ER
        # -----------------------------
        if has_ER:
            mu_ER = hits.Edep[mask_ER] / u.e2eV
            sigma_ER = np.sqrt(self.fanofactor * np.maximum(mu_ER, 0.0))
            n_e_ER = np.round(np.random.normal(mu_ER, sigma_ER)).astype(int)
            n_e_ER[n_e_ER < 0] = 0
            n_obs_e[mask_ER] = n_e_ER

        # --- Treat nuclear recoils as ER when quenching is off ---
        if not self.quenching and has_NR:
            mu_NR = hits.Edep[is_NR] / u.e2eV
            sigma_NR = np.sqrt(self.fanofactor * np.maximum(mu_NR, 0.0))
            n_NR = np.round(np.random.normal(mu_NR, sigma_NR)).astype(int)
            n_NR[n_NR < 0] = 0
            n_obs_e[is_NR] = n_NR

        # Update quenched energy array
        hits.Edep = pd.Series(E_eff)

        # -----------------------------
        # DEBUG 1 – before PCC/diffusion
        # -----------------------------
        if self.quenching_debug and (has_NR or has_ER):
            event_type = "MIXED" if (has_NR and has_ER) else ("NR-only" if has_NR else "ER-only")
            print(f"\n[QF DEBUG 1] — Event type: {event_type}")
            header = f"{'i':>5} | {'PDG':>12} | {'Type':>3} | {'E_before[eV]':>15} | {'E_after[eV]':>15} | {'QF':>6} | {'n_e_preDiff':>15}"
            print(header)
            print("-" * len(header))
            nr_idx = np.where(is_NR)[0]
            er_idx = np.where(~is_NR)[0]
            show_idx = np.concatenate([nr_idx[:3], er_idx[:3]])
            for i in show_idx:
                kind = "NR" if is_NR[i] else "ER"
                e_before = E[i]
                e_after  = E_eff[i]
                qf_val = (e_after / e_before) if (kind == "NR" and e_before > 0) else 1.0
                print(f"{i:5d} | {pdg[i]:>12d} | {kind:>3} | {e_before:15.3f} | {e_after:15.3f} | {qf_val:6.3f} | {n_obs_e[i]:15.3f}")

        ### Quenching End ###

        n_obs_e = n_obs_e.astype(int)

        if self.apply_pcc == 1:
            z_um = hits.z *(u.mm/u.um)  # Convert mm to microns
            #print("[PCC] Applied to hits at z (µm):", z_um[:10])
            eps = self.eps_func(z_um)
            # Identify hits in the PCC transition region
            low_z_mask = (z_um >= 663.0) & (z_um <= 669.0)

            # If debug is enabled, and first time this region appears, print BEFORE scaling
            if getattr(self, 'pcc_debug', 0) == 1:
                if not hasattr(Diffusion, 'pcc_debug_printed') and np.any(low_z_mask):
                    print(f"[PCC DEBUG] Found {np.sum(low_z_mask)} hits with z ∈ [663, 669] µm")
                    print("z_um[:10]:", z_um[low_z_mask][:10])
                    print("eps[:10]:", eps[low_z_mask][:10])
                    print("n_obs_e (before):", n_obs_e[low_z_mask][:10])

            n_obs_e = np.round(n_obs_e * eps).astype(int)
            # If debug is enabled, print AFTER scaling (once)
            if getattr(self, 'pcc_debug', 0) == 1:
                if not hasattr(Diffusion, 'pcc_debug_printed') and np.any(low_z_mask):
                    print("n_obs_e (after):", n_obs_e[low_z_mask][:10])
                    Diffusion.pcc_debug_printed = True


        ##########DIFFUSION PROCESS ###################################################################
        ## Only if the process is active in the json file, i.e., if "only_pair_creation is equal" = 0

        if(self.only_pair_creation==0):
            ### GAUSSIAN DISTRIBUTION OF THE E-H PAIR AROUND THE NOMINAL POSITION
            ###########################################################################################
            ### variance proportional to z, allow a value of z, where to start for fully diffusion
            ### for values below z_offset no diffusion applied (sigma=0)
            var_xy=np.zeros_like( hits.z )
            mask_zoffset=hits.z>self.z_offset

            var_xy[np.array(mask_zoffset)]=-1.0*self.A*np.log(1.0-self.B*(hits.z[mask_zoffset]-self.z_offset))
            # check for negative values of var_xy
            var_xy[var_xy<=0.0]=0

            # Note that for depth EQUAL to 1000 um the log is -inf, and nan for those depth > 1000.0u
            # however the depth of the CCD, i.e. the silicon bulk will never be as much as that
            if (var_xy==np.inf).any() or (var_xy==np.nan).any():
                print(" ---- ")
                print("\033[33;m DiffuseElectrons :: Informative WARNING\033[1;m")
                print(" This success (eventid+ccdid) has some energy depositons at the maximum thickness of the CCD")
                print("\t The variance for this success has been set to the `maximum variance`, pixel size")
                print("\t The problem is comming from the simulations, where an event at the edge has no sense!")
                print(" ---- ")
                var_xy[np.logical_or(var_xy == np.inf, var_xy == np.nan)] =  cfg.xpixelsize

            ### place all e-h pair created at the nominal position
            nominal_posx_e=np.repeat(hits.x,n_obs_e)
            nominal_posy_e=np.repeat(hits.y,n_obs_e)
            varxy_e = np.repeat(var_xy,n_obs_e)

            sigma_xy_e = np.sqrt(varxy_e)

            # --- Apply precomputed sigma map if repulsion model is on ---
            if (getattr(self, 'repulsion', 0) == 1
                    and getattr(self, 'sigma_map_loaded', False)
                    and hits.Edep.size > 0
                    and np.sum(hits.Edep.values) < 20000.0):
                # Energy per electron cloud in keV [and go from eV to keV]
                nominal_E = np.repeat(hits.Edep.values, n_obs_e) / 1e3

                # Depth per electron cloud in microns (from mm), then to meters for the map
                # mm ---> um ---> m
                z_um = np.repeat(hits.z.values, n_obs_e) * (u.mm / u.um)
                z_m  = z_um * 1e-6

                try:
                    # returns σ in µm if your map stores µm; adjust next line accordingly
                    sigma_from_map = self.sigma_interpolator(np.column_stack([nominal_E, z_m]))
                    # convert: um to mm
                    sigma_xy_e = np.asarray(sigma_from_map, dtype=float) * (u.um / u.mm)
                except Exception as e:
                    print(f"[Diffusion] Repulsion map interpolation failed: {e}")

            ### add energy dependence
            else:
                ### repulsion ==0 so, execute old energy dependent model ###
                nominal_E = np.repeat(hits.Edep.values,n_obs_e)
                sigma_xy_e_zmax = np.sqrt( -1.0*self.A*np.log(1.0-self.B*(self.z_max-self.z_offset)) )
                beta = self.alpha*(u.pix2mm*(u.um/u.mm)) / sigma_xy_e_zmax
                sigma_xy_e = sigma_xy_e * (self.alpha0 + beta*nominal_E )

            ### add also the sigma_xy as an attribure of the hits object
            hits.sigma_xy = sigma_xy_e

            ### shift each e-h pair created from its nominal position
            hits.x = np.random.normal(nominal_posx_e,sigma_xy_e)
            hits.y = np.random.normal(nominal_posy_e,sigma_xy_e)

        ########NO DIFFUSION PROCESS, if "only_pair_creation" = 1 ###################

        else:
            ### place all e-h pair created at the nominal position
            nominal_posx_e=np.repeat(hits.x,n_obs_e)
            nominal_posy_e=np.repeat(hits.y,n_obs_e)
            hits.x = nominal_posx_e
            hits.y = nominal_posy_e



        ## the depth for each e-h pair created given an energy lost, is equal
        ##  to the nominal z of the Edep
        hits.z = np.repeat(hits.z,n_obs_e)

        ### each e-h pair created corresponds to one carried charge, i.e. 1
        hits.Edep = np.ones_like(hits.x)

        ### for each e-h pair, add (if needed) its attributes: PDG and time
        if hasattr(hits,'pdg'):
            pdg_electrons = np.repeat(hits.pdg.values,n_obs_e)
            hits.pdg = pdg_electrons
        if hasattr(hits,'time'):
            time_electrons = np.repeat(hits.time.values,n_obs_e)
            hits.time = time_electrons

        ### include the number of electron-hole pairs for each energy deposition
        ### this will be used during pixelization
        hits.N_carried_charges = hits.Edep
        ### go to internal units of Edep, the rest are already in the internal units
        hits.Edep = hits.Edep*u.e2eV

        if hits.Edep.size == 0:
            hits.killed = True

    def execute_process_karthik(self,hits):
        """:obj:`Diffusion.execute_process` is a method to model the charge pair creation
        and charge diffusion for each energy loss given by the numpy.ndarray `hits.Edep`.

        While the parameters of the model :math:`A`, :math:`B` and :math:`z_{offset}` were obtained by
        assuming units of **um**, the positions for hits are in **mm** (default units of geant4).
        **Internally the positions will be converted to um, but stored in mm.**

        Parameters
        ----------
        hits : :obj:`pysimdamicm.utils.data_foramts.G4HitCollection`
            A collection of hits (subset of hits from the geant4
            simulations) with at least the following attributes: time, and
            `__column_axis__`.

        Returns
        -------
        hits : :obj:`pysimdamicm.utils.data_foramts.G4HitCollection`
            with actualized values of x,y,z, Edep, time and pdg

            Units (as the inputs ones):
                * Edep in eV
                * position in mm
                * time in seconds
                * pdg adimensional

        Note
        ----
        **hits** must have **x**, **y**, **z** and **Edep** as attributes, if not an
        **AttributeError** will be raised.

        """

        ######################################################################################## CHARGE GENERARION: KARTHICK MODEL
        self.load_sigma_map_lazy()

        ######################################################################################## KUSH QUENCHING PATCH
        E = np.asarray(hits.Edep, dtype=float)
        E_eff = E.copy()
        n_obs_e = np.zeros_like(E, dtype=float)
        pdg = np.asarray(getattr(hits, 'pdg', np.zeros_like(E, int))).astype(int)

        # -----------------------------
        # Identify nuclear recoils
        # -----------------------------
        is_ion = ((pdg // 10) >= 100014028) & ((pdg // 10) <= 100014030)
        is_neutron = pdg == 2112
        is_proton = pdg == 2212

        # NR if ion, or n/p with E <= prod_cut_eV
        is_NR = is_ion | ((is_neutron | is_proton) & (E <= self.prod_cut_eV))
        mask_ER = ~is_NR
        has_NR = np.any(is_NR)
        has_ER = np.any(mask_ER)

        # -----------------------------
        # Apply quenching to NR only
        # -----------------------------
        if self.quenching and has_NR:
            E_ion_NR, F_NR, nbar_NR = self._sample_nr_energy_poisson(E[is_NR])
            n_obs_e[is_NR] = np.round(E_ion_NR / u.e2eV).astype(int)
            n_obs_e[n_obs_e < 0] = 0
            E_eff[is_NR] = E_ion_NR
        else:
            E_eff = E.copy()

        # Overwrite Edep globally so downstream diffusion/PCC use quenched energies
        hits.Edep = pd.Series(E_eff)

        # -----------------------------
        # DEBUG 1 – before Karthik
        # -----------------------------
        if self.quenching_debug and (has_NR or has_ER):
            event_type = "MIXED" if (has_NR and has_ER) else ("NR-only" if has_NR else "ER-only")
            print(f"\n[QF DEBUG 1] Quenching applied before Karthik — Event type: {event_type}")
            header = f"{'i':>5} | {'PDG':>12} | {'Type':>3} | {'E_before[eV]':>15} | {'E_after[eV]':>15} | {'QF':>6} | {'n_e_preKarthik':>15}"
            print(header)
            print("-" * len(header))

            nr_idx = np.where(is_NR)[0]
            er_idx = np.where(~is_NR)[0]
            show_idx = np.concatenate([nr_idx[:3], er_idx[:3]])

            for i in show_idx:
                kind = "NR" if is_NR[i] else "ER"
                e_before = E[i]
                e_after = E_eff[i]
                qf_val = (e_after / e_before) if (kind == "NR" and e_before > 0) else 1.0
                print(f"{i:5d} | {pdg[i]:>12d} | {kind:>3} | {e_before:15.3f} | {e_after:15.3f} | {qf_val:6.3f} | {n_obs_e[i]:15.3f}")

        # Temporarily hide NR energies/electrons so Karthik only processes ER.
        Edep_quenched = hits.Edep.copy()
        nobs_backup = n_obs_e.copy()

        if self.quenching and has_NR and has_ER:
            hits.Edep[is_NR] = 0.0
            n_obs_e[is_NR] = 0.0
        ######################################################################################## END QUENCHING PATH


        sumEnergy = hits.Edep.sum()
        if sumEnergy > 50.:
            #The number of electrons worth considering varies. Higher energy deposits have
            # a very wide spread of possible numbers of electrons produced. If n is round(E/e2eV),
            # we want to calculate what values of m give us p_m(E) / p_n(E) ~= 0.05, and make
            # our range of test values span this interval, as this would give us a decent sample
            # of electron numbers that have a reasonable chance of being produced. For very high
            # energies this could result in hundreds of values being tested, but this is to be expected
            n_eh_avg = np.round(sumEnergy/u.e2eV)
            prob_const = np.sqrt(((n_eh_avg * u.e2eV - sumEnergy)/(u.e2eV * np.sqrt(n_eh_avg *
                self.fanofactor))) ** 2 - 2 * np.log(0.05))
            upper_bound = np.round((2 * sumEnergy + self.fanofactor * prob_const ** 2 * u.e2eV + prob_const *
                np.sqrt((self.fanofactor * prob_const * u.e2eV)** 2 + 4 * sumEnergy * self.fanofactor * u.e2eV)) / (2 * u.e2eV))
            lower_bound = np.round((2 * sumEnergy + self.fanofactor * prob_const ** 2 * u.e2eV - prob_const *
                np.sqrt((self.fanofactor * prob_const * u.e2eV) ** 2 + 4 * sumEnergy * self.fanofactor * u.e2eV)) / (2 * u.e2eV))
            n_eh_range =  np.arange(lower_bound, upper_bound, dtype = int)

            n_eh_probs = (1 / np.sqrt(2 * np.pi * n_eh_range * self.fanofactor)) * np.exp((-1 / (2 * n_eh_range * self.fanofactor)) *
                    ((n_eh_range * u.e2eV - sumEnergy) / u.e2eV) ** 2)
            n_eh_probs = n_eh_probs / n_eh_probs.sum()

        else:
            sumEnergy = np.round(sumEnergy / 0.05) * 0.05
            if sumEnergy < 1.1:
                sumEnergy = 1.1
            hit_energy = np.searchsorted(self.ehenergies, sumEnergy)
            n_eh_probs = self.ehprobs[hit_energy]
            n_eh_range = np.arange(1, 21)

        if np.sum(n_eh_probs) > 0:
            sum_obs_e = np.random.choice(n_eh_range, p = n_eh_probs)
            hits_Edep_np = hits.Edep.to_numpy().flatten()
            electron_Dist = hits_Edep_np / hits_Edep_np.sum()
            indices = np.arange(hits_Edep_np.size)
            ### Two methods for distribution of electrons. As of yet undecided which...
            # Method 1: full numbers of electrons in pixels are certain, the remainder is probabilistic
            n_obs_e = np.floor(sum_obs_e * electron_Dist)
            electron_Remainder = sum_obs_e - n_obs_e.sum(dtype = int)
            if electron_Remainder > 0:
                try:
                    electron_Dist_Remainder = (sum_obs_e * electron_Dist - n_obs_e) / (sum_obs_e * electron_Dist - n_obs_e).sum()
                    added_indices = np.histogram(np.random.choice(indices, size = electron_Remainder, p = electron_Dist_Remainder), bins = np.append(indices, indices.size))[0]
                    n_obs_e = (n_obs_e + added_indices).astype(int)
                except:
                    print("sumEnergy: {}".format(sumEnergy))
                    print("n_eh_range: {}".format(n_eh_range))
                    print("n_eh_probs: {}".format(n_eh_probs))
                    print("sum_obs_e: {}".format(sum_obs_e))
                    print("n_obs_e: {}".format(n_obs_e))
                    print("electron_Remainder: {}".format(electron_Remainder))
        else:
            sum_obs_e = 0
            n_obs_e = np.zeros(hits.Edep.to_numpy().size)


        # ===============================================================
        # Kush Quenching Patch: restore quenched energies and NR electrons
        # ===============================================================
        if self.quenching and has_NR:
            hits.Edep = Edep_quenched
            n_obs_e[is_NR] = nobs_backup[is_NR]

        # -----------------------------
        # DEBUG 2 – after Karthik
        # -----------------------------
        if self.quenching_debug and (has_NR or has_ER):
            print(f"\n[QF DEBUG 2] Final restored state after Karthik — Event type: {event_type}")
            header = f"{'i':>5} | {'PDG':>12} | {'Type':>3} | {'E_final[eV]':>15} | {'n_e_final':>12} | {'Source':>20}"
            print(header)
            print("-" * len(header))

            nr_idx = np.where(is_NR)[0]
            er_idx = np.where(~is_NR)[0]
            show_idx = np.concatenate([nr_idx[:3], er_idx[:3]])

            for i in show_idx:
                kind = "NR" if is_NR[i] else "ER"
                src = "Deterministic NR" if kind == "NR" else "Karthik ER"
                e_final = hits.Edep[i]
                print(f"{i:5d} | {pdg[i]:>12d} | {kind:>3} | {e_final:15.3f} | {n_obs_e[i]:12.3f} | {src:>20}")

        ## patch ENDS

        ########################################################################################3 DIFFUSE EACH e- TO THE SURFACE
        n_obs_e = np.asarray(n_obs_e, dtype=int)
        n_obs_e[n_obs_e < 0] = 0
        n_obs_e = pd.Series(n_obs_e)

        if self.apply_pcc == 1:
             z_um = hits.z *(u.mm/u.um)  # Convert mm to microns
             #print("[PCC] Applied to hits at z (µm):", z_um[:10])
             eps = self.eps_func(z_um)
             # Identify hits in the PCC transition region
             low_z_mask = (z_um >= 663.0) & (z_um <= 669.0)

             # If debug is enabled, and first time this region appears, print BEFORE scaling
             if getattr(self, 'pcc_debug', 0) == 1:
                if not hasattr(Diffusion, 'pcc_debug_printed') and np.any(low_z_mask):
                       print(f"[PCC DEBUG] Found {np.sum(low_z_mask)} hits with z ∈ [663, 669] µm")
                       print("z_um[:10]:", z_um[low_z_mask][:10])
                       print("eps[:10]:", eps[low_z_mask][:10])
                       print("n_obs_e (before):", n_obs_e[low_z_mask][:10])

             n_obs_e = np.round(n_obs_e * eps).astype(int)
             # If debug is enabled, print AFTER scaling (once)
             if getattr(self, 'pcc_debug', 0) == 1:
                if not hasattr(Diffusion, 'pcc_debug_printed') and np.any(low_z_mask):
                     print("n_obs_e (after):", n_obs_e[low_z_mask][:10])
                     Diffusion.pcc_debug_printed = True

        ##########DIFFUSION PROCESS ###################################################################
        ## Only if the process is active in the json file, i.e., if "only_pair_creation is equal" = 0
        if(self.only_pair_creation==0):
            ### GAUSSIAN DISTRIBUTION OF THE E-H PAIR AROUND THE NOMINAL POSITION
            ###########################################################################################
            ### variance proportional to z, allow a value of z, where to start for fully diffusion
            ### for values below z_offset no diffusion applied (sigma=0)
            var_xy=np.zeros_like( hits.z )
            mask_zoffset=hits.z>self.z_offset

            var_xy[np.array(mask_zoffset)]=-1.0*self.A*np.log(1.0-self.B*(hits.z[mask_zoffset]-self.z_offset))
            # check for negative values of var_xy
            var_xy[var_xy<=0.0]=0

            # Note that for depth EQUAL to 1000 um the log is -inf, and nan for those depth > 1000.0u
            # however the depth of the CCD, i.e. the silicon bulk will never be as much as that
            if (var_xy==np.inf).any() or (var_xy==np.nan).any():
                print(" ---- ")
                print("\033[33;m DiffuseElectrons :: Informative WARNING\033[1;m")
                print(" This success (eventid+ccdid) has some energy depositons at the maximum thickness of the CCD")
                print("\t The variance for this success has been set to the `maximum variance`, pixel size")
                print("\t The problem is comming from the simulations, where an event at the edge has no sense!")
                print(" ---- ")
                var_xy[np.logical_or(var_xy == np.inf, var_xy == np.nan)] =  cfg.xpixelsize

            ### place all e-h pair created at the nominal position
            nominal_posx_e=np.repeat(hits.x,n_obs_e)
            nominal_posy_e=np.repeat(hits.y,n_obs_e)
            varxy_e = np.repeat(var_xy,n_obs_e)

            sigma_xy_e = np.sqrt(varxy_e)

            # --- Apply precomputed sigma map if repulsion model is on ---
            if (getattr(self, 'repulsion', 0) == 1
                    and getattr(self, 'sigma_map_loaded', False)
                    and hits.Edep.size > 0
                    and np.sum(hits.Edep.values) < 20000.0):

                # Energy per electron cloud in keV [convert eV to keV]
                nominal_E = np.repeat(hits.Edep.values, n_obs_e) / 1e3

                # Depth per electron cloud in microns (from mm), then to meters for the map
                # mm -- um -- m
                z_um = np.repeat(hits.z.values, n_obs_e) * (u.mm / u.um)
                z_m  = z_um * 1e-6

                try:
                    # returns σ in µm if your map stores µm; adjust next line accordingly
                    sigma_from_map = self.sigma_interpolator(np.column_stack([nominal_E, z_m]))

                    # Convert map σ to millimeters for the rest of the pipeline.
                    # If your sigma_map values are µm: multiply by (u.um / u.mm) = 1e-3
                    # um --> mm
                    sigma_xy_e = np.asarray(sigma_from_map, dtype=float) * (u.um / u.mm)

                except Exception as e:
                    print(f"[Diffusion] Repulsion map interpolation failed: {e}")
            else:
                ### add energy dependence
                nominal_E = np.repeat(hits.Edep.values,n_obs_e)
                sigma_xy_e_zmax = np.sqrt( -1.0*self.A*np.log(1.0-self.B*(self.z_max-self.z_offset)) )
                beta = self.alpha*(u.pix2mm*(u.um/u.mm)) / sigma_xy_e_zmax
                sigma_xy_e = sigma_xy_e * (self.alpha0 + beta*nominal_E )

            ### add alsot the sigma_xy as an attribure of the hits object
            hits.sigma_xy = sigma_xy_e


            ### shift each e-h pair created from its nominal position
            hits.x = np.random.normal(nominal_posx_e,sigma_xy_e)
            hits.y = np.random.normal(nominal_posy_e,sigma_xy_e)

        else:
            ### place all e-h pair created at the nominal position
            nominal_posx_e=np.repeat(hits.x,n_obs_e)
            nominal_posy_e=np.repeat(hits.y,n_obs_e)
            hits.x = nominal_posx_e
            hits.y = nominal_posy_e



        ## the depth for each e-h pair created given an energy lost, is equal
        ##  to the nominal z of the Edep
        hits.z = np.repeat(hits.z,n_obs_e)

        ### each e-h pair created corresponds to one carried charge, i.e. 1
        hits.Edep = np.ones_like(hits.x)

        ### for each e-h pair, add (if needed) its attributes: PDG and time
        if hasattr(hits,'pdg'):
            pdg_electrons = np.repeat(hits.pdg.values,n_obs_e)
            hits.pdg = pdg_electrons
        if hasattr(hits,'time'):
            time_electrons = np.repeat(hits.time.values,n_obs_e)
            hits.time = time_electrons

        ### include the number of electron-hole pairs for each energy deposition
        ### this will be used during pixelization
        hits.N_carried_charges = hits.Edep
        ### go to internal units of Edep, the rest are already in the internal units
        hits.Edep = hits.Edep*u.e2eV

        if hits.Edep.size == 0:
            hits.killed = True


###############################################################################################
#####       CONTINUOUS READOUT MODEL
###############################################################################################
class ContinuousReadout(DigitizeProcess):
    """Class to model the continuous readout (electronics process).

    While a pixel is readed, the collected charge (from the previous pixel)
    suffers a shift along the columns (y-axis), and part of this collected
    charge is
    transfered to the following pixel different from the pixel where charge
    was collected.

    **List of Attributes:**

    Attributes
    ----------

    pixel_read_time

        Time to read a pixel in seconds, default 0.001

    ampli

        Number of amplifiers used to read the full CCD, default 4

    __column_axis__

        Axes where a shift on the position will be applyied due to readout
        continuous effects, default is y

    __sequence_id__

        Unique identifier for the reconstruction process, default is 20

        Used by `ProcessManager.execute_process` to sort all activated
        process, i.e. included in the full reconstruction process contained
        by `ProcessManager.__sequence__`.


    Note
    ----
    Assuming
        * y-axis as the columns of the CCD (horitzontal register)
        * x-axis as the rows (serial register), and
        * z-axis as the direction of the electric field in the dopped region of the silicon detector.
        * Geant4 units as default units: mm, eV, seconds

    """
    __sequence_id__ = 20
    __column_axis__ = 'y'
    __name__ = 'ContinuousReadout'

    def __init__(self):
        super().__init__()

        # parameters of the model
        # time to read one pixel in seconds
        self.pixel_read_time = float(0.001)*u.s
        self.ampli = int(4)
        self.__init__={'pixel_read_time':u.s,'ampli':1}

    def execute_process(self, hits):
        """Class method to apply a shift on the column position due to
        readout time

        Parameters
        ----------

        hits : :obj:`pysimdamicm.utils.data_foramts.G4HitCollection`
            A collection of hits (subset of hits from the geant4
            simulations) with at least the following attributes: x, y, z and Edep.

        Raises
        ------
        AttributeError
            When hits do not have __column_axis__ as un attribute

        """

        if ContinuousReadout.pixel_read_time > 0.0:
            time_zero = 0.0
            pos_time_shift = np.floor( (hits.time - time_zero)/self.pixel_read_time )
        else:
            return

        values_axes = (getattr(hits,__column_axis__))
        setattr(hits,__column_axis__, values_axes+pos_time_shift)

        if __verbose__:
            print( "VERBOSE ContinuousReadout.execute_process: \n shift on {0} is {1}".format(
                __axes_shift__,pos_time_shift) )




###############################################################################################
#####       PIXELIZE SIGNAL PROCESS
###############################################################################################
class PixelizeSignal(DigitizeProcess):
    """Class for the pixelization process to convert continuos positions
    (in mm) to pixel units.

    This is a generalization of a histogram2d function. A histogram divides
    the space into bins, and returns the count of the number of points in
    each bin. `PixelizeSignal.execute_process` allows the computation of
    the sum, mean, median, or other statistics of a set of values
    (x,y,z,Edep,[time], [pdg]) within each bin.

    This process must be applied after electron diffusion and continuous
    readout processes (if applicable).

    Assume Geant4 units: mm, eV, seconds

    **List of Attributes:**

    Attributes
    ----------

    N_pixels_in_x

        Number of pixels on the x-axis (columns) within the active region
        of the CCD, default 6000 pixels

    shift_pixel_in_x

        Number of pixels on the x-axis to skip (non-active silicon region),
        default 0 pixels

    pixel_size_in_x

        Size of the pixel on the x-axis in mm, default 0.015 mm

    N_pixels_in_y

        Number of pixels on the y-axis (rows) within the active region of
        the CCD, default 6000 pixels

    shift_pixel_in_y

        Number of pixels on the y-axis to skip (non-active silicon region),
        default 0

    pixel_size_in_y

        Size of the pixel on the y-axis in mm, default 0.015 mm


    Note
    ----
    Assuming
        * y-axis as the columns of the CCD (horitzontal register)
        * x-axis as the rows (serial register), and
        * z-axis as the direction of the electric field in the dopped region of the silicon detector.

    So the pixelization is done on the xy-plane, without altering the depth values (x-axis)

    """
    __name__ = 'PixelizeSignal'

    ### Unique identifier for the full reconstruction process.
    __sequence_id__ = 30

    __statistic_function_for_Edep__ = 'sum'

    __statistic_function_for_depth = 'mean'
    __DoDepth = True
    __statistic_function_for_pdg = None
    __DoPDG = True
    __statistic_function_for_time = 'min'
    __DoTime = True


    __Edep_in_e__ = False

    # XXX POSSIBLE value to select part of the e-h pair created XXX
    __charge_threshold = 0.0

    ### others
    #__AXES = ['x','y']
    __AXES = ['y','x']

    def __init__(self):
        super().__init__()

        # parameters of the model
        self.shift_pixel_in_x = int(0)*u.pixel

        self.shift_pixel_in_y = int(0)*u.pixel

        self.unit = "pixel"

        self.__units__={'shift_pixel_in_x':u.pixel,'shift_pixel_in_y':u.pixel}

    def get_ccd_shape(self):
        """Read the CCD dimensions from the attribute ccd_shape in singleton Units

            Read as  ccd_shape = (rows,cols) = (y-axis, x-axis)
        """
        self.N_pixels_in_y,self.N_pixels_in_x = u.ccd_shape
        self.pixel_size_in_x = u.ccd_pixel_size_x
        self.pixel_size_in_y = u.ccd_pixel_size_y

    def execute_process(self, hits):
        """Pixelize all hit attributes using a single NumPy grouping pass.


        The numerical definitions are kept equivalent to the previous
        implementation:

        * energy and carried charge are summed;
        * depth and diffusion sigma are averaged;
        * the PDG code is the most frequent value, with the lowest PDG code
          selected when several values have the same multiplicity;
        * particle-origin flags indicate whether at least one matching PDG
          code is present;
        * ``time_pixel`` is energy weighted and ``std_time_pixel`` uses the
          population standard deviation (``ddof=0``).
        """
        self.get_ccd_shape()

        if self.__statistic_function_for_Edep__ != 'sum':
            raise NotImplementedError(
                    "The NumPy pixelizer currently supports only sum for Edep."
                    )
        if self.__statistic_function_for_depth != 'mean':
            raise NotImplementedError(
                    "The NumPy pixelizer currently supports only mean for depth."
                    )

        # Convert continuous coordinates to integer pixel indices.  For the
        # active region, floor is equivalent to the positive-coordinate
        # binning used by the previous SciPy implementation.
        x_pixel_all = np.floor(
                np.asarray(hits.x, dtype=float) / self.pixel_size_in_x
                ).astype(np.int64)
        y_pixel_all = np.floor(
                np.asarray(hits.y, dtype=float) / self.pixel_size_in_y
                ).astype(np.int64)

        # Drop out-of-detector charges before grouping.  The inclusive upper
        # bound deliberately preserves the behaviour of
        # pixels_within_active_region().  It can be changed separately after
        # validating the possible off-by-one issue in that method.
        active_mask = (
                (x_pixel_all >= self.shift_pixel_in_x)
                & (x_pixel_all <= self.N_pixels_in_x + self.shift_pixel_in_x)
                & (y_pixel_all >= self.shift_pixel_in_y)
                & (y_pixel_all <= self.N_pixels_in_y + self.shift_pixel_in_y)
                )

        if not np.any(active_mask):
            hits.x_pixel = np.empty(0, dtype=np.int64)
            hits.y_pixel = np.empty(0, dtype=np.int64)
            hits.Edep_pixel = np.empty(0, dtype=float)
            hits.killed = True
            hits.__attr_to_clusterize__ += "_pixel"
            return

        x_pixel = x_pixel_all[active_mask]
        y_pixel = y_pixel_all[active_mask]

        # Encode each (y, x) pair as one integer.  np.unique then returns a
        # stable lexicographic ordering equivalent to np.where on the old 2-D
        # statistic array: first y, then x.
        n_x_slots = self.N_pixels_in_x + 1
        flat_pixel = (
                (y_pixel - self.shift_pixel_in_y) * n_x_slots
                + (x_pixel - self.shift_pixel_in_x)
                )
        unique_pixel, inverse, counts = np.unique(
                flat_pixel,
                return_inverse=True,
                return_counts=True,
                )
        n_groups = unique_pixel.size

        grouped_y = unique_pixel // n_x_slots + self.shift_pixel_in_y
        grouped_x = unique_pixel % n_x_slots + self.shift_pixel_in_x

        edep = np.asarray(hits.Edep, dtype=float)[active_mask]
        edep_sum = np.bincount(inverse, weights=edep, minlength=n_groups)
        selected = edep_sum > self.__charge_threshold

        hits.x_pixel = grouped_x[selected].astype(np.int64, copy=False)
        hits.y_pixel = grouped_y[selected].astype(np.int64, copy=False)
        hits.Edep_pixel = edep_sum[selected]

        if hasattr(hits, 'N_carried_charges'):
            carried_charge = np.asarray(
                    hits.N_carried_charges, dtype=float
                    )[active_mask]
            carried_charge_sum = np.bincount(
                    inverse,
                    weights=carried_charge,
                    minlength=n_groups,
                    )
            hits.N_carried_charges_pixel = carried_charge_sum[selected].astype(int)

        if self.__DoDepth and hasattr(hits, 'z'):
            depth = np.asarray(hits.z, dtype=float)[active_mask]
            depth_sum = np.bincount(inverse, weights=depth, minlength=n_groups)
            hits.z_pixel = (depth_sum / counts)[selected]

        if hasattr(hits, 'sigma_xy'):
            sigma_xy = np.asarray(hits.sigma_xy, dtype=float)[active_mask]
            sigma_sum = np.bincount(
                    inverse,
                    weights=sigma_xy,
                    minlength=n_groups,
                    )
            hits.sigma_xy_pixel = (sigma_sum / counts)[selected]

        if self.__DoPDG and hasattr(hits, 'pdg'):
            pdg = np.asarray(hits.pdg)[active_mask]

            # Count every distinct (pixel group, PDG) pair.  Candidates are
            # then ordered by group, decreasing multiplicity and increasing
            # PDG code.  Taking the first candidate per group reproduces the
            # tie-breaking of np.unique(...)+np.argmax used previously.
            order = np.lexsort((pdg, inverse))
            group_sorted = inverse[order]
            pdg_sorted = pdg[order]

            pair_start_mask = np.empty(group_sorted.size, dtype=bool)
            pair_start_mask[0] = True
            pair_start_mask[1:] = (
                    (group_sorted[1:] != group_sorted[:-1])
                    | (pdg_sorted[1:] != pdg_sorted[:-1])
                    )
            pair_starts = np.flatnonzero(pair_start_mask)
            pair_counts = np.diff(np.append(pair_starts, group_sorted.size))
            pair_groups = group_sorted[pair_starts]
            pair_pdgs = pdg_sorted[pair_starts]

            # Locate the first distinct PDG pair belonging to every pixel group.
            group_pair_start_mask = np.empty(pair_groups.size, dtype=bool)
            group_pair_start_mask[0] = True
            group_pair_start_mask[1:] = (
                    pair_groups[1:] != pair_groups[:-1]
                    )
            group_pair_starts = np.flatnonzero(group_pair_start_mask)

            # Determine the highest PDG multiplicity in every pixel.
            max_pair_count = np.maximum.reduceat(
                    pair_counts,
                    group_pair_starts,
                    )

            # pair_pdgs is already sorted increasingly inside every group.
            # Therefore, the first maximum-multiplicity candidate also reproduces
            # the previous lowest-PDG tie breaking.
            mode_candidate_indices = np.flatnonzero(
                    pair_counts == max_pair_count[pair_groups]
                    )
            mode_candidate_groups = pair_groups[mode_candidate_indices]

            first_mode_candidate = np.empty(
                    mode_candidate_groups.size,
                    dtype=bool,
                    )
            first_mode_candidate[0] = True
            first_mode_candidate[1:] = (
                    mode_candidate_groups[1:]
                    != mode_candidate_groups[:-1]
                    )

            pdg_mode = np.empty(n_groups, dtype=pdg.dtype)

            pdg_mode[mode_candidate_groups[first_mode_candidate]] = pair_pdgs[mode_candidate_indices[first_mode_candidate]]

            hits.pdg_pixel = pdg_mode[selected]

#            def grouped_any(condition):
#                """Return one boolean per pixel group for a hit condition."""
#                return np.bincount(
#                        inverse,
#                        weights=np.asarray(condition, dtype=np.uint8),
#                        minlength=n_groups,
#                        ) > 0
#
#            neutron_flag = grouped_any(pdg == 2112)
#            gamma_flag = grouped_any(pdg == 22)
#            alpha_flag = grouped_any(pdg == 1000020040)
#            ion_flag = grouped_any(pdg > 1000000000)
#            electron_flag = grouped_any((pdg == 11) | (pdg == -11))
#
#            # Preserve the exact logic of the previous silicon flag.  The two
#            # conditions are intentionally evaluated independently before
#            # checking whether the most frequent PDG is an ion.
#            silicon_lower_flag = grouped_any(
#                    (pdg - 1000000000) >= 140280
#                    )
#            silicon_upper_flag = grouped_any(
#                    (pdg - 1000000000) <= 140309
#                    )
#            silicon_flag = (
#                    silicon_lower_flag
#                    & silicon_upper_flag
#                    & (pdg_mode >= 1000000000)
#                    )

            # Encode all particle-presence conditions in one byte per distinct
            # (pixel, PDG) pair.
            pair_flags = np.zeros(pair_pdgs.size, dtype=np.uint8)

            pair_flags |= (
                (pair_pdgs == 2112).astype(np.uint8) << 0
            )
            pair_flags |= (
                (pair_pdgs == 22).astype(np.uint8) << 1
            )
            pair_flags |= (
                (pair_pdgs == 1000020040).astype(np.uint8) << 2
            )
            pair_flags |= (
                (pair_pdgs > 1000000000).astype(np.uint8) << 3
            )
            pair_flags |= (
                ((pair_pdgs == 11) | (pair_pdgs == -11)).astype(np.uint8)
                << 4
            )

            shifted_pdg = pair_pdgs - 1000000000

            pair_flags |= (
                (shifted_pdg >= 140280).astype(np.uint8) << 5
            )
            pair_flags |= (
                (shifted_pdg <= 140309).astype(np.uint8) << 6
            )

            # Combine the flags of all distinct PDGs present in each pixel.
            group_flags = np.bitwise_or.reduceat(
                pair_flags,
                group_pair_starts,
            )

            neutron_flag = (group_flags & (1 << 0)) != 0
            gamma_flag = (group_flags & (1 << 1)) != 0
            alpha_flag = (group_flags & (1 << 2)) != 0
            ion_flag = (group_flags & (1 << 3)) != 0
            electron_flag = (group_flags & (1 << 4)) != 0
            silicon_lower_flag = (group_flags & (1 << 5)) != 0
            silicon_upper_flag = (group_flags & (1 << 6)) != 0
            silicon_flag = (
                    silicon_lower_flag
                    & silicon_upper_flag
                    & (pdg_mode >= 1000000000)
                    )



            hits.neutron_pixel = neutron_flag[selected].astype(float)
            hits.gamma_pixel = gamma_flag[selected].astype(float)
            hits.silicon_pixel = silicon_flag[selected].astype(float)
            hits.alpha_pixel = alpha_flag[selected].astype(float)
            hits.ion_pixel = ion_flag[selected].astype(float)
            hits.electron_pixel = electron_flag[selected].astype(float)

        if self.__DoTime and hasattr(hits, 'time'):
            hit_time = np.asarray(hits.time, dtype=float)[active_mask]

            weighted_time_sum = np.bincount(
                inverse,
                weights=edep * hit_time,
                minlength=n_groups,
            )
            time_pixel = np.divide(
                weighted_time_sum,
                edep_sum,
                out=np.full(n_groups, np.nan, dtype=float),
                where=edep_sum != 0,
            )
            hits.time_pixel = time_pixel[selected]

            # Population standard deviation, matching scipy.stats with
            # statistic='std'.  The centred second pass is more numerically
            # stable than E[t^2] - E[t]^2.
            time_sum = np.bincount(
                inverse,
                weights=hit_time,
                minlength=n_groups,
            )
            time_mean = time_sum / counts
            squared_residual_sum = np.bincount(
                inverse,
                weights=(hit_time - time_mean[inverse]) ** 2,
                minlength=n_groups,
            )
            hits.std_time_pixel = np.sqrt(
                squared_residual_sum / counts
            )[selected]
            hits.time_img_pixel = (
                time_pixel[selected] / u.img_exp_time
            ).astype(int)

        # All charges were filtered before grouping, so the output pixels are
        # already guaranteed to lie inside the active detector region.
        if hits.x_pixel.size == 0:
            hits.killed = True

        hits.__attr_to_clusterize__ += "_pixel"

    def pixels_within_active_region(self,hits):

        #print(" Restrictions ... x:{},{}, y:{},{}".format(self.shift_pixel_in_x,
        #    self.N_pixels_in_x+self.shift_pixel_in_x,
        #    self.shift_pixel_in_y, self.N_pixels_in_y+self.shift_pixel_in_y))

        ### x-axis: shift_pixel_in_x <= x_pixel <= N_pixels_in_x + shift_pixel_in_x
        mask_x=np.logical_and(hits.x_pixel>=self.shift_pixel_in_x,hits.x_pixel<=(self.N_pixels_in_x+self.shift_pixel_in_x))

        ### y-axis: shift_pixel_in_y <= y_pixel <= N_pixels_in_y + shift_pixel_in_y
        mask_y=np.logical_and(hits.y_pixel>=self.shift_pixel_in_y,hits.y_pixel<=(self.N_pixels_in_y+self.shift_pixel_in_y))

        ### both requirements are mandatory
        mask=np.logical_and(mask_x,mask_y)

        for attr in filter(lambda a:a.count("pixel")>0,hits.__dict__.keys()):
            setattr(hits,attr,getattr(hits,attr)[mask])

        if len(hits.x_pixel)==0:
            hits.killed = True


###############################################################################################
#####       DARK CURRENT MODEL
###############################################################################################
class DarkCurrent(DigitizeProcess):
    """Dark Current process.

    Most photodetectors (such as photodiodes, CCD sensors, ...) produce a signal current which is
    more or less proportional to the incident optical power. However, even in the absence of any
    light input, there is often some tiny amount of current, known as dark current.

    The DC current is often caused by thermal excitation (generation) of carriers --not necessarily
    directly from valence to conduction band, but possible through defect states related to crystal
    defects or impurities. The rate of such thermal processes depends not only on the active area,
    but also critically on the temperature and on the band gap energy of the material, and also on
    the operation voltage (particularly near the breakdown voltage, where impact ionization can
    occur). At high voltages, tunneling through the depletion region may also contribute.
    Dark currents may also be generated by some leakage currents which are not related to thermal
    excitation.

    .. math::
       :nowrap:

       \\begin{equation}
         I_{DC} \propto T^{\\frac{3}{2}} exp{ (\\frac{ -E_{g} }{K_{B}T}) }
       \end{equation}



    So, DC is a continuous thermal generation of electron-hole pairs in the depletion region. As the
    vias increases the depletion region grows and the dark current also grows. Once the whole wafer
    is depleted the dark current will be remain constant.

    Dark current is assumed to follow a poisson distribution where lambda is proportional to the
    exposure time of the CCD image.


    **List of Attributes:**

    Attributes
    ----------

        darkcurrent

            number of electrons per pixel per day

        exp_time

            exposure time of the dataset (time that CCDs have been exposed to ionizing events), in days

    row_readtime

            exposure time of a row, in days


        img_size

            only for running mode **cropped-image**
            Number of pixels in the x-axis to create the dark current image surrounding the hit collection of the event (squared image)

        min_enlarge

            minimum number of pixels to add when the image size is smaller than the cluster elongation.
            In this cases, the size of dark  current image will be <cluster_size> + <min_enlarge>

        mode

            Name of the running mode to generate the dark current image. Possible values:

                * 'full-image'

                    all events will share the same dark current image with the size of the CCD
                    (in this case img_size is ignored)

                * 'cropped-image'

                    for each hit collection, a new DC image will be created, in this case the
                    DC image should be smaller than the real CCD size, and img_size will be used
     exp_mode

        Method to calculate the mean DC:

                * 'exposure': the DC mean is calculated only on the base of the exp_time. Each pixel has the same exposure time.

                * 'exposure-per-row': the DC mean is calculated row by row, taking into account the row_readtime.



        n_rows_overscan
            number of extra pixels for the overscan region in rows (x-axis)

        n_rows_prescan
            number of extra pixels for the prescan region in rows (x-axis)

        n_cols_overscan
            number of extra pixels for the overscan region in cols (y-axis)

        n_cols_prescan
            number of extra pixels for the prescan region in cols (y-axis)


    Example
    -------

        >>> import pysimdamicm as dam
        >>> dc = dam.detector_response.DarkCurrent()
        >>> dc.__name__
        'DarkCurrent'
        >>> dc.__sequence_id__
        40
        >>> dc.__units__
        {'img_size': 1, 'mode': 1, 'darkcurrent': 1.1574074074074073e-05, 'exp_time': 86400.0,'min_enlarge': 1}
        >>> dc_params = {'exp_time':8/24., 'darkcurrent': 0.001 , 'mode':''}
        >>> dc.set_parameters(**dc_params)
        Running on mode: full-image
    >>> dc.dark_current_image
    array([[0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.],
           ...,
           [0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.]])
    >>> dc.dark_current_image.mean()
       0.0012602795833333822


    """
    __name__="DarkCurrent"

    ### Unique identifier for the full reconstruction process.
    __sequence_id__ = 40

    def __init__(self):
        super().__init__()

        ### EXECUTION MODE: full-image or cropped-image
        self.mode = 'cropped-image'
        self.exp_mode = 'exposure' #or exposure-per-row
        self.rng = np.random.default_rng(seed=self.__seed__)

        # lambda of the poisson distribution in units of electrons/pixel/hour
        self.darkcurrent=float(0.001)*u.e/u.pixel/u.day
        self.exp_time=float(8.0/24.)*u.day
        self.row_readtime=float(86400/86400)*u.day

        ### DEFINE parameters for R.O.I.
        self.img_size=int(300)
        self.min_enlarge=int(30)


        self.__units__={'darkcurrent':u.e/u.pixel/u.day,'exp_time':u.day,'img_size':u.pixel, 'row_readtime':u.day,'min_enlarge':u.pixel,"mode":1,"exp_mode":1}
        self.unit = "e/pixel"

    def get_overscan_regions(self):
        # when full image is created (simulate noise with also extra-regions)
        self.n_rows_overscan = u.n_rows_overscan
        self.n_rows_prescan  = u.n_rows_prescan
        self.n_cols_overscan = u.n_cols_overscan
        self.n_cols_prescan  = u.n_cols_prescan


    @property
    def dark_current_image(self):
        return self.__dark_current_image
    @dark_current_image.setter
    def dark_current_image(self,val):
        lam = val[0]*val[1]
        if self.recreate_image or self.full_image_counter == 0:
            self.ccd_shape = list(u.ccd_shape)
            self.get_overscan_regions()
            self.ccd_shape[0] += self.n_rows_overscan+self.n_rows_prescan
            self.ccd_shape[1] += self.n_cols_overscan+self.n_cols_prescan
        #insert here a lambda which is row dependent. Lambda components have increasing values
            if self.exp_mode == "exposure-per-row":
                        lam = lam + val[0] * self.row_readtime * np.arange(self.ccd_shape[0])
        #create image which has reverse axis: CCD X is rows, and CCD Y is columns. The image is transposed later to have correct shape.
            self.__dark_current_image = self.rng.poisson(lam,size =(self.ccd_shape[1],self.ccd_shape[0]))*u.e2eV
            self.__dark_current_image = np.transpose(self.__dark_current_image)
            #self.__dark_current_image = self.rng.poisson(lam,self.ccd_shape)*u.e2eV
            self.full_image_counter += 1
            self.recreate_image = False
            print(" -- Image dark current shape: ", self.__dark_current_image.shape)
            print("    Generate Dark Current Image: mean={} eV, std={} eV".format(
                round(self.__dark_current_image.mean(),3),round(self.__dark_current_image.std(),3)))
            print(" -- Image dark current: ", self.__dark_current_image)

    def execute_process(self):
        """Process implementation is done for two different running modes:

            **full-image**

                see :obj:`DarkCurrent.__execute_process_full_image__`

            **cropped-image**

                see :obj:`DarkCurrent.__execute_process_cropped_image__`
        """
        pass

    def __execute_process_cropped_image__(self,hits):
        """Add simulated Dark current image into the attribure **noise** of hits.

        Running on mode **cropped-image**, i.e. one smaller image will be re-generated
        for each event.

        This run mode is not appropiate for geant4 simulations under the mode
        'full decay', in this case use **full-image** mode.

        Parameters
        ----------

            hits : :obj:`pysimdamicm.io.data_format.G4HitCollection`
        """
        hits.mode = self.mode
        if not hasattr(hits,'roi'):
            hits.get_roi_fixed_length(self.img_size,self.min_enlarge)

        ### generating poisson numbers to fill the DC image
        #E_pixel_dc_noise=self.rng.poisson(self.darkcurrent * self.exp_time,hits.roi['size'])*u.e2eV

        #insert here a lambda which is row dependent. Lambda components have increasing values
        min_crop = min(hits.y_pixel)-hits.roi['shift']['y'][0] #finding the bottom,left corner of cropped image
        if self.exp_mode == "exposure-per-row":
                lam = self.darkcurrent * self.exp_time + self.darkcurrent * self.row_readtime * np.arange(min_crop,min_crop + hits.roi['size'][0])
        if self.exp_mode == "exposure":
                lam = self.darkcurrent * self.exp_time
        E_pixel_dc_noise=self.rng.poisson(lam,size= (hits.roi['size'][1],hits.roi['size'][0]))*u.e2eV
        E_pixel_dc_noise = np.transpose(E_pixel_dc_noise)

        print(" -- Image dark current shape: ", E_pixel_dc_noise.shape)
        print(" -- Image dark current: ", E_pixel_dc_noise)

        if not hasattr(hits,'noise'):
            hits.noise = []
        hits.noise.append(E_pixel_dc_noise)

    def __execute_process_full_image__(self,hits):
        """Execute process for mode 'full-image'

        In this case, the dark current image will be generated once and
        used in all processed events.

        Parameters
        ----------

            hits : :obj:`pysimdamicm.io.data_format.G4HitCollection`

        """
        hits.mode = self.mode
        if not hasattr(self,'dark_current_image'):
            self.dark_current_image = (self.darkcurrent,self.exp_time)


        ### define the ROI needed to account for the total energy per pixel (hits.get_total_signal)
        ##  that attribute ROI (region of interest) is necessary when the noise image is smaller than
        ##  the total CCD region, and a translation to the smallest size must be done to account for
        ## the total charge per pixel
        ## adding `-1`: arrays index starts at 0, and  6000 pixels represents the pixel 5999
        hits.roi = {'shift': {
                    'x': (min(hits.x_pixel)-1,0),
                    'y': (min(hits.y_pixel)-1,0)
                    },
                    'size': self.ccd_shape
                    }

        ### the detector has more than one type of noise: recorded at noise attribute
        if not hasattr(hits,'noise'):
            hits.noise = []
        hits.noise.append(self.dark_current_image)
        ### only when fits file is recorded
        hits.darkcurrent=self.darkcurrent
        hits.exp_time   = self.exp_time



###############################################################################################
#####       NOISE MODEL
###############################################################################################
class ElectronicNoise(DigitizeProcess):
    """Electronic Noise process.


    Electronic noise is assumed to follow a gaussian distribution.


    **List of Attributes:**

    Attributes
    ----------

        pedestal

            mean number of electrons per pixel

        sigma

            standar desviation for the gaussian distribution

        nSamples

            for skipper CCDs the number of samples that one pixel have been read; in this case, the sigma of the Gaussian becomes

        .. math::

           \sigma = \\frac{\sigma}{nSamples}

        img_size

            only for running mode **cropped-image**
            Number of pixels in the x-axis to create the electronic noise image surrounding the hits collections of the event (squared image)

        min_enlarge

            minimum number of pixels to add when the image size is smaller than the cluster elongation.
            In this cases, the size of dark  current image will be <cluster_size> + <min_enlarge>

        mode

            Name of the running mode to generate the dark current image. Possible values:

                * 'full-image'

                    all events will share the same dark current image with the size of the CCD
                    (in this case img_size is ignored)

                * 'cropped-image'

                    for each hit collection, a new DC image will be created, in this case the
                    DC image should be smaller than the real CCD size, and img_size will be used

    """

    __name__="ElectronicNoise"
    ### Unique identifier for the full reconstruction process.
    __sequence_id__ = 50

    def __init__(self):
        """
        """
        super().__init__()
        ### EXECUTION MODE: full-image or cropped-image
        self.mode = 'cropped-image'

        self.nSamples = int(1)
        self.rng = np.random.default_rng(seed=self.__seed__)

        self.sigma   =float(0.25)*u.e
        self.pedestal=float(0.0)*u.e

        self.img_size=int(300)
        self.min_enlarge=int(30)

        self.units = "e/pixel"
        self.__units__={'pedestal':u.e,'sigma':u.e,'img_size':u.pixel,'min_enlarge':u.pixel,'mode':1}

    def get_overscan_regions(self):
        # when full image is created (simulate noise with also extra-regions)
        self.n_rows_overscan = u.n_rows_overscan
        self.n_rows_prescan  = u.n_rows_prescan
        self.n_cols_overscan = u.n_cols_overscan
        self.n_cols_prescan  = u.n_cols_prescan


    @property
    def electronic_noise_image(self):
        return self.__electronic_noise_image
    @electronic_noise_image.setter
    def electronic_noise_image(self,val):
        pedestal,sigma = val
        if self.recreate_image or self.full_image_counter == 0:
            self.ccd_shape = list(u.ccd_shape)
            self.get_overscan_regions()
            self.ccd_shape[0] += self.n_rows_overscan+self.n_rows_prescan
            self.ccd_shape[1] += self.n_cols_overscan+self.n_cols_prescan
            self.__electronic_noise_image =  self.rng.normal(pedestal,sigma/np.sqrt(self.nSamples),self.ccd_shape)*u.e2eV
            self.full_image_counter += 1
            self.recreate_image = False
            print(" -- Image electronic noise shape: ", self.__electronic_noise_image.shape)
            print("    Using: mu_0={} e-/pix, sigma_0={} e-/pix".format(pedestal,sigma))
            print("    Generate Electronic Noise Image: \n\t values for the data sample \n\t mean={} eV,std={} eV".format(
                self.__electronic_noise_image.mean(),self.__electronic_noise_image.std()))

    def execute_process(self,hits):
        """Process implementation is done for two different running modes:

            **full-image**

                see :obj:`ElectronicNoise.__execute_process_full_image__`

            **cropped-image**

                see :obj:`ElectronicNoise.__execute_process_cropped_image__`
        """
        pass

    def __execute_process_cropped_image__(self,hits):
        """Add simulated Electronic noise into the attribure **noise** of hits.

        Running on mode **cropped-image**, i.e. one smaller image will be re-generated
        for each event.

        This run mode is not appropiate for geant4 simulations under the mode
        'full decay', in this case use **full-image** mode.

        Parameters
        ----------

            hits : :obj:`pysimdamicm.io.data_format.G4HitCollection`
        """
        hits.mode = self.mode
        if not hasattr(hits,'roi'):
            hits.get_roi_fixed_length(self.img_size,self.min_enlarge)

        ### generating gaussian values as electronic noise
        ###     for skipper CCDs sigma --> sigma/sqrt(nSamples)
        E_pixel_readout_noise=self.rng.normal(self.pedestal,self.sigma/np.sqrt(self.nSamples),hits.roi['size'])*u.e2eV

        if not hasattr(hits,'noise'):
            hits.noise = []
        hits.noise.append(E_pixel_readout_noise)


    def __execute_process_full_image__(self,hits):
        """Execute process for mode 'full-image'

        In this case, the electronic noise image will be generated once and
        used in all processed events.

        Parameters
        ----------

            hits : :obj:`pysimdamicm.io.data_format.G4HitCollection`

        """

        hits.mode = self.mode
        if not hasattr(self,'electronic_noise_image'):
            self.electronic_noise_image = (self.pedestal,self.sigma)

        ### define the ROI needed to account for the total energy per pixel (hits.get_total_signal)
        ##  that attribute ROI (region of interest) is necessary when the noise image is smaller than
        ##  the total CCD region, and a translation to the smallest size must be done to account for
        ## the total charge per pixel
        ## adding `-1`: arrays index starts at 0, and  6000 pixels represents the pixel 5999
        hits.roi = {'shift': {
                    'x': (min(hits.x_pixel)-1,0),
                    'y': (min(hits.y_pixel)-1,0)
                    },
                    'size': self.ccd_shape
                    }

        ### the detector has more than one type of noise: recorded at noise attribute
        if not hasattr(hits,'noise'):
            hits.noise = []
        hits.noise.append(self.electronic_noise_image)

        ### only when fits file is recorded
        hits.pedestal=self.pedestal
        hits.sigma   = self.sigma


###############################################################################################
#####       PIXEL SATURATION
###############################################################################################
class PixelSaturation(DigitizeProcess):
    """
    Pixel saturation is when the incident light at a pixel causes one of the color channels of the
    sensor to respond at its maximum value.

    Saturation and blooming are related phenomena that occur in all CCD image sensors under conditions
    in which either the finite charge capacity of individual photodiodes, or the maximum charge transfer
    capacity of the CCD, is reached.

    Once saturation occurs at a charge collection site, accumulation of additional photo-generated
    charge results in overflow, or blooming, of the excess electrons into adjacent device
    structures.


    """

    __name__="PixelSaturation"
    ### Unique identifier for the full reconstruction process.
    __sequence_id__ = 55

    def __init__(self):
        """
        """
        super().__init__()

        self.saturation = 65535*u.ADC*u.ADC2eV
        self.__units__={'saturation':u.ADC*u.ADC2eV}

    def execute_process(self,hits):
        """Set the attribute `pix_saturation` of the singleton :obj:`pysimdamicm.io.G4Units.Units`

        This attribute is used when the total CCD image is computed, see method :obj:`pysimdamicm.io.data_format.get_total_signal()`

        Parameters
        ----------

            hits : :obj:`pysimdamicm.io.data_format.G4HitCollection`

        """
        hits.pix_saturation = self.saturation
        setattr(u,"pix_saturation",self.saturation)


###############################################################################################
#####       PIXEL SATURATION
###############################################################################################
class PasteClusterProcess(DigitizeProcess):
    """

    """

    __name__="PasteClusterProcess"
    __sequence_id__ = 60

    __distributions = ['linear','compton','uniform']

    def __init__(self):
        """
        """
        super().__init__()

        self.blank_outdir = None

        ### string/pattern to directory with blank images
        self.blank_dir = None
        self.extension = 0
        ### model to randomly generate the new position in rows
        self.row_cls_dist_fx = 'compton'
        ### number of clusters to paste per image
        self.clusters_per_blank = None
        # number of cluster per blank that should be pasted in a uniform way,
        # these clusters are clusters hitting the CCD during exposure time
        self.clusters_per_blank_texp = None

        self.in_ADU      = True
        self.calibration = np.nan
        self.sigma       = None

        # ADD parameter to control randomness on the selection of blank images
        self.random_image_start = True

        self.__units__ = {'blank_outdir':1,'blank_dir':1,'row_cls_dist_fx':1,'clusters_per_blank':1,'extension':1,'in_ADU':1,
                'calibration':1,'sigma':1, 'clusters_per_blank_texp':1, 'random_image_start':1}

    def execute_process(self,hits):
        """

        Parameters
        ----------

            hits : :obj:`pysimdamicm.io.data_format.G4HitCollection`

        """
        if not hasattr(self,'_blanks'):
            # Asserts
            if not self.row_cls_dist_fx in PasteClusterProcess.__distributions:
                raise NotImplementedError("The distribution {} is not implemented.".format())
            if self.clusters_per_blank <1:
                raise AttributeError("The number of cluster per image must be >0.")
            #fdir = os.path.dirname(self.blank_dir)
            #if not os.path.isdir(fdir):
            #    raise OSError("Not found directory for blank images {}".format(fdir))
            if not os.path.isdir(self.blank_outdir):
                raise OSError("Directory {} does not exist".format(self.blank_outdir))

            ### iterator object BlankImages
            self._blanks = BlankImages(self.blank_dir,self.clusters_per_blank,self.blank_outdir,self.extension,
                    self.in_ADU,self.row_cls_dist_fx,calibration=self.calibration,sigma=self.sigma,
                    Ncls_flat=self.clusters_per_blank_texp,random_image_start=self.random_image_start)

        _ = next(self._blanks)
        self._blanks.append_clusters(hits)

    def __del__(self,hits=None):
        """
        """
        # save current image before killing the object
        try:
            self._blanks.save_blank_image()
        except AttributeError:
            print("Nothing to delete, _blanks does not exists!")


########################################################################################
#
#
########################################################################################
class BlankImages(object):
    """
    """
    def __init__(self,filenames_pattern,required_clusters_per_blank,outdir,extension=0,
            in_ADU=True,row_dist_fx='compton',calibration=np.nan,sigma=np.nan,Ncls_flat=0,
            random_image_start=True):

        if in_ADU:
            # define calibration with e2eV and ADC2eV (both given by the user)
            self.cal = u.e2eV/u.ADC2eV
        else:
            self.cal = 1.0

        self.isgauss = False
        if sigma is not None:
            self.sig     = sigma
            self.isgauss = True

        if not np.isnan(calibration):
            self.isgauss = True
            self.cal = calibration

        self._charge_conversion_function  = lambda isgauss: (1/u.e2eV)*np.random.normal(self.cal,self.sig) if isgauss else self.cal

        # initialization of the conversion factor
        self._charge_conversion = self._charge_conversion_function(self.isgauss)
        # set distribution function for the rows
        self.row_dist_fx = row_dist_fx

        # output directory to store the blank images after appending clusters
        self.outdir = outdir

        # list of files in the given directory following the pattern name
        #   randomly shuffle the list of files to be sued as blanks
        self._filelist = sorted(glob(filenames_pattern))

        self._len  = len(self._filelist)
        if self._len==0:
            msm = "There is no blank images at {}".format(filenames_pattern)
            raise RuntimeError("\x1b[31m {} \x1b[m".format(msm))

        # number of clusters pasted into the blank
        self._nmax_clusters = required_clusters_per_blank
        # initialization of counter and current index (pointing to an image from the list)
        self._appended_clusters = 0
        if random_image_start:
            random.shuffle(self._filelist)
            self._current_index = np.random.randint(self._len)
        else:
            self._current_index = 0

        # number of clusters pasted into the blank that should follow a flat distribution
        self.Ncls_flat = Ncls_flat

        # read the blank data, and keep the extension (needed to read the header before storing final image)
        self._extension = extension
        self._blank = fits.getdata(self._filelist[self._current_index % self._len],self._extension)#.astype(int)
        # monte carlo truee image, any cluster pasted on the _blank images, will be also pasted in
        # this "zero-charged" image, being the monte carlo true simulation
        # this will also be recorded in the output fits file image
        self._MCT      = np.zeros_like(self._blank).astype(float)
        self._MCT_ID   = np.zeros_like(self._blank).astype(int)
        self._MCT_EMCT = np.zeros_like(self._blank).astype(float)
        self._MCT_ERAW = np.zeros_like(self._blank).astype(float)

    def __next__(self):
        """
        """
        if self._appended_clusters == self._nmax_clusters:
            # save image before change the iter
            self.save_blank_image()
            # increase iter to the next image
            self._current_index += 1
            # initialize the clusters appended in the image
            self._appended_clusters = 0
            self._blank = fits.getdata(self._filelist[self._current_index % self._len],self._extension) #.astype(int)
            # initialize also the monte carlo true image
            self._MCT   = np.zeros_like(self._blank).astype(float)
            self._MCT_ID= np.zeros_like(self._blank).astype(int)
            self._MCT_EMCT = np.zeros_like(self._blank).astype(float)
            self._MCT_ERAW = np.zeros_like(self._blank).astype(float)
            # restart calibration constant
            self._charge_conversion = self._charge_conversion_function(self.isgauss)
        return self._blank

    def save_blank_image(self,overwrite=True):
        """
        """
        image_file_name = self._filelist[self._current_index % self._len]

        # get header from blank image
        header = fits.getheader(image_file_name,self._extension)
        # add some parameters
        header['NMAX'] = (self._nmax_clusters,"Maximum number of evets")
        header['NEVTS'] = (self._appended_clusters,"Number of appended events")
        header['eV2ADU'] = (self._charge_conversion*u.e2eV,"Conversion used: ADU/e-")
        # create the HDU list: using header from blank
        hdul = fits.HDUList([fits.PrimaryHDU(data=self._blank, header=header)])
        hdul.append(fits.ImageHDU(data=self._MCT,name="MonteCarloTruth"))
        # append another extension with the cluster id
        hdul.append(fits.ImageHDU(data=self._MCT_ID,name="MCT_ID"))
        # append another extension with the energy/charge from the blank image before cluster has been injected
        hdul.append(fits.ImageHDU(data=self._MCT_ERAW,name="MCT_ERAW"))
        # append another extension with the total energy of the cluster
        hdul.append(fits.ImageHDU(data=self._MCT_EMCT,name="MCT_EMCT"))

        # save image
        fitsname = "{}/psimulCCDimg_{}".format(self.outdir,
                os.path.basename(image_file_name).replace(".fits","_Ncls{}".format(int(self._appended_clusters))))
        _rhash = hash(hash(time()) + abs(hash(self)))
        fitsname += "_{}.fits".format(_rhash)

        hdul.writeto(fitsname,overwrite=overwrite)


    def append_clusters(self,hits):
        """
        """
        killed,(new_rows,new_cols,Edep) = self._random_position(hits)
        if killed:
            return
        ## round energy to the nearest integer (ADU units is an integer variable)
        ############### APPLY QUADARTIC CALIBRATION ---- COMPTON
        if self.row_dist_fx in ["compton"]:
            # FIXME --- this is hardcoded for compton
            quadratic_cal = np.polynomial.polynomial.Polynomial([0.4088,5.177,7.85e-6])
            Edep_e = Edep/u.e2eV
            Edep_ADCs = np.round(quadratic_cal(Edep_e)).astype(int)
        else:
            #print(np.sum(Edep))
            #print(self._charge_conversion)
            Edep_ADCs = (Edep*self._charge_conversion).astype(float)
            #print(np.sum(Edep_ADCs))
            #input("press enter ...")

        # before appending the clusters, sum up the energy already in the image
        self._MCT_ERAW[new_rows,new_cols] += self._blank[new_rows,new_cols]

        # Append the cluster into the current blank
        self._blank[new_rows,new_cols] += Edep_ADCs
        # Update the number of clusters appendend into the current blank
        self._appended_clusters += 1

        # add event pixels also to the monte carlo true image (in units of eV)
        self._MCT[new_rows,new_cols] += Edep
        self._MCT_ID[new_rows,new_cols]   = int(self._appended_clusters)
        self._MCT_EMCT[new_rows,new_cols] = float(np.sum(Edep))
        #print(np.sum(Edep), self._MCT_EMCT[new_rows,new_cols])
        #print(self._blank[new_rows,new_cols] )

        # Apply saturation if is invoked, pix_saturation is in units of eV
        if hasattr(hits,"pix_saturation") and getattr(hits,"pix_saturation")>0:
            max_charge_ADU = hits.pix_saturation/u.e2eV * self._charge_conversion
            self._blank[self._blank>max_charge_ADU] = max_charge_ADU
            hits.Npix_sat = sum(self._blank>max_charge_ADU)

    def _random_position(self,hits,axis=0):
        """

        Parameters
        ----------

            axis : int
                Axis (0:rows, 1:columns) in which the displacement should be applied

        """
        # flag to control if we end-up with pixels in the new position
        killed = False

        ### to paste the cluster inside the active region
        prescan  = [u.n_rows_prescan,u.n_cols_prescan]
        overscan = [u.n_rows_overscan,u.n_cols_overscan]
#        print(prescan,overscan)

        # real active region size of the blank image
        active_region = [ self._blank.shape[0]-prescan[0]-overscan[0], self._blank.shape[1]-prescan[1]-overscan[1] ]
#        print(active_region)

        displacement = [0,0]
        # simulated positions
        positions = [hits.y_pixel,hits.x_pixel]
        # event elongation
        Dcol = int(max(hits.x_pixel) - min(hits.x_pixel))
        Drow = int(max(hits.y_pixel) - min(hits.y_pixel))
        elongation = [Drow,Dcol]

        if self.row_dist_fx in ['linear']:
            row_max = active_region[axis]
            nrow = np.sqrt(np.random.uniform(0.,1.0)) * row_max

        elif self.row_dist_fx in ['uniform']:
            row_max = active_region[axis]
            nrow = np.random.uniform(0.,row_max)

        else:
            # number of rows in the active region, the only place where we should paste clusters
            row_max = active_region[axis]

            # clusters during exposure time should follow a flat distriution. The number of cluster
            # per image during exposure time are given by: self.Ncls_flat
            if self.Ncls_flat>0 and self._appended_clusters<self.Ncls_flat:
                # clusters that occurs during exposure time
                nrow = np.random.uniform(0,row_max)
            else:
                # clusters during exposure time
                # ratio between first row and last row: how much frequent should be 1(last row) from 0 (first row)
                # from the normalized data this should be
                m = float(6.82)
                ### starts the random algorithm
                s_rm = 1-(1/m)**2
                val = (1-random.uniform(0,s_rm))**.5
                nrow = row_max*(m*val-1)/(m-1)

        # actualize the displacement
        displacement[axis] = int(nrow)
        # move simulated positions into the blank image (that maybe smaller than the simulated one)
        positions[axis] = (positions[axis] - positions[axis].max()) + prescan[axis] +  displacement[axis]


        # For axis without displacement:
        # r = (self._blank.shape[1-axis] - prescan[1-axis] - overscan[1-axis]) / float(u.ccd_shape[1-axis])
        # displacement[1-axis] = int(positions[1-axis].min()*r) - elongation[1-axis]
        Ldim = self._blank.shape[1-axis] - prescan[1-axis] - overscan[1-axis] - elongation[1-axis]
        Lstart = prescan[1-axis]
        displacement[1-axis] = random.randint(Lstart,Ldim)
        positions[1-axis] = positions[1-axis]-positions[1].min() + displacement[1-axis]

        # this translation can lead to negative pixels: clusters are killed partially
        try:
            new_rows,new_cols,E = zip(*list(filter(lambda p: p if (p[0]>=0 and p[1]>=0) else None, zip(positions[0],positions[1],hits.Edep_pixel))))
        except ValueError:
            hits.killed = True
            new_rows,new_cols,E = None,None,None
            ## In some rare cases, all pixels will be placed in such a way that none of them will fall inside the area
            killed=True
        # remove pixels out from blank region: cluster not fully appended
        try:
            new_rows,new_cols,E = zip(*list(filter(lambda p: p if (p[0]<self._blank.shape[0] and p[1]<self._blank.shape[1]) else None,
                zip(new_rows,new_cols,E))))
        except ValueError:
            ## In some rare cases, all pixels will be placed in such a way that none of them will fall inside the area
            new_rows,new_cols,E = None,None,None
            hits.killed = True
            killed=True


        # list objects: convert into arrays
        E = np.array(E)
        new_rows = np.array(new_rows)
        new_cols = np.array(new_cols)

        #raise
        # Add attributes to hits to dump into the ouptut ROOT file
        setattr(hits,'rnd_rows',new_rows)
        setattr(hits,'rnd_cols',new_cols)
        setattr(hits,'rnd_Edep',E)

        return killed,(new_rows,new_cols,E)

