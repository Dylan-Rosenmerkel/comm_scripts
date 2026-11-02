import numpy as np

def Singleton(cls):
    def wrapper():
        if not hasattr(Singleton,'__instance__'):
            Singleton.__instance__=cls()
        return Singleton.__instance__
    return wrapper

@Singleton
class Units(object):
    """Define the system units by default and its conversion factors


    Attributes
    ----------
    mm : float
        milimeter, default unit for coordenates

    eV : float
        electronvolt, default unit for energy

    s : float
        seconds, default unit for time
        
    um,cm,m : float
        convertion factors wrt the default unit `mm`

    keV,MeV,GeV : float
        convertion factors wrt the default unit `eV`
    
    min,hour,day : float
        convertion factors wrt the default unit `s`

    e2eV : float
        Mean energy (in units of eV) to create an electron-hole pair in the
        silicon detector at the operational temperartre, default is 3.77eV
        (for an operating temperature of 140K).
        
    ADC2eV : float
        Value to convert ADC (analog-to-digital converter) units to keV,
        default is 2.6E-4.

    """

    def __init__(self):

        ### default units as the ones from geant4 simulations
        self.mm = float(1.0)
        self.eV = float(1.0)
        self.s  = float(1.0)
        self.pixel = int(1)
        self.e = int(1)
        self.eVee = float(1.0)

        ### ADCu
        self.ADC = 1
        
        #### Conversion factors 
        self.e2ADC = 14.5*self.ADC
        # at T~126K (compton setup)
        self.e2eV = 3.74*self.eV
        self.eV2ADC = 1/self.e2eV * self.e2ADC
        self.ADC2eV = 1/self.e2ADC * self.e2eV
        #### added for comissioning: look like U-10 while for L is 11
        self.ADC2e  = 10.5

        #### CCD dimensions
        ##  (rows,cols) = (y-axis, x-axis)
        self.ccd_shape=(int(4000),int(6000))*self.pixel
        self.ccd_pixel_size_x=float(15./1000.)*self.mm
        self.ccd_pixel_size_y=float(15.0/1000.)*self.mm
        self.ccd_thickness=0.675*self.mm
        ### overscan region
        self.n_rows_overscan = 0
        self.n_rows_prescan = 0
        self.n_cols_overscan = 0
        self.n_cols_prescan = 0

        self.pix2mm=self.ccd_pixel_size_x
        self.mm2pix=1/self.ccd_pixel_size_x

        ### define conversions for all the others
        ### long
        self.um = 1e-3 * self.mm
        self.cm = 10 * self.mm
        self.m  = 1e3 * self.mm
        ### energy
        self.keV = 1e3 *self.eV
        self.MeV = 1e6 *self.eV
        self.GeV = 1e9 *self.eV
        ### time
        self.us = 1e-6*self.s
        self.ms = 1/1000*self.s
        self.minute = 60.0*self.s
        self.hour = 60.0*self.minute
        self.day = 24.0*self.hour

        ### pixel saturation
        self.pix_saturation = 0.0*self.ADC*self.ADC2eV

        ### add sensitive detector name
        self.detector_name = "CCDSensor_PV"

        ### non related with units
        self.n_figure = 1
  
        ### Added to prevent HR image it's not on the images used for dqm
        self.me_dc_gain   = (np.nan,np.nan)
        self.me_dc_lambda = (np.nan,np.nan)
        self.me_dc_sigma  = (np.nan,np.nan)
        self.me_dc_mu0    = (np.nan,np.nan)
        
        ### image exposure time to group events happening within a time interval
        #       used mostly when running simulations in the full decay mode
        #       for coincidences anlysis
        self.img_exp_time = 1e9*self.day
        
        ### dictionary to add parameters to be added in the fits file header of the file
        #       As some parameters can come from other files (like an image with HR, from which we
        #       fit the dark current, gain and so on, but that we want to save those values to the
        #       less resolution fits file)
        self._tohdr = {}
        
        self._gain = {}
        
        # The values of z within the CCD from Geant4 are inverted: 0 (back) while max (front of the ccd)
        # values shoudl be inverted
        self.invert_ccd_z_values = True

    def set_gain(self,amp,gain):
        self._gain[amp] = gain


