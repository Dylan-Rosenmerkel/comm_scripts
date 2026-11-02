import numpy as np
from pysimdamicm.activity import metadata

def get_activity(isotope, Texp=61.0, Tcool=180, Trun=360):
    """
    isotope ::  name of the isotope as follows, i.e. 60Co will be 60a27z

    Texp    ::  equivalent activation time in units of days
    Tcool   ::  the cooldown time for the simulated volume, in days
    Trun    ::  time since the detector is taking data, in days

    """

    umBqkg2decaysPerKgPerday =  0.000001*60*60*24
    ### saturation
    S_umBq_kg =  metadata.isotope_half_life[isotope]["sat"]
    S_decays_perKg_perday = umBqkg2decaysPerKgPerday * S_umBq_kg
    ### half life time
    t_half_life = metadata.isotope_half_life[isotope]["t_half_life"]

    lambda_iso = np.log(2.0)/t_half_life

    # due to exposure time
    act_Texp  = 1.0 - np.exp( -lambda_iso * Texp )
    # due to cooldown
    act_Tcool = np.exp( -lambda_iso * Tcool )

    activity = S_decays_perKg_perday * act_Texp * act_Tcool

    if Trun>0.0:
        # due to running time
        act_Trun  = 1.0/(lambda_iso*Trun) * (1.0 - np.exp(-lambda_iso * Trun) )
        activity = activity * act_Trun


    return activity

