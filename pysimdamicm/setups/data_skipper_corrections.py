""":mod:`pysimdamicm.skiat.data_skipper_correctons`
    
    skipperimg is a collection of algorithms for CCD image processing
    Module of the main methods for the skiat of a CCD image      
    
.. moduleauthor:: Nuria Castello-Mor
"""
import numpy as np
##############################################################################################################
#   
#   CCCDrone    (Leach --> fits file) has some bug recording the data from the leach:
#                   the first two columns are moved to the end, and the first rows of this block of
#                   two columns are moved to the last row !!! 
#
#   CorrectLeachBugProcess :: function to correct fro this LeachBug directly to data
#
##############################################################################################################
def InvertPolarity(image):
    """Invert the image

    If the data was taken without the inverter the saturation 
    takes place at the lowest values of ADUs, while any signal from
    ionizing particles will have a negative impluse signal, having
    lower values of ADUs than the non-charged pixel, i.e. the signal 
    (ADU values) is inversely proportional to the pixel charge. 

    InvertPolarity will just invert the proportionality of the signal 
    with the ionized charge by doing

    .. math::

       q_{i,j}^{'} = max(q_{i,j}) - q_{i,j}

    
    """
    print("Correction INFO. The pixel charge values will be inverted to be\n",
          "\t proportional to the ionization signal: max(qi) - qi\n")

    return image.max() - image


##############################################################################################################
#   
#   CCCDrone    (Leach --> fits file) has some bug recording the data from the leach:
#                   the first two columns are moved to the end, and the first rows of this block of
#                   two columns are moved to the last row !!! 
#
#   CorrectLeachBugProcess :: function to correct fro this LeachBug directly to data
#
##############################################################################################################
def CorrectLeachBugProcess(image_data,ampl):
    """

    There is a bug on the CCDDrone (UChicago setup) during data recording. Instead of recording the
    columns in a sorted way, the framework store the last two columns at the begining of the image
    and moving one row down (in this two bad-recorded columns). Each amplifier works in the same
    wrong way. 

    **Image representation**

    The following is a graphical representation of the misprinted data (using two amplifiers to
    read, i.e. AMPL='UL')


                 ORIGINAL DATA                           NEW RE-ORDERED DATA 
    
        index   012 .....       -2-1                    012 ...         -2-1    ncoltot
    
        0   ABxxxxxxxxxxxxxxxxxxGH                    xxxxxxxxxxxCDIJxxxxxxxxxxx
        1   CDxxxxxxxxxxxxxxxxxxIJ            -->     xxxxxxxxxxxCDIJxxxxxxxxxxx
            CDxxxxxxxxxxxxxxxxxxIJ            -->     xxxxxxxxxxxCDIJxxxxxxxxxxx
            CDxxxxxxxxxxxxxxxxxxIJ            -->     xxxxxxxxxxxCDIJxxxxxxxxxxx
            CDxxxxxxxxxxxxxxxxxxIJ                    xxxxxxxxxxxCDIJxxxxxxxxxxx
                .        .                               .       .
                .        .                               .       .
            CDxxxxxxxxxxxxxxxxxxIJ                    xxxxxxxxxxxCDIJxxxxxxxxxxx
       -2   CDxxxxxxxxxxxxxxxxxxIJ                    xxxxxxxxxxxEFKLxxxxxxxxxxx
       -1   EFxxxxxxxxxxxxxxxxxxKL                    xxxxxxxxxxxABGLxxxxxxxxxxx
    nrows                                                        ||||

    """
        
    print("Correction INFO. Correcting data for Leach bug:\n",
            "\t the first two column moved to the end,\n",
            "\t move one-row down (only the two last columns).\n")

    nrows,nallcols = image_data.shape
    

    if ampl == 'UL':
        # two amplifiers have been used to read the full CCD
        ncoltot = int(nallcols/2)
    else:
        ncoltot = nallcols

    image = np.zeros_like(image_data)
    image[:,0:ncoltot-2] = image_data[:,2:ncoltot]
    image[:-1,ncoltot-2] = image_data[1:,0]
    image[:-1,ncoltot-1] = image_data[1:,1]
    image[-1,ncoltot-2]  = image_data[0,0]
    image[-1,ncoltot-1]  = image_data[0,1]

    if ampl == 'UL':
        ### second half of the image
        image[:,ncoltot+2:]  = image_data[:,ncoltot:-2]
        image[:-1,ncoltot]   = image_data[1:,-2]
        image[:-1,ncoltot+1] = image_data[1:,-1]
        image[-1,-2]         = image_data[0,-2]
        image[-1,-1]         = image_data[0,-1]

    return image
