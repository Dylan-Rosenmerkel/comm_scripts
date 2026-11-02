.. _howtouse_analysis_fitdc:

The process `FitDarkCurrentProcess` is the process to fit the dark current. 


The distribution of pixel values in a CCD comes from the convolution of the pixel charge with the
pixel readout noise. The pixel charge is the sum of a Poisson-distributed leakage current
accumulated during the exposure and a ionized signal charge (from a ionizing
particle that interacts with the silicon target). The readout noise is parametrized from the pixel
value distribution of blanks and overscans, and found to be well-described by the convolution of a
Poisson with average :math:`\lambda` and a Gaussian of standard deviation :math:`\sigma_{pix}`. 

If the input image contains tracks, the best is to estimate the dark current only using the first
two single electron peaks where contribution from any ionizing particle is almost negligible. In
this case, the distribution of pixel values in a CCD is modeled by the function 


    .. math::

       \prod(p) = \mathcal{N} \sum \limits_{n=0}^{K\text{-peaks}} \left[\sum\limits_{j=0}^{n}\mathcal{P}\left(n-j|\frac{\lambda}{k}\right)\circledast\mathcal{G}\left(p|\frac{\mu_{0}}{k}, \frac{\sigma}{k} \right) \right]
       \quad\quad\quad (Eq.~1)

where


    * n corresponds to the number of electrons of the corresponding single electron peak

    * k is the relative calibration constant (should be 1 if the used calibration constant is the
      optimal one, if not this parameter will give us the re-fitted calibration constant), in units
      of ADU/:math:`e^{-}`

    * :math:`\lambda` is the Poisson parameter, i.e. the dark current in units of ADU/pix/img

    * :math:`\mu_{0}` is the center of the zero-electron peak (if :math:`\lambda<1` (otherwise the
      center of the highest peak). This is the offset accounting for pedestal subtractoin. If the input image 
      is the pedestal-subtracted image then this should be around 0.

    * :math:`\sigma_{pix}` is the electronic noise noise, i.e. the standard deviation of the gaussian 
      to be the same for all fitted peaks (in units of ADU/pix)


The fit is done with ROOT through python (the plot will be displayed with TCanvas instead of
matplotlib).


**attributes of the process**

Apart from the common parameters already listed in the previous section, the **user** has access to the following paramter:


    * **image**: name of the image data member to be used (by default is `pedestal_subtracted` (do not change 
      unless you are a developer)
    * **method**: method to estimate the dark current (up to now, only with the ROOT package)
     
            * **root** 
            
            * **python** (probably coming soon)

    * **n-peaks**: number of single electron peaks to use to estimate the dark current (at least 2,
      :math:`n` in Eq. 1)

    * **n-sigma-fit**: number of sigmas to define the width of the spectral window to fit the single
      electron peak with charge 0, to estimate the starting point for all paramters in the fit
      (**mu-gauss**, **sigma-gauss**, and **lambda-poisson** which can also be given by the user as
      command lines)
        
    * **mu-gauss**: initial value for the position of the single electron peak of charge 0 electrons

    * **sigma-gauss**: initial value for the electronic noise (assumed to be the same for all peaks) 

    * **lambda-poisson**: initial value for the dark current in units of :math:`e^{-}/` pix 

    * **do_calibration**: set if the image used to fit the dark current is not calibrated (i.e. if the
      process FitCalibrationConstant is not booked). In this case the parameter **calibration** must
      be informed to properly calibrate the signal

    * **calibration**: calibration constant

    * **fit-options**: option for the ROOT TGraph::Fit, by default "QSL" (Minuit is the default
      library used as minimizer)

A set of parameters to define the limits for the different fitting parameters (only via JSON file)   
    * **mu_min**, **mu_max**: minimmum and maximum allowed values for the **mu-gauss** parameter
    * **sigma_min**, **sigma_max**: minimmum and maximum allowed values for the **sigma-gauss**
      parameter
    * **dc_min**, **dc_max**: minimum and maximum allowed values for the **lambda-poisson**
      parameter
    * **gain_min**, **gain_max**: minimum and maximum allowed values for the **calibration**
      parameter

**how to run**

How to run with and without a configuration file:

   * using the default values for all realted attributes

   .. code-block:: bash

      $ panaSKImg -s CompressSkipperProcess,PedestalSubtractionProcess,FitDarkCurrentProcess data_image_name.fits --display

   * or setting through line commands

   .. code-block:: bash

      $ panaSKImg -s CompressSkipperProcess,PedestalSubtractionProcess,FitDarkCurrentProcess --do-calibration --calibration 41.8 --display --verbose

   * or by using the configuration json file for the parameter setting, and just set the list  of process by command line as follows

   .. code-block:: bash

      $ panaSKImg --json panaSKImg_configuration.json -s CompressSkipperProcess,PedestalSubtractionProcess,FitDarkCurrentProcess data_image_name.fits --display


In all previous modes to run any common parameter (to select a set of skips, rows, or columns, ... ) can be also included.

**output**

A single plot will be displayed for this process:

    * A TCanvas object with the dark current fit (using Eq. 1)


**output example**

   
    .. figure:: image/FitDarkCurrentProcess_output_fit.png

       Example of the dark current fit to an image with single electron resolution (with 500 skips)

