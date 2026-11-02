.. _howtouse_analysis_fitcalibration:

The process `FitCalibrationConstant` studies the linearity with single electron peaks. The calibration constant 
is computed by fitting each single electron peak to a single gaussian.

Internally, this process will first estimate the calibration constant, `k`, as the distance between the first and 
second peak (correspondding to 0 and 1 electron per pixel) 

    .. math::

       k = \mu_{1 e^{-}} - \mu_{0 e^{-}}

and use this first estimation to get the
signal in units of electrons 

    .. math::

       q^{'}_{i,j} = q_{i,j} / k

in which single electron peaks should be found at 0,1,2,... n values. Each peak will be fitted individually 
by using an spectral window centered to 1,2,...,n values with a width defined by :math:`\sigma MAD`.


This process requires outputs from processes **CompressSkipperProcess** and **PedestalSubtractionProcess**, 
which should be also included to **sequence**.

**attributes of the process**

Apart from the common parameters already listed in the previous section, the **user** has access to the following paramter:


    * **image**: name of the image data member to be used (by default is `pedestal_subtracted` (do not change 
      unless you are a developer). Note that this process run over the **pedestal_subtracted**
      image, which means such process should be booked.

    * **n-peaks**: number of peaks to fit starting from peak at 0 electrons.
      
    * **calibration**: starting point for the calibration constant `k`, if the fit between the two first peaks does not converge 
      (mostly to help the fitting process)
        
    * **n-sigma-win-fit**: number of sigmas to define the width of the spectral window to fit the
      single electron peaks, by default 7 (play with this parameter in case the fit does not
      converge).

**how to run**

How to run with and without a configuration file:

   * using the default values for all realted attributes

   .. code-block:: bash

      $ panaSKImg -s CompressSkipperProcess,PedestalSubtractionProcess,FitCalibrationConstant data_image_name.fits --display

   * or setting through line commands

   .. code-block:: bash

      $ panaSKImg -s CompressSkipperProcess,PedestalSubtractionProcess,FitCalibrationConstant data_image_name.fits --calibration 40 --n-peaks 20 --display --verbose --in-overscan

   * or by using the configuration json file for the parameter setting, and just set the list  of process by command line as follows

   .. code-block:: bash

      $ panaSKImg --json panaSKImg_configuration.json -s CompressSkipperProcess,PedestalSubtractionProcess,FitCalibrationConstant data_image_name.fits --display


In all previous modes to run any common parameter (to select a set of skips, rows, or columns, ... ) can be also included.

**output**

At least four figures will be displyed for this process:

    * A single plot with the single electron peaks where the first and second peak are mark. If they
      are not found correctly the user can play with parameters **calibration** and
      **n_sigma_win_fit** to tune properly this process.

    * A single plot with the gaussian fit for the first 10 single electron peaks. Note that each peak is fitted individually after normalization, so 
      the amplitudes between peaks will not be shown. 

    * A figure with four plots: 
        
            * reconstructed charge vs true charge with a linear fit

            * relative differences from fit vs true charge

            * residual fits vs true charge and,

            * sigma vs true charge

    
    * The pixel charge distribution in two range: from 0 to 30 electrons, and from 30 up to 200
      electrons.




**output example**

   
    .. figure:: image/FitCalibrationConstant_calibration_guess.png

       Example of the pixel charge distribution in units of electrons showing the detected first and
       second electron peak (i.e. at 0 and 1 electrons).

    .. figure:: image/FitCalibrationConstant_fit_peaks.png
       :scale: 60
       :align: center

       Example of the first fitted gaussians (all normalized to 1).
       
    .. figure:: image/FitCalibrationConstant_lineality.png
       :align: center

       Lineality plot for the calibration constant (:math:`\mu` from the gaussian fit to single
       electrons peaks) and some other quantities from the fit)

    .. figure:: image/FitCalibrationConstant_PCD_plot.png
       :scale: 50
       :align: center

       Pixel charge distribution for low and high electron range: 0-20 e- and 30-200 e-


