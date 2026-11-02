.. _howtouse_analysis_fftnoise:

The process `FFTNoisePlot` utilizes the Fast Fourier Transform (FFT) to study the spectral noise of
the single skip images, as well as the noise in rows of the mean compressed image. The FFT is the most 
convenient tool for calculating spectral noise power distribution of discre-time signals.


For each pixel the FFT of the time-domain single skip measurements will be computed 

    .. math::
       
       \sum_{rows} \sum_{cols} \frac{|FFT( q_{i,j}^{k} - \langle q_{i,j}\rangle |  k=skip_{start}, ... , skip_{end})|}{N_{cols} \times N_{rows}} (1)


For each row of the compressed image (by applying the mean function to all single skip
measurements) the FFT is computed


    .. math::

       \sum_{rows} \frac{ |FFT(q^{'}_{i,j} - \langle q^{'}_{i} \rangle  | j=col_{start}, ..., col_{end} )| }{N_{cols}}   (2)


This process requires **CompressSkipperProcess**, so **CompressSkipperProcess** has to be included
to **sequence**.


**attributes of the process**

This process does not has any special attribute. The relevant here are the starting and end point of the skips, rows and columns 
(**id_skip_start**, **id_skip_stop**, **id_row_start**, **id_row_stop**, **id_col_start**, **id_col_stop**).


**how to run**

How to run with and without a configuration file:

   * using the default values for all realted attributes

   .. code-block:: bash

      $ panaSKImg -s CompressSkipperProcess,FFTNoisePlot data_image_name.fits --display

   * or setting through line commands

   .. code-block:: bash

      $ panaSKImg -s CompressSkipperProcess,FFTNoisePlot data_image_name.fits --skip-start 10 --row-start 5 --display

   * or by using the configuration json file for the parameter setting, and just set the list  of process by command line as follows

   .. code-block:: bash

      $ panaSKImg --json panaSKImg_configuration.json -s ChargeLossPlot,FFTNoisePlot data_image_name.fits --display


In all previous modes to run any common parameter (to select a set of skips, rows, or columns, ... ) can be also included.

**output**

Two output products will be displayed if --display is invoked:

    * The mean compressed image used for one of the FFT plots
    * The FFT magnitude over the single skips (equation 1)
    * The FFT magnitude over each row in the mean compressed image (equation 2)


**output example**

    .. figure:: image/FFTNoisePlot_output_0.png
       :scale: 75

       This image will be show up because CompressSkipperProcess  is also invoked as a requirement
       for the FFTNoisePlot process.

proper output of the process `FFTNoisePlot`:

    .. figure:: image/FFTNoisePlot_output_1.png

       FFT magnitude as a function of the requested time to read n skip measurments, and
       on the rigth, the FFT magnitude of the rows as a function of the requested time to read a
       full row.


