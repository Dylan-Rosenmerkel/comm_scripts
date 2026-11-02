.. _howtouse_analysis_pedestalsubtraction:

The process `PedestalSubtractionProcess` also known as `equalization` is the process of 
substracting the pedestal or baseline from an image. The baseline or pedestal is the ADU values for
a pixel with zero exposure time. 


The CCD images contain a two-dimensional stacked history (projected on the x-y plane) of all
ionization produced throughout an exposure, where each image pixel value is proportional to the
collected number of charge carriers. The pixel values are in ADU (analog-to-digital units), a 16 bit
number: normally between 0 to 65535, but it depends on the current version of the readout
electronics (see invert polarity for instance). The ADU value is related to the number of electrons
counted at the sense node by way to the `gain k`  (ADU per electron counted during readout), a
constant based on operating condition and is derived from calibration (also known as `calibration
constant`. This constant is then derived from the gain taking into account the energy required to
produce an electron-hole pair.


Once the charge is in the serial register, reading the serial register past the length of a row will
mean one gets an image that is larger in columns (`x-axis`) than the size of the sensor (active
region). These extra pixels will by definition contain almost no charge sine they will have only
been exposed for the amount of time it took to read the entirety of a row. These 0 exposure pixels
make up the `x-overscan` and provide a snapshot of the baseline or pedestal of an image, i.e. the
ADU value associated with no charge.

Despite one should expect similar values of pedestal per row and/or per column in a good data taken
condition, It is not always like this. **`PedestalSubtractionProcess` has the ability to calculate the
pedestal per column, per row or in the full region**. It is also possible to use the overscan region 
only, or the full sensitive region
(after masking those pixels with pixel charge below a certain value). This is useful when the data
taken has long readout times, or show noise pattern along rows and/or columns.



**attributes of the process**

Apart from the common parameters already listed in the previous section, the **user** has access to the following paramter:


    * **image**: name of the image data member to be used (by default is `mean_compressed` (do not change 
      unless you are a developer)
    * **method**: method to estimate the pedestal: 
      
            * gaussian fit (**gauss_fit**): to fit a gaussian to the zero-charged pixel value, and
              if statistics is not enough the mean value will be used 
            
            * simple mean (**median**)

    * **in-overscan**: by default only the overscan region will be used to estimate the pedestal. Nevertheless, the 
      full image can also be used, for that just unset this parameter. In this case, some pixels can be masked
        
        .. math::
           
           q_{i,j}  - median(q_{i,j})  > (\sigma \times MAD)

      where :math:`\sigma` is the parameter **n_sigma_to_mask**.

    * **axis**: parameter to set in which axis the pedestal should be computed: 
        
            * **row**: to compute the pedestal per row, and subtract it row by row

            * **col**: to compute pedestal per column, and subtract column by column (in this case the full image should be used)

            * **both**: to compute pedestal per row, and then per column over the row-pedestal subtracted image

            * **none**: when the full region is used to estimate a unique pedestal (valid for images that have homogeneous pedestal subtraction) 

    * **n-sigma-win-fit**: number of sigmas to define the width of the spectral window to fit the
      zero pexel charge to a gaussian, by default 7 (only for method **gauss_fit**).

    * **n-sigma-to-mask**: number of sigmas to define the maximum pixel charge to take into account
      when the full image is used to estimate the pedestal (only for method **median**).

    * **histequ**: set if the images should be equalized. This is mostly to improve contrast between
      pixels of low charge (similar to what ds9 does).
    
    * **show_fit**: set for extra plots related to the gaussian fit process

**how to run**

How to run with and without a configuration file:

   * using the default values for all realted attributes

   .. code-block:: bash

      $ panaSKImg -s CompressSkipperProcess,PedestalSubtractionProcess data_image_name.fits --display

   * or setting through line commands

   .. code-block:: bash

      $ panaSKImg -s CompressSkipperProcess,PedestalSubtractionProcess data_image_name.fits --axis none --method median --display

   * or by using the configuration json file for the parameter setting, and just set the list  of process by command line as follows

   .. code-block:: bash

      $ panaSKImg --json panaSKImg_configuration.json -s CompressSkipperProcess,PedestalSubtractionProcess data_image_name.fits --display


In all previous modes to run any common parameter (to select a set of skips, rows, or columns, ... ) can be also included.

**output**

At least two figure will be displyed for this process:

    * A single plot with the compressed image 

    * A figure with the raw image (the compressed one) as well as the pedestal subtracted image. If the axis is row, col or both, it will be also displayed the pedestal as a funciton of the given axis. 

    * If **show-fit** is set, the gaussian fit to the zero charge pixel will be also displayed.


**output example**

   
    .. figure:: image/PedestalSubtractionProcess_output_fit_per_rows.png

       Example of compressed image (raw data), image after pedestal subtraction and on the right
       side, the pedestal for each row (for this example axis=row, and method=gauss_fit over the
       overscan region).

    .. figure:: image/PedestalSubtractionProcess_output_fit_per_rows_0.png
       :scale: 60
       :align: center

       Example of all fitted gaussians, in this example one per row (if axis=full, only one gaussian will be shown)
       
    .. figure:: image/PedestalSubtractionProcess_output_fit_per_rows_1.png
       :scale: 50
       :align: center

       Example of the fitted :math:`\mu` values (i.e. the pedestal) as a function of the row axis.

    .. figure:: image/PedestalSubtractionProcess_output_fit_per_rows_2.png
       :scale: 50
       :align: center

       Example of the fitted :math:`\sigma` values as a function of the row axis.


    .. figure:: image/PedestalSubtractionProcess_output_masked.png
       :scale: 70
       :align: center

       Example of the masked image used to estimate the pedestal (only when --in-overscan is False)

