.. _howtouse_analysis_readoutnoise:

The process `RNvsNskipsPlot` studies the read out noise as a function of the number of averaged
samples. It will group the set of skip measurements in blocks of skips increasing in steps of 
**n_skips_per_block** to compute the standard desviation of the mean compressed image. 

If the input image is not a blank image, only the overscan region will be used. 

For an image with 500 skips grouped in blocks of 100, the readout noise will be computed for 5
images: 1) image with the first 100 skips, 2) image with the first 200 skips, 3) image with the
first 300 skips, ... and so on.

Note that `RNvsNskipsPlot` will call internally **CompressSkipperProcess** several times, so it is not
necessary to be included at **sequence**.


**attributes of the process**

Apart from the common parameters already listed in the previous section, the **user** has access to the following paramter:

    * **n-skips-per-block**: number of skips specifying the incrementation
    * **is-blank**: set if the input image is a blank image (if not, overscan region will be used)
    

**how to run**

How to run with and without a configuration file:

   * using the default values for all realted attributes

   .. code-block:: bash

      $ panaSKImg -s RNvsNskipsPlot data_image_name.fits --display

   * or setting through line commands

   .. code-block:: bash

      $ panaSKImg -s RNvsNskipsPlot data_image_name.fits --n-skips 100 --display

   * or by using the configuration json file for the parameter setting, and just set the list  of process by command line as follows

   .. code-block:: bash

      $ panaSKImg --json panaSKImg_configuration.json -s RNvsNskipsPlot data_image_name.fits --display


In all previous modes to run any common parameter (to select a set of skips, rows, or columns, ... ) can be also included.

**output**

One one figure is produced by this process:

    * The standard desviation or the MAD (for overscan) for a series of images groupped by increasing the number of skips (defined by the paramteter `n-skips-per-block`)


**output example**

    .. figure:: image/RNvsNskipsPlot_output.png
       :scale: 60

       Example of the output figure for the readout noise using block of 100 skips.
       

