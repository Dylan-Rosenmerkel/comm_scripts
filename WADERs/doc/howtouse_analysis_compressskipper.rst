.. _howtouse_analysis_compressskipper:

The process `CompressSkipperProcess` compress the single skip images into a unique one by applying a
function to all single skip measuremeants (by default the mean value).

Internally when the number of single skip measurements is higher than 1, the raw data image is
stored as an array of dimension 3 dimensions with shape

    .. math::
       
       image( N_{rows}, M_{cols}, K_{skips} )


This image is a data member of the raw data object (a **RawFits** class) under the name **image**. 

`CompressSkipperProcess` will create a new data member of the raw data object under the name
**image_<func>_compressed** where **<func>** is the function name used to reduce the set of single 
skip measurements (i.e. mean, std, ...).  This function must be by construciton a function from the 
python package **numpy**. This new data member **image_<func>_compressed** is a 2D image with shape 

    .. math::

       image( N_{rows}, M_{cols} )

where each single pixel of this image is the result of

     .. math::

        q_{i,j}^{'} = F( q^{k}_{i,j} | k=skip_{start}, ... , skip_{end} )


**attributes of the process**

Apart from the common parameters already listed in the previous section, the **user** has access to the following paramter:


    * **func-to-compress**: list of function names from the python package **numpy** (by default: ['mean']). For each 
      funciton in the list a compressed image will be created. 


**how to run**

How to run with and without a configuration file:

   * using the default values for all realted attributes

   .. code-block:: bash

      $ panaSKImg -s CompressSkipperProcess data_image_name.fits --display

   * or setting through line commands

   .. code-block:: bash

      $ panaSKImg -s CompressSkipperProcess data_image_name.fits --func-to-compress mean std median --display

   * or by using the configuration json file for the parameter setting, and just set the list  of process by command line as follows

   .. code-block:: bash

      $ panaSKImg --json panaSKImg_configuration.json -s ChargeLossPlot data_image_name.fits --display


In all previous modes to run any common parameter (to select a set of skips, rows, or columns, ... ) can be also included.

**output**

Two output products can be produced by this process:

    * A single plot with a compressed image per function listed in the attribute **func-to-compress**. This will be show up using --display and 
      recorded as an eps and pdf file if --save-plots was invoked.

    * If --save-img is also invoked, the images will be recorded as a fits file (and stored in the
      same folder were the input image is, or in another folder if the option --output is used)


**output example**

    .. figure:: image/CompressSkipperProcess_output.png

       Example of the output plot when runnning only CompressSkipperProcess. For this example three
       statistical functions were booked: mean, std and median.

       

