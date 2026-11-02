.. _howtouse_analysis_chargeloss:


The process `ChargeLossPlot` is mainly used to sudy the non-conservative charge between skips. 
A set of images will be displayed to search for ghost pixels around tracks: zoom into any cluster 
to see if there is a charge loss/gain, i.e. black or white ghost track pixels.


**attributes of the process**

Apart from the common parameters already listed above, the **user** has access to the following paramters:


    * **skip_id_list**: list of single skip measurements (by default: [0,1,2]). For each skip index in this list 
      an image with only this measurement per pixel will be shown. 

    * **skip_id_baseline**: the index of one of the single skip measurements to be used as baseline (image of
      reference). This baseline image will be subtracted to each single skip images and also
      displayed. 

    * **histequ**: set if the images should be equalized. This is mostly to improve contrast between pixels 
      of low charge (similar to what ds9 does).

    * **gray_palette**: to use a grey palette instead of the viridis one (by default is color code).


**how to run**

How to run with and without a configuration file:

   * using the default values for all realted attributes

   .. code-block:: bash

      $ panaSKImg -s ChargeLossPlot data_image_name.fits --display 

   * or setting through line commands

   .. code-block:: bash

      $ panaSKImg -s ChargeLossPlot --skip-id-list 0 2 14 --skip-id-baseline 50 data_image_name.fits  --histequ --gray-palette --display

   * or by using the configuration json file for the parameter setting, and just set the list  of process by command line as follows

   .. code-block:: bash

      $ panaSKImg --json panaSKImg_configuration.json -s ChargeLossPlot data_image_name.fits --display

In all previous modes to run any common parameter (to select a set of skips, rows, or columns, ... ) can be also included.

**output**

Several images and distributions will be shown.

    * A set of **single skip images**. In this example, an image for the 1st, 3rd and 15th
      measurement is shown (note that the skip id number starts with 0).
    * The **baseline image**. In this example, the iamge of reference is the 51st skip measurement.
    * For each single skip images a **baseline-subtracted image** will be also displayed.
    * The mean of the pixel charge distribution for each single skip image: 
      
            :math:`\langle q_{i,j}^{skip} \rangle |_{skip}`

    * The standard deviation of the pixel charge distribution for each single skip image: 
      
            :math:`std(q_{i,j}^{skip})|_{skip}`

    * The mean of the baseline-subtracted pixel charge for each single skip image: 
      
            :math:`\langle q_{i,j}^{skip} - q_{i,j}^{baseline} \rangle |_{skip}`

    * If **CompressSkipperProcess** was also booked, it will also show two more images: the compressed 
      image by using the mean pixel value and its standard deviation.


**output example**

    .. figure:: image/ChargeLossPlot_output.png

       Example of the output plot when runnning only ChargeLossPlot.

       

