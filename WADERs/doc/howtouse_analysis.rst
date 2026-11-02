.. _howtouse_analysis:

******************************
What is `panaSKImg` used for?
******************************

**panaSKImg** is a python3 script to apply a chain of processess (skipper images test for
commissioning, analysis and/or reconstruction) over a real data CCD image. The following analysis
are already available:

    1. **ChargeLossPlot**: Study of the loss of pixel charge between skips
    2. **FFTNoisePlot**: Study FFT of skips and pixels
    3. **RNvsNskipsPlot**: Study noise vs n. skips
    4. **FitCalibrationConstant**: Measure linearity with single electron peaks
    5. **FitDarkCurrentProcess**: Measure dark current
    6. **CompressSkipperProcess**: Compress a set of single skip measurements into a single image using an statistical function
	7. **PedestalSubtractionProcess**:  Subtract the pedestal: row by row, column by column, using overscan only, ... 
	8. Reconstruction: signal patter recognition and cluster finder (same as the one used with simulations)


There are several ways to run **panaSKImg** script, 

    * by using a json file to set all needed parameters (see example at the end), 

        .. code-block:: bash

           $ panaSKImg --json panaCCDimg_configuration.json data_image_name.fits

    * or just by using several comman line options to explicitally add a sequence of processes and set
      (if needed) the corresponding optional attributes of each booked process. The parameters available
      for each process are explained process by process in the following sections.

        .. code-block:: bash

           $ panaSKImg -s ChargeLossPlot;CompressSkipperProcess  data_image_name.fits


**Input JSON file**
*******************

The json file has two different main sections:

    * input: parameters related with the input data. **image** referes to those paramters needed to
      understand which part of the data corresponds to the sensitive detector, overscan region and
      prescan regions; it has also a set of paramters that can be used to confine a specific region
      of the CCD to be used during execution. **onvention** applies to those parameters that must be
      read from the fits file header: number of skips, number of pixels per axis, amplifier,
      exposure time, readout time, as well as the binning in each axis used to read the image.

    .. code-block:: python

	   {
  		"input":
  		{
  		    "image":
  		    {
  		        "extensions" :0, 
  		        "skip_image":true,
  		        "axis_to_compress":1,
  		        "correct_leach_bug":true,
  		        "correct_polarity":false,
  		        "id_skip_start":0,
  		        "id_skip_end":-1,
  		        "id_row_start":0,
  		        "id_row_end":-1,
  		        "id_col_start":0,
  		        "id_col_end":-1,
  		        "n_rows_overscan":0,
  		        "n_rows_prescan" :0, 
  		        "n_cols_overscan":69,
  		        "n_cols_prescan" :9, 
  		        "active_region_rows":null,
  		        "active_region_cols":null
  		        },  
  		    "scp":
  		    {
  		        },  
  		    "convention":
  		    {
  		        "Nskips":"NDCMS",
  		        "Ncols":"NAXIS1",
  		        "Nrows":"NAXIS2",
  		        "Npbin":"NPBIN",
  		        "Nsbin":"NSBIN",
  		        "ampl":"AMPL",
  		        "exposure_time":"MEXP",
  		        "read_time":"MREAD"
  		        }   
  		},  
       ...


    * process: there are a section for each processes to properly set their attributes. The attribute 
      **sequence** is used to define the list of processes (using semicolon) that must be executed and in what order
      it should be done. `If the process is not included in the **sequence** atribute, despite it is
      listed in the json file will not be executed!`

    .. code-block:: python

	   {
       		 "sequence":"ChargeLossPlot;CompressSkipperProcess;PedestalSubtractionProcess;FitCalibrationConstant;FitDarkCurrentProcess",
       		 "CompressSkipperProcess":
       		 {
       		     "func_to_compress":["mean"],
       		     "save_image":true
       		 },
       		 "PedestalSubtractionProcess":
       		 {
       		     "image":"mean_compressed",
       		     "method":"gauss_fit",
       		     "in_overscan":true,
       		     "axis":"full",
       		     "n_sigma_win_fit":6,
       		     "n_sigma_to_mask":3,
       		     "histequ":true,
       		     "show_fit":true,
       		     "save_image":true,
       		     "save_plots":false
       		 },
       		 "FitCalibrationConstant":
       		 {
       		     "image":"pedestal_subtracted",
       		     "n_peaks":8,
       		     "calibration":10,
       		     "n_sigma_win_fit":3,
       		     "save_plots":true
       		 },
       		 "FitDarkCurrentProcess":
       		 {
       		     "image":"pedestal_subtracted",
       		     "method":"root",
       		     "do_calibration":false,
       		     "n_peaks":2,
       		     "n_sigma_fit":2,
       		     "mu_gauss":0.0,
       		     "sigma_gauss":0.2,
       		     "lambda_poisson":0.05,
       		     "fit_options":"QSL",
       		     "save_as":true
       		 },
       		 "ChargeLossPlot":
       		 {
       		     "skip_id_list":[0,1,2],
       		     "skip_id_baseline":-1,
       		     "histequ":false,
       		     "save_plots":true,
       		     "gray_palette":false
       		 },
       		 "FFTNoisePlot":
       		 {
       		     "save_plots":true
       		 },
       		 "RNvsNskipsPlot":
       		 {
       		     "n_skips_per_block":100,
       		     "is_blank":false,
       		     "include_row_dependency":false,
       		     "save_plots":true
       		 },
       		 "SignalPatternRecognition":
       		 {
       		     "method"    : 1,
       		     "image"     : "mean_compressed",
       		     "isdata"    : true,
       		     "threshold" : 1200,
       		     "units"     : "ADC"
       		 },
       		 "ClusterFinder":
       		 {
       		     "method"               : 1,
       		     "max_nearest_neighbor" : 1
       		 }
	   }



**General command lines**
*************************

**reduced image region: rows, cols, skips**

There are some coomon parameters to properly select only a portion of the full data taken: set of
columns, rows and/or single skips:

    * **id-skip-start**, **id-skip-end**: first and last skip index to take into account (use -1 to indicate the
      last taken skip)
    * **id-col-start**, **id-col-end**: first and last column index to take into account (use -1 to indicate the 
      last column) 
    * **id-row-start**, **id-row-end**: first and last row index to take into account (use -1 to indicate
      the last row)


example:

    .. code-block:: bash

       $ panaSKImg --json panaCCDimg_configuration.json data_image_name.fits --id-skip-start 10

In this example, the single skip measurements from 0 to 9, will not be considered and will not be
included in the analysis. **In this case, all booked process will share the same image
region despite its value in the json file.**

If you want to define a region for each specific process, these parameters can also be included to any 
process inside the json file: just replace the
character **-** by **_**, i.e. use **id_skip_start** when is included in the json input file (the
same for the rest, see json file section).



**save outputs**

Both plots and images can be recorded with the following boolean attributes

    * **save-img**: set to record the intermediate images as fits
    * **save-plots**: set to save all displayed plots as eps/pdf
    * **output**: where outputs will be stored

example:

    .. code-block:: bash

       $ panaSKImg --json panaCCDimg_configuration.json --id-skip-start 10  --save-img --save-plots -o <path_to_my_folder> data_image_name.fits

**display plots**

Finally, there are two more optional parameters to set/unset plots and output messages:

    * **display**: will show up all executed plots (before ending)
    * **verbose**: will display some debugging plots and messages

example:

    .. code-block:: bash

       $ panaSKImg --json panaCCDimg_configuration.json --id-skip-start 10  --display --verbose

**invert image**

Some of the images were taken with inverted polarity, which means that the pixel changer (in ADUs)
is inversely proportional to the ionizing pixel charge. The paramter **invert** has been also
included as a command line


     .. code-block:: bash

        $ panaSKImg --json panaCCDimg_configuration.json data_image_name.fits --id-skip-start 10 --invert


*******************
Process by Process
*******************

All processes can be run by using an input configuration json file (as the one shown at the begining), 
where the attribute **sequence** is used to list all processes the used want to book. In the json file 
there are a section (dictionary) for each processes where the relevant process attributes can be modified 
by the user.

**available processes**

Apart from this `modis operandi`, each process can be invoked by pure command lines. For that the user 
must known the names of the available processes. This can be done 

     .. code-block:: bash

        $ panaSKImg --list-process help

the process names will be listed

     .. code-block:: bash
        :linenos:

		ChargeLossPlot
		ClusterFinder
		CompressSkipperProcess
		ContinuousReadout
		DarkCurrent
		Diffusion
		ElectronicNoise
		FFTNoisePlot
		FitCalibrationConstant
		FitDarkCurrentProcess
		PedestalSubtractionProcess
		PixelSaturation
		PixelizeSignal
		RNvsNskipsPlot
		SignalPatternRecognition


**1. CompressSkipperProcess**
*****************************

   .. include:: howtouse_analysis_compressskipper.rst

**2. PedestalSubtractionProcess**
*********************************

   .. include:: howtouse_analysis_pedestalsubtraction.rst

**3. FitDarkCurrentProcess**
****************************

   .. include:: howtouse_analysis_fitdc.rst

**3. ChargeLossPlot**
*********************

   .. include:: howtouse_analysis_chargeloss.rst

**4. FFTNoisePlot**
*******************

   .. include:: howtouse_analysis_fftnoise.rst

**5. RNvsNskipsPlot**
*********************

   .. include:: howtouse_analysis_readoutnoise.rst

**6. FitCalibrationConstant**
*****************************

   .. include:: howtouse_analysis_fitcalibration.rst

*****************************
gev cluster: How to use it
*****************************

The official repository of pysimdamicm can be found at

    .. code-block:: bash

       /data/official_repos/pysimdamicm

There are several ways to use it:

    * **use the already installed version as a virtual environment**. Within
      `/data/official_repos/pysimdamicm` a folder under the name `venv_analysis` has a version
      already installed in a virtual environment. So, any user with an account at gev can use this
      already installed version. Follow the next steps:

        * Include the path where pysimdamim has been installed to the environment variable **PATH**.
          This can be done by adding this three lines in the `.bashrc` (or equivalent,
          `.bash_profile`, ...). So any time you access gev, this step will be already done.

            .. code-block:: bash
               :linenos:

               export PATH=${PATH}:/data/official_repos/pysimdamicm/venv_analysis/bin
               export PATH=${PATH}:/data/official_repos/pysimdamicm/venv_analysis/lib

               export PYTHONPATH=$PYTHONPATH:/data/official_repos/pysimdamicm/lib

        * Activate the virtual environment by running. 

            .. code-block:: bash

               source /data/official_repos/pysimdamicm/venv_analysis/bin/activate

         you will go from 

            .. code-block:: bash

               [castello@gev ~]$
         to 

            .. code-block:: bash

               (venv_analysis) [castello@gev ~]$


        To get out from this mode just type

            .. code-block:: bash

               (venv_analysis) [castello@gev ~]$ deactivate


    * install your own version in your `.local` home:

        * just go into `/data/official_repos/pysimdamicm` and run

            .. code-block:: bash

               source install_user.py

        All required packages as well as pysimdamicm will be installed at `${HOME}/.local`. In order
        to be able to see this package from any directory, you should include this directory to the
        environment variable **PATH**. Add this lines to your  `.bashrc` file (or equivalent, `.bash_profile`, ...)

            .. code-block:: bash
               :linenos:

               export PATH=${PATH}:${HOME}/.local/lib
               export PATH=${PATH}:${HOME}/.local/bin

               export PYTHONPATH=$PYTHONPATH:${HOME}/.local/lib


