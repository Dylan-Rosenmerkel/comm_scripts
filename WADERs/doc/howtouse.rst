.. _howtouseit:

*************************
How to run `psimulCCDimg`
*************************

**psimulCCDimg** is a python3 script to apply a chain of processess (detector response and/or
reconstruction) over a geant4 simulations (output from `DAMICMG4 C++ code 
here <https://gitlab.in2p3.fr/damicm/DAMICM_G4Sims/tree/v1.0.0>`_), to mainly obtain a simulation
of the DAMIC-M detector.


**************
Running modes
**************
There are several ways to run `psimulCCDimg`:

	1. Production mode: to run on bash mode (several output messages are displayed)
	2. HELP mode: to display help on the optional arguments
	3. JSON mode: to display a description for any parameter on the configuration JSON file
	4. debug mode: to display an interactive plot with several information 

In the following sections more details on each running mode.

**1. Production mode**
**********************

1. All configuration parameters through the JSON file
 

    .. code-block:: bash

       $ psimulCCDimg <config_file.json>
 

2. All configuration parameters through the JSON file, except the geant4 root input file

    .. code-block:: bash

       $ psimulCCDimg <config_file.json> --g4file <geant4_file.root>


    2.1 In this case, more than one input root file can be informed

        .. code-block:: bash

           $ psimulCCDimg <config_file.json> --g4file  CCDSensor_PV_60a27z_*


    All files with the pattern started by `CCDSensor_PV_60a27z_` will be processes (sequentially). 


    2.2 If the geant4 root file is not under the working directory, you can use the option `--dir`

        .. code-block:: bash

           $ psimulCCDimg <config_file.json> --g4file <geant4_file.root> --dir <path_to_g4file>



Where 

	* **<config_file.json>** is an input configuration JSON file (see `json example <https://gitlab.in2p3.fr/damicm/pysimdamicm/blob/v1.0.0/pysimdamicm/config/psimulCCDimg_config_file.json>`_ for an example, or see an example below). In the example below, only **Diffusion**, **PixelizeSignal** and **ClusterFinder** are included in the full process chain: the attribute **active** is set to `True` (i.e. 1) for this three processess.
	
	* **<geant4_file.root>** is a geant4 simulation ROOT file. The input file name is also a configuration parameter in the JSON file (see line 5, in the JSON example below).



A tipical output should be

.. code-block:: bash
   :linenos:

   username@darkB612:~/workdir/LBC/test$ psimulCCDimg psimulCCDimg_config_file_full-image.json --g4file LBC_ColdCopper_1_PV_60a27z_s3_N5000.root
   
    Beginning new ROOT session
    OBJ: TStyle	Modern	Modern Style : 0 at: 0x55f9e02f2470
    WARNING: Parameter <PixelSaturation.units> will be ignored (Not implemented) 
    WARNING: Parameter <DarkCurrent.units> will be ignored (Not implemented) 
    WARNING: Parameter <Diffusion.units> will be ignored (Not implemented) 
    WARNING: Parameter <ElectronicNoise.units> will be ignored (Not implemented) 
    WARNING: Parameter <PixelizeSignal.units> will be ignored (Not implemented) 
    WARNING: Parameter <SignalPatternRecognition.units> will be ignored (Not implemented) 
    WARNING: Parameter <DarkCurrent.mode> will be ignored (Not implemented) 
    WARNING: Parameter <ElectronicNoise.mode> will be ignored (Not implemented) 
    WARNING: Parameter <ContinuousReadout.units> will be ignored (Not implemented) 
    Config.activate_configuration --- Add ClusterFinder process to the simulated process chain.
    Config.activate_configuration --- Add SignalPatternRecognition process to the simulated process chain.
    Config.activate_configuration --- Add ElectronicNoise process to the simulated process chain.
    Config.activate_configuration --- Add PixelizeSignal process to the simulated process chain.
    Config.activate_configuration --- Add Diffusion process to the simulated process chain.
    Config.activate_configuration --- Add DarkCurrent process to the simulated process chain.
   
    -- INFO. Input root file:  ./LBC_ColdCopper_1_PV_60a27z_s3_N5000.root
    -- INFO. Output root file:  ./testing_full-image_log02bf83e_LBC_ColdCopper_1_PV_60a27z_s3_N5000.root
        Set mode to `cropped-image`.
        Set mode to `full-image`.
    -- Generate Electronic Noise Image: mean=0.0,std=0.942
        Set mode to `cropped-image`.
        Set mode to `full-image`.
    -- Generate Dark Current Image: mean=0.001,std=0.069
   
    --- Loading geant4 data
         * CCDOut loaded (0.21 sec),
         * EventOut loaded (0.11 sec)
   
    --- Reconstruction process starts:
         * 30/5000 events  to process, 0.6%
   
   
    .......................................... event 552 (0% processed)
   
    .......................................... event 1714 (20% processed)
   
    .......................................... event 1924 (40% processed)
   
    .......................................... event 3413 (60% processed)
   
    .......................................... event 3883 (80% processed)
    Time statistics: 
     - Internal process (related to intrinsic noise): 
        - Mean time for clustering algorithm:  0.0025697052478790283 sec std= 0.003044839363409989  T= 0.0013705094655354817  min
        - Mean time for Filtering CCD image:   0.29295869171619415 sec std= 0.06273855824815325  T= 0.15624463558197021  min
     - Reconstruction processes done in 0.21 min (for processing 29 events)
   
    Output Root Rile: ./testing_full-image_log02bf83e_LBC_ColdCopper_1_PV_60a27z_s3_N5000.root
   
   
     -------------- Done!



**Lines 5-13**

	 Warnings to highlight that the units parameters (in each Process) is not used at all. This parameter is used to inform the units of each parameter (of the model, used to simulate each process).

**Lines 14-19**

	An output for each activated process will be displayed (in this example Diffusion, PixelizeSignal, DarkCurrent, ElectronicNoise, SignalPatterRecognition and ClusterFinder). Note that, the order do not corresponds to the executed order within the process chain.

**Lines 21-22**

	Outmput message with the name of the input file that will be processed, and its output file name

**Lines 23-29**

	In this example, intrinsic detector noise is also modeled by the convolution of DarkCurrent and ElectronicNoise. The mode by default is **cropped-image** (see later on). Line 23 correspond to the instance of the class, and 24 when the **mode** parameter of this class is set to the JSON file value (the same for DarkCurrent).

**Lines 30-33**

	Input ROOT file is loaded as DataFrame by using the python package uproot (time required to convert the ROOT format to DataFrane is displayed). For large files (~Gb) this can take a few minutes.

**Lines 34-47**

	Output messages to inform of the fraction of events (with hits) that have been processed

**Lines 47-51**
	
	Some statistics on time execution in different parts of the process (not really important for production, only for developers)

**Line 56**
	
	You will have this line only if simulations has been processed successfully

	



**2. HELP mode**
****************

.. code-block:: bash

   $ psimulCCDimg --help

Results in 

    .. code-block:: bash
       :linenos:
    
       username@darkB612:~/workdir/LBC/test$ psimulCCDimg --help
       usage: psimulCCDimg [-h] [--g4file G4FILE [G4FILE ...]] [--dir G4DIR]
                           [--debug] [--dparam DEBUGPARAM]
                           [--cls-pix-min CLS_PIX_MIN] [--dlevel DEBUG_LEVEL]
                           [--verbose] [--json]
                           jsonfile
       
       positional arguments:
         jsonfile              JSON configuration file. Run `psimulCCDimg --json
                               help` to list configuration parameters
       
       optional arguments:
         -h, --help            show this help message and exit
         --g4file G4FILE [G4FILE ...]
                               geant4 simulations root file to be processed,
                               root_file_name on json file
         --dir G4DIR           path of the geant4 simulations root file, and where
                               the output will also be recorded
         --debug               Increase output verbosity for debug mode (include
                               interactive plots for cluster reconstruction process)
         --dparam DEBUGPARAM   Parameter for the color code on the image plot, for
                               debug mode.
         --cls-pix-min CLS_PIX_MIN
                               [ONLY DEBUG MODE] Minimum pixel size to zoom in a
                               cluster.
         --dlevel DEBUG_LEVEL  [Deprecated] Level of the debug.
         --verbose             [Deprecated] Increase output verbosity.
         --json                Print description for each parameter on the JSON
                               configuration file.

  

**3. JSON mode**
**************** 
.. code-block:: bash

   $ psimulCCDimg --json

Results in 

    .. code-block:: bash
       :linenos:
 
       Beginning new ROOT session
       OBJ: TStyle	Modern	Modern Style : 0 at: 0x556c82f09520
       Valid options are: 
       "detector_response":
           "PixelizeSignal":
             "active" : Include the detector response and/or reconstruction process to the data acquisition process chain (by default  1)
             "N_pixels_in_x" : number of pixels along the x-axis (by default  4000)
             "shift_pixel_in_x" : skip the first `shift_pixel_in_x` pixels from the left (x-axis) (by default  0)
             "pixel_size_in_x" : pixel size along the x-axis in units of mm (by default  0.015)
             "N_pixels_in_y" : number of pixels along the y-axis (by default  6000)
             "shift_pixel_in_y" : skip the first `shift_pixel_in_y` pixels from the left (y-axis) (by default  0)
             "pixel_size_in_y" : pixel size along the y-axis in units of mm (by default  0.015)
           "ContinuousReadout":
             "pixel_read_time" : time to read a pixel in units of seconds (by default  0.001)
             "ampli" :  Readout: number of amplifiers used for skipper
       		 continous readout: 1/2/4 (by default  4)
             "active" : Include the detector response and/or reconstruction process to the data acquisition process chain (by default  0)
           "ElectronicNoise":
             "active" : Include the detector response and/or reconstruction process to the data acquisition process chain (by default  0)
             "img_size" : x and y-axis CCD region size. Both noises, dark current and electronic noise,
       		 are simulated in a reduced CCD region for computational reassons (by default  300)
             "min_enlarge" : minimum extra-pixel size when the elongation of the cluster is equal to img_size (by default  30)
             "pedestal" : mean value in units of electrons/pixel (by default  0.0)
             "sigma" : std value in units of electrons/pixel (by default  0.25)
           "Diffusion":
             "B" : coefficient for the xy dispersion measured by the 
       		 detector as a function of the depth (by default  0.000886)
             "active" : Include the detector response and/or reconstruction process to the data acquisition process chain (by default  1)
             "z_offset" : starting z-position in the silicon bulk (by default  0.0)
             "fanofactor" : fano factor to model e-h pair creation 
       		 within the silicon bulk (by default  0.16)
             "A" : coefficient for the xy dispersion measured by the 
       		 detector as a function of the depth (by default  216.2)
             "alpha" : Parameter for the second term on the diffusion model, the proportionality with the energy in units of pixel/eVee (by default  5.97e-05)
           "PixelSaturation":
             "active" : Include the detector response and/or reconstruction process to the data acquisition process chain (by default  0)
             "saturation" : Number of ADC units for a saturated pixel (by default  65535)
           "DarkCurrent":
             "min_enlarge" : minimum extra-pixel size when the elongation of the cluster is equal to img_size (by default  30)
             "img_size" : x and y-axis CCD region size. Both noises, dark current and electronic noise,
       		 are simulated in a reduced CCD region for computational reassons (by default  300)
             "exp_time" : time for the darkcurrent mesurement in units of days, being darkcurrent*exp_time
       		 the lambda parameter in the poisson distribution (by default  0.3333333333)
             "active" : Include the detector response and/or reconstruction process to the data acquisition process chain (by default  0)
             "darkcurrent" : dark current value in units of e-/pixel/day (by default  0.001)
       "in_root":
           "root_file_name" : ROOT file name of the geant4-based DAMICG4 simulation (by default  None)
           "in_dir_path" : path for <in_root.root_file_name> (by default  .)
       "out_root":
           "out_dir_path" : path for the output root file (by default  .)
           "save_CCDimg" : boolean option to store the image for each cluster in a npz format
       		 with two independent arrays: `energy` and `noise` for the pixel energy of the geant4 simulation and 
       		 the pixel noise distribution due to dark current and/or electronic noise. (by default  0)
           "prefix_root_name" : output file name prefix, the response of the detector 
       		 will be sotored as <dir_path>/<prefix_root_name>_<root_file_name>.root (by default  out_simulCCDimg_)
           "store_pixel_info" : If set, the properties of each pixel in the cluster will be stored in the output ROOT file (by default  1)
       "reconstruction":
           "SignalPatternRecognition":
             "threshold" : minimum pixel charge as signal (by default  10)
             "method" : algorithm number for cluster finder (by default  1)
             "active" : Include the detector response and/or reconstruction process to the data acquisition process chain (by default  0)
           "ClusterFinder":
             "active" : Include the detector response and/or reconstruction process to the data acquisition process chain (by default  0)
             "method" : algorithm number for cluster finder (by default  1)
             "max_nearest_neighbor" : maximum distance between two pixels with non-zero-charge that belong to the same cluster (by default  2)
       "CF":
           "ADC2eV" : convertion factor from ADC units to keV (by default  0.00026)
           "e2eV" : mean energy to create an e-h pair in the silicon 
       		 bulk at 140K (in units of keV) (by default  3.77)
       



**4. DEBUG mode**
*****************

.. code-block:: bash

   $ psimulCCDimg <config_file.json> --g4file <geant4_file.root> --debug

or 

.. code-block:: bash

   $ psimulCCDimg <config_file.json> --g4file <geant4_file.root> --debug --cls-pix-min 5


The parameter **--cls-pix-min** (see output of `psimulCCDimg --help`) should be used to set the minimum size of a cluster (number of pixels) to be displayed in the debuging figure. This has special importance when noise is simulated, to avoid to display clusters only from the noise.


The output message during the debug mode is the same as the one during production mode, with an extra message for each pair (event,ccd):


    .. code-block:: bash
       :linenos:
		
	    --- clusters for event,ccd=552,1

Furthermore, a figure (with matplotlib.pyplot) will be displayed. Here an example:  

	.. figure:: debug_mode_output_figure_ex.png


**subplot on the left**

	corresponds to simulated intrinsic noise (when applicable). In this figure several information is displayed:
		
		1. e-h pair creation as red small dots (if active)
		
		2. energy loss without apply any process (geant4 information)

		3. noise as a color-code image (with an image size depending on the runnning mode)

**subplots on the right**

	zoom in on each cluster with al leat `--cls-pix-min` pixels (by default 3). Clusters found with the ClusterFinder algorithm.



**zoom in**

Zooming in the subplot on the left, you will be able to see in more details

	.. figure:: debug_mode_output_figure_ex_zoomin.png


	* size of the markers is proportional to the depth (z-axis). 

	* small red dotted markers e-h pair created by Diffusion
	
	* color-code image  energy map of the total signal (noise+sims)

  

************************
Configuration JSON file
************************

The configuration JSON file should provide information regarding:
	
	1. inputs
		
		.. code-block:: json
		
		   {
		   	"in_root" :
           		{   
               		"in_dir_path" : ".",
               		"root_file_name": "None"
           		},  
		   ...

	2. outputs

		.. code-block:: json
		
		   ...
		   	"out_root" :
           		{   
					"out_dir_path" : ".",
					"prefix_root_name"  : "out_simulCCDimg_croppedImageMode",
					"store_pixel_info"  : 1,
					"save_CCDimg"       : 0
           		},  
		   ...

	3. Processes used to model the response of the detector (diffusion, pixelization and electronics)

		.. code-block:: json
		
		   ...
			"detector_response" :
				{
					... ## processes related with the detector response
				}
	

	4. Processes related with the cluster reconstruction 

		.. code-block:: json
		
		   ...
			"reconstruction" :
				{
					... ## processes related with the cluster reconstraction
				}
	
	5. General paramters (attributes of the singleton :obj:`pysimdamicm.utils.G4utils.Units`

	    .. code-block:: json

		   ...
	        "CF":
       			{
		           "ADC2eV": 0.26,
        		   "e2eV"  : 3.77,
        		   "detector_name": "CCDSensor_PV"
		       }

  	 Where the paramter **ADC2eV** is the convertion factor to go from ADU to eV; and **e2eV** is the average energy used to create an carried charge (e-h pair) within the sensitive detector (silicon bulk). The parameter **detector_name** is the name of the sensitive detector as appears in the input ROOT file (RunInfo TTRee).


JSON file Example
*****************

.. code-block:: json
   :linenos:

   {
       "in_root" : 
       {   
           "in_dir_path" : ".",
           "root_file_name": "None"
       },  
       "out_root":
       {   
           "out_dir_path" : ".",    
           "prefix_root_name"  : "testing_cropped-image",
           "store_pixel_info"  : 1,
           "save_CCDimg"       : 0
       },  
       "detector_response" : 
       {   
           "Diffusion":
           {
               "A"         : 216.2,
               "B"         : 0.000886,
               "z_offset"  : 0.0,
               "alpha"     : 0.0000597,
               "units"     : "A:um*um;B:1/um;z_offset:um;alpha:pixel/eVee",
               "fanofactor": 0.16,
               "active"    : 1 
           },
           "ContinuousReadout":
           {
               "pixel_read_time" : 0.001,
               "units"           : "s/pixel",
               "ampli"           : 4,
               "active"          : 0
           },
           "PixelizeSignal":
           {
               "N_pixels_in_x"    : 4000,
               "shift_pixel_in_x" : 0,
               "pixel_size_in_x"  : 0.015,
               "N_pixels_in_y"    : 6000,
               "shift_pixel_in_y" : 0,
               "pixel_size_in_y"  : 0.015,
               "units"             : "N_pixels_in_x:pixel;pixel_size_in_x:mm;shift_pixel_in_x:pixel;N_pixels_in_y:pixel;shift_pixel_in_x:pixel:shift_pixel_in_y:mm",
               "active"           : 1
           },
           "ElectronicNoise":
           {
               "mode"      : "cropped-image",
               "pedestal"  : 0.0,
               "sigma"     : 0.25,
               "units"     : "e/pixel;img_size:pixel;min_enlarge:pixel",
               "img_size"  : 300,
               "min_enlarge":30,
               "active"    : 1
           },
           "DarkCurrent":
           {
               "mode"        : "cropped-image",
               "darkcurrent" : 0.001,
               "exp_time"    : 0.3333333333,
               "img_size"    : 300,
               "min_enlarge" : 30,
               "units"       : "darkcurrent:e/pixel/day;exp_time:day;img_size:pixel;min_enlarge:pixel",
               "active"      : 1
           },
           "PixelSaturation":
           {
               "saturation" : 65535,
               "units"      : "saturation:ADC",
               "active"     : 0
           }
       },
       "reconstruction":
       {
           "SignalPatternRecognition":
           {
               "method"    : 1,
               "threshold" : 30,
               "units"     : "ADC",
               "active"    : 1
           },
           "ClusterFinder":
           {
               "method"               : 1,
               "max_nearest_neighbor" : 2,
               "active"               : 1
           }
       },
       "CF":
       {
           "ADC2eV": 0.26,
           "e2eV"  : 3.77,
           "detector_name":"CCDSensor_PV"
       }
   }

******************
Output ROOT file
******************

The output of `psimulCCDimg` will have at least three ROOT TTres, and if ClusterFinder is invoked one more tree.
	
	1. detector_config: only one entry, metadata for each invoked process
	2. geant4_config: only one entry, metadata from the geant4 root file (+ Nclusters)
	3. pixelizedEvent: one entry per event hitting at least one CCDs
	4. clustersRec: one entry per event hitting at least one CCD


detector_config TTree
**********************

A branch for each parameter of the invoked (in the JSON configuration file `active=1`) process will be added to this tree.
Those inactive processes will not have any entry on the tree. 


    .. code-block:: bash
       :linenos:
		
       >>> tree.detector_config.Show(0)
       ======> EVENT:0
        Diffusion_A     = 0.0002162
        Diffusion_B     = 0.886
        Diffusion_z_offset = 0
        Diffusion_fanofactor = 0.16
        Diffusion_alpha = 5.97e-05
        Diffusion_z_max = 0.669
        PixelizeSignal_shift_pixel_in_x = 0
        PixelizeSignal_N_pixels_in_x = 4000
        PixelizeSignal_pixel_size_in_x = 0.015
        PixelizeSignal_shift_pixel_in_y = 0
        PixelizeSignal_N_pixels_in_y = 6000
        PixelizeSignal_pixel_size_in_y = 0.015
        PixelSaturation_saturation = 65535
        ClusterFinder_method = 1
        ClusterFinder_max_nearest_neighbor = 2
        e2eV            = 3.77
        ADC2eV          = 0.26
       >>> tree.detector_config.GetEntries()
       1



geant4_config TTree
*******************

Several information from the simulated geometry is included in this tree: properties of the sensitive detecter (`ccd_XXX`), as well as from the simulated component (`sim_vol_XXX`). The latter attributes has sense only if the primary particle has been placed in the bulk or surface of a component (volume) of the simulated geant4 geometry.


Moreover, the number of events (`NEvts`), the number of CCDs (`NCCDs`), the useed random seed (`Seed`) and the primary particle (`primary` and/or `ion`) type.

The parameter `n_vols` corresponds to the multi-volume simulation mode available on DAMICG4 (our geant4 simulation code). In the example (see below), 6 volumes where simulated at the same time (see `sim_vol_pvnames`), so `n_vols=6`. 

Finally, `Nclusters`, the total number of clusters found for ClusterFinder is added to this tree. The main reason for that, is to check quickly if a simulation produced any signal without reading the main tree `clustersRec`. 


    .. code-block:: bash
       :linenos:
	
       >>> tree.geant4_config.Show(0)
       ======> EVENT:0
        NEvts           = 10000
        NCCDs           = 2
        Seed            = 321
        primary         = mu+
        ion             = 
        ccd_mass        = 0.0169857
        ccd_vol         = 7.29
        ccd_density     = 2.33
        ccd_surface     = 220.05
        sim_vol_pvnames = ColdCopper_1_PV;ColdCopper_2_PV;ColdCopper_3_PV;ColdCopper_4_PV;ColdCopper_5_PV;ColdCopper_6_PV
        n_vols          = 6
        sim_vol_density = 8.96, 
                         8.96, 8.96, 8.96, 8.96, 8.96
        sim_vol_mass    = 2.19933, 
                         0.17004, 0.92041, 2.19911, 2.00181, 2.53219
        sim_vol_volume  = 245.46, 
                         18.9776, 102.724, 245.437, 223.417, 282.611
        sim_vol_surface = 935.063, 
                         157.099, 1324.64, 235.619, 518.365, 743.176
        Nclusters       = 1865
       >>> tree.geant4_config.GetEntries()
       1

A more **standard output** will be 

    .. code-block:: bash
       :linenos:

       >>> tree.geant4_config.Show(0)
       ======> EVENT:0
        NEvts           = 5000
        NCCDs           = 2
        Seed            = 172
        primary         = ion
        ion             = 60a27z
        ccd_mass        = 0.0169857
        ccd_vol         = 7.29
        ccd_density     = 2.33
        ccd_surface     = 220.05
        sim_vol_pvnames = ColdCopper_1_PV
        n_vols          = 1
        sim_vol_density = 8.96
        sim_vol_mass    = 2.25263
        sim_vol_volume  = 251.41
        sim_vol_surface = 1878.88
        Nclusters       = 35
       >>> tree.geant4_config.GetEntries()
       1
       >>> 





pixelizedEvent TTree
********************

A more detailed description of the output can be found at :obj:`pysimdamicm.utils.data_formats.PixelizedEvent`


    .. code-block:: bash
       :linenos:

       >>> d.pixelizedEvent.Show(0)
       ======> EVENT:0
        event           = 11
        Npix            = (vector<int>*)0x31384b0
        ccd             = (vector<int>*)0x50744c0
        pixels_Edep     = (vector<vector<float> >*)0x515d0f0
        pixels_N_carried_charges = (vector<vector<int> >*)0x5209790
        pixels_pdg      = (vector<vector<float> >*)0x5204170
        pixels_sigma_xy = (vector<vector<float> >*)0x51fb160
        pixels_time     = (vector<vector<float> >*)0x51fca70
        pixels_x        = (vector<vector<int> >*)0x51fe3d0
        pixels_y        = (vector<vector<int> >*)0x54a9600
        pixels_z        = (vector<vector<float> >*)0x510fdc0
        pp_energy       = (vector<vector<float> >*)0x51fcf70
        pp_momx         = (vector<vector<float> >*)0x5211280
        pp_momy         = (vector<vector<float> >*)0x54ed300
        pp_momz         = (vector<vector<float> >*)0x5078730
        pp_posx         = (vector<vector<float> >*)0x432e070
        pp_posy         = (vector<vector<float> >*)0x51f7df0
        pp_posz         = (vector<vector<float> >*)0x54e8c60
       >>> d.pixelizedEvent.GetEntries()
       324
       

clustersRec TTree
******************

A more detailed description of the output can be found 
at :obj:`pysimdamicm.utils.data_formats.Cluster`. One entry per event, and within this 
one entry per cluster.

In summary, there for each cluster there are several statistic (min, max, std, average, RMS, ...) to define the position 
of the cluster (mean pixel coordenate, energy-weighted pixel position average, ...) as 
well as, the depth (z-axis) of the cluster within the silicon bulk (or sensitive area). 
Other parameters like the elongation (DX) or the pixel where the maximum energy loss take place (QmaxX, being 
Qmax as the maximum pixel energy loss for the given cluster).

For the time several statistic have also been used.


Some pixel information is keeped to track the signature of the cluster (position, energy, time and PDF: `pixels_XXXX`).

Finally, information from the primary particle is also included in this tree (branches startint with the name `pp_XXXX`).


    .. code-block:: bash
       :linenos:

       >>> d.clustersRec.Show(0)
       ======> EVENT:0
        event           = 11
        Nclusters       = 5
        DX              = (vector<float>*)0x39aa230
        DY              = (vector<float>*)0x4b2ac20
        DZ              = (vector<float>*)0x486f970
        Energy          = (vector<float>*)0x4878df0
        Npix            = (vector<int>*)0x4b1b160
        PosX            = (vector<float>*)0x4b29a80
        PosY            = (vector<float>*)0x4b2eff0
        PosZ            = (vector<float>*)0x4b34530
        Qmax            = (vector<float>*)0x4885730
        QmaxX           = (vector<float>*)0x487f750
        QmaxY           = (vector<float>*)0x4b349b0
        QmaxZ           = (vector<float>*)0x4b21940
        RMSX            = (vector<float>*)0x4b2a990
        RMSY            = (vector<float>*)0x2b12300
        RMSZ            = (vector<float>*)0x4b20490
        ccd             = (vector<int>*)0x487cf80
        cluster_id      = (vector<int>*)0x48905c0
        maxX            = (vector<float>*)0x4b1ce20
        maxY            = (vector<float>*)0x48702a0
        maxZ            = (vector<float>*)0x488e390
        meanTime        = (vector<float>*)0x487f150
        meanX           = (vector<float>*)0x4b1f9e0
        meanY           = (vector<float>*)0x487e830
        meanZ           = (vector<float>*)0x4b2c580
        minTime         = (vector<float>*)0x4b2d460
        minX            = (vector<float>*)0x486d780
        minY            = (vector<float>*)0x4878270
        minZ            = (vector<float>*)0x4877fd0
        pixels_E        = (vector<vector<float> >*)0x4da7c80
        pixels_time     = (vector<vector<float> >*)0x4dcb2d0
        pixels_x        = (vector<vector<float> >*)0x4dc7700
        pixels_y        = (vector<vector<float> >*)0x4cc0db0
        pixels_z        = (vector<vector<float> >*)0x4e26780
        pp_energy       = (vector<vector<float> >*)0x4c7a7e0
        pp_momx         = (vector<vector<float> >*)0x4de3430
        pp_momy         = (vector<vector<float> >*)0x4de3f20
        pp_momz         = (vector<vector<float> >*)0x4d06070
        pp_posx         = (vector<vector<float> >*)0x4c91f50
        pp_posy         = (vector<vector<float> >*)0x4e61530
        pp_posz         = (vector<vector<float> >*)0x4c7a4b0
        primary_part    = (vector<vector<float> >*)0x4e650e0
        timeQmax        = (vector<float>*)0x4e37880
        wTime           = (vector<float>*)0x4ecbd40
       >>> d.clustersRec.GetEntries()
       324
       >>> 



********************************************
Modes to mimic the Intrinsic Detector Noise
********************************************

The intrinsic detector noise is the convolution of a Dark current noise and the noise due to electronics (readout), modeled by a poisson and gaussian distribution, respectively. Use the full CCD size is computationally inefficient.

Both processes can be run under two different modes:
	
	1. cropped-image mode

	2. full-image mode 


cropped-image mode
******************

An squared image with smaller size than the full detector active region will be generated to mimic the intrinsic detector noise being the cluster in the center of the image (cropper image for the cluster). 

This image will be randomly recreated for each event hits collection. The cluster is placed in the center of the image, except for those cases
where the cluster within the full size do not have a pad (on the left or right) with the haph size of the desired cropped image (clusters on the corners are then respected).

It is the fastest mode.

.. code-block:: json
   :linenos:

   "ElectronicNoise":
		{
		 "mode"        : "cropped-image",
		 "pedestal"    : 0.0,
         "sigma"       : 0.25,
		 "img_size"    : 300,
		 "min_enlarge" : 30,
         ...
		}
   "DarkCurrent":
		{
		 "mode"        : "cropped-image",
		 "darkcurrent" : 0.001,
         "exp_time"    : 0.3333333333,
		 "img_size"    : 300,
		 "min_enlarge" : 30,
		...
		}




full-image mode
***************

The shape for the intrinsic detector noise image corresponds to the same as the active region of the detector (:obj:`pysimdamicm.utils.G4utils.Units.ccd_shape`).

In this case the parameters **img_size** and **min_enlarge** will be ignored.


Is slower than the other running mode, specially if noise is simulated and no effective SignalPatternRecognition is used (too many clusters from noise are included in the output).


.. code-block:: json
   :linenos:

   "ElectronicNoise":
		{
	     mode = "full-image"
         ...
		}



